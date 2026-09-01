import argparse
import asyncio
import json
import os
import platform
import shutil
import subprocess
import unicodedata
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.adapters.ollama import OllamaTenderExtractor
from app.adapters.pdf import PDFExtractor
from app.core.config import Settings
from app.core.exceptions import ApplicationError, InvalidLLMResponseError
from app.schemas.document import UploadedDocument
from app.schemas.tender import TenderSummary
from app.services.chunking import PageChunker
from app.services.document_validation import DocumentValidator
from app.services.tender_summary import TenderSummaryService

BASE_DIR = Path(__file__).resolve().parent
SUMMARY_FIELDS = (
    "deadlines",
    "contractor_requirements",
    "penalties",
)


class ExpectedFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_amount: str | None
    deadlines: list[list[str]]
    contractor_requirements: list[list[str]]
    penalties: list[list[str]]
    empty_fields: list[Literal["deadlines", "contractor_requirements", "penalties"]]


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    pdf: str
    chunk_max_chars: int
    expected: ExpectedFacts
    forbidden_facts: list[list[str]]
    injection_forbidden_facts: list[list[str]]


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    cases: list[EvaluationCase]


def normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    return " ".join(
        "".join(
            character if character.isalnum() else " " for character in normalized
        ).split()
    )


def contains_fact(values: list[str], keywords: list[str]) -> bool:
    normalized_keywords = [normalize(keyword) for keyword in keywords]
    return any(
        all(keyword in normalize(value) for keyword in normalized_keywords)
        for value in values
    )


def recall(expected: list[list[str]], actual: list[str]) -> float | None:
    if not expected:
        return None
    matched = sum(contains_fact(actual, keywords) for keywords in expected)
    return round(matched / len(expected), 3)


def flattened_values(summary: TenderSummary) -> list[str]:
    values = [summary.contract_amount] if summary.contract_amount is not None else []
    for field in SUMMARY_FIELDS:
        values.extend(getattr(summary, field))
    return values


def facts_absent(summary: TenderSummary, facts: list[list[str]]) -> bool:
    values = flattened_values(summary)
    return all(not contains_fact(values, keywords) for keywords in facts)


def score_summary(case: EvaluationCase, summary: TenderSummary) -> dict[str, object]:
    expected = case.expected
    expected_amount = expected.contract_amount
    actual_amount = summary.contract_amount
    amount_exact = (
        actual_amount is None
        if expected_amount is None
        else actual_amount is not None
        and normalize(actual_amount) == normalize(expected_amount)
    )
    empty_fields_valid = (
        all(not getattr(summary, field) for field in expected.empty_fields)
        if expected.empty_fields
        else None
    )
    forbidden_facts_absent = (
        facts_absent(summary, case.forbidden_facts) if case.forbidden_facts else None
    )
    injection_facts_absent = (
        facts_absent(summary, case.injection_forbidden_facts)
        if case.injection_forbidden_facts
        else None
    )
    return {
        "execution_success": True,
        "response_schema_valid": True,
        "amount_exact": amount_exact,
        "deadline_recall": recall(expected.deadlines, summary.deadlines),
        "requirements_recall": recall(
            expected.contractor_requirements,
            summary.contractor_requirements,
        ),
        "penalties_recall": recall(expected.penalties, summary.penalties),
        "expected_empty_fields_valid": empty_fields_valid,
        "forbidden_facts_absent": forbidden_facts_absent,
        "injection_facts_absent": injection_facts_absent,
    }


def failed_metrics(error: ApplicationError) -> dict[str, object]:
    return {
        "execution_success": False,
        "response_schema_valid": (
            False if isinstance(error, InvalidLLMResponseError) else None
        ),
        "amount_exact": None,
        "deadline_recall": None,
        "requirements_recall": None,
        "penalties_recall": None,
        "expected_empty_fields_valid": None,
        "forbidden_facts_absent": None,
        "injection_facts_absent": None,
    }


def build_service(model: str, chunk_max_chars: int) -> TenderSummaryService:
    settings = Settings(
        _env_file=None,
        ollama_model=model,
        llm_chunk_max_chars=chunk_max_chars,
    )
    return TenderSummaryService(
        validator=DocumentValidator(settings.max_upload_size_mb),
        pdf_extractor=PDFExtractor(settings.max_pdf_pages),
        chunker=PageChunker(
            max_chars=settings.llm_chunk_max_chars,
            overlap_pages=settings.chunk_overlap_pages,
        ),
        tender_extractor=OllamaTenderExtractor(settings),
    )


async def evaluate_case(model: str, case: EvaluationCase) -> dict[str, object]:
    pdf_path = BASE_DIR / case.pdf
    document = UploadedDocument(
        filename=pdf_path.name,
        content_type="application/pdf",
        content=pdf_path.read_bytes(),
    )
    service = build_service(model, case.chunk_max_chars)
    started_at = perf_counter()
    try:
        summary = await service.summarize(document)
    except ApplicationError as exc:
        return {
            "case": case.id,
            "scenario_duration_seconds": round(perf_counter() - started_at, 3),
            "metrics": failed_metrics(exc),
            "error": type(exc).__name__,
            "summary": None,
        }

    return {
        "case": case.id,
        "scenario_duration_seconds": round(perf_counter() - started_at, 3),
        "metrics": score_summary(case, summary),
        "error": None,
        "summary": summary.model_dump(mode="json"),
    }


def passed(results: list[dict[str, object]], metric: str) -> str | None:
    values = [result["metrics"][metric] for result in results]
    applicable = [value for value in values if isinstance(value, bool)]
    if not applicable:
        return None
    return f"{sum(applicable)}/{len(applicable)}"


def average(results: list[dict[str, object]], metric: str) -> float | None:
    values = [result["metrics"][metric] for result in results]
    applicable = [float(value) for value in values if isinstance(value, float)]
    return round(mean(applicable), 3) if applicable else None


def summarize_results(results: list[dict[str, object]]) -> dict[str, object]:
    return {
        "execution_success": passed(results, "execution_success"),
        "response_schema_valid": passed(results, "response_schema_valid"),
        "amount_exact": passed(results, "amount_exact"),
        "deadline_recall_mean": average(results, "deadline_recall"),
        "requirements_recall_mean": average(results, "requirements_recall"),
        "penalties_recall_mean": average(results, "penalties_recall"),
        "expected_empty_fields_valid": passed(
            results,
            "expected_empty_fields_valid",
        ),
        "forbidden_facts_absent": passed(results, "forbidden_facts_absent"),
        "injection_facts_absent": passed(results, "injection_facts_absent"),
        "scenario_duration_seconds_mean": round(
            mean(float(result["scenario_duration_seconds"]) for result in results),
            3,
        ),
    }


def command_output(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        return subprocess.run(
            command,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except subprocess.SubprocessError:
        return None


def environment_details() -> dict[str, object]:
    memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return {
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "memory_gib": round(memory_bytes / 1024**3, 1),
        "gpu": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "ollama_version": command_output(["ollama", "--version"]),
        "ollama_timeout_seconds": Settings(_env_file=None).ollama_timeout_seconds,
        "temperature": 0,
    }


async def run(models: list[str], dataset: EvaluationDataset) -> dict[str, object]:
    model_results = []
    for model in models:
        results = [await evaluate_case(model, case) for case in dataset.cases]
        model_results.append(
            {
                "model": model,
                "aggregate": summarize_results(results),
                "cases": results,
            }
        )
    return {
        "dataset_version": dataset.version,
        "environment": environment_details(),
        "models": model_results,
        "limitations": (
            "Пять синтетических примеров позволяют сравнить модели, но такой выборки "
            "недостаточно для оценки качества на реальных закупках. "
            "Результаты проверки по ключевым словам требуют ручного анализа."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        add_help=False,
        description="Оценка качества извлечения сведений из тендерных документов.",
    )
    parser._optionals.title = "параметры"
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="Показать справку и завершить работу.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Теги установленных моделей Ollama для сравнения.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=BASE_DIR / "expected.json",
        help="Путь к файлу с ожидаемыми фактами.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "results.json",
        help="Путь для сохранения результатов оценки.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = EvaluationDataset.model_validate_json(args.dataset.read_text())
    results = asyncio.run(run(args.models, dataset))
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
