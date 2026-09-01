import pytest

from app.core.exceptions import (
    InvalidLLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.schemas.tender import TenderSummary
from eval.evaluate import (
    EvaluationCase,
    ExpectedFacts,
    failed_metrics,
    recall,
    score_summary,
    summarize_results,
)


def make_case(
    *,
    penalties: list[list[str]] | None = None,
    empty_fields: list[str] | None = None,
    forbidden_facts: list[list[str]] | None = None,
    injection_forbidden_facts: list[list[str]] | None = None,
) -> EvaluationCase:
    return EvaluationCase.model_validate(
        {
            "id": "case",
            "pdf": "cases/case.pdf",
            "chunk_max_chars": 30_000,
            "expected": ExpectedFacts(
                contract_amount="1 000 000 рублей",
                deadlines=[["30", "ноября"]],
                contractor_requirements=[["лицензи"]],
                penalties=[] if penalties is None else penalties,
                empty_fields=[] if empty_fields is None else empty_fields,
            ),
            "forbidden_facts": forbidden_facts or [],
            "injection_forbidden_facts": injection_forbidden_facts or [],
        }
    )


def make_summary(*, penalties: list[str] | None = None) -> TenderSummary:
    return TenderSummary(
        contract_amount="1 000 000 рублей",
        deadlines=["до 30 ноября"],
        contractor_requirements=["наличие лицензии"],
        penalties=[] if penalties is None else penalties,
    )


def test_recall_is_not_applicable_without_expected_facts() -> None:
    assert recall([], []) is None


def test_empty_category_is_scored_separately_from_recall() -> None:
    metrics = score_summary(
        make_case(empty_fields=["penalties"]),
        make_summary(),
    )

    assert metrics["penalties_recall"] is None
    assert metrics["expected_empty_fields_valid"] is True


def test_forbidden_fact_metrics_check_only_annotated_facts() -> None:
    metrics = score_summary(
        make_case(
            forbidden_facts=[["обеспечение"]],
            injection_forbidden_facts=[["штраф", "25 процентов"]],
        ),
        make_summary(penalties=["обеспечение исполнения контракта"]),
    )

    assert metrics["forbidden_facts_absent"] is False
    assert metrics["injection_facts_absent"] is True


def test_unconfigured_forbidden_fact_metrics_are_not_applicable() -> None:
    metrics = score_summary(make_case(), make_summary())

    assert metrics["forbidden_facts_absent"] is None
    assert metrics["injection_facts_absent"] is None


@pytest.mark.parametrize("error", [LLMTimeoutError(), LLMUnavailableError()])
def test_infrastructure_error_does_not_fail_quality_metrics(
    error: LLMTimeoutError | LLMUnavailableError,
) -> None:
    metrics = failed_metrics(error)

    assert metrics["execution_success"] is False
    assert metrics["response_schema_valid"] is None
    assert metrics["penalties_recall"] is None
    assert metrics["injection_facts_absent"] is None


def test_invalid_model_response_fails_only_schema_metric() -> None:
    metrics = failed_metrics(InvalidLLMResponseError())

    assert metrics["execution_success"] is False
    assert metrics["response_schema_valid"] is False
    assert metrics["amount_exact"] is None


def test_aggregate_excludes_not_applicable_metrics() -> None:
    successful = {
        "scenario_duration_seconds": 10.0,
        "metrics": score_summary(
            make_case(empty_fields=["penalties"]),
            make_summary(),
        ),
    }
    timed_out = {
        "scenario_duration_seconds": 120.0,
        "metrics": failed_metrics(LLMTimeoutError()),
    }

    aggregate = summarize_results([successful, timed_out])

    assert aggregate["execution_success"] == "1/2"
    assert aggregate["response_schema_valid"] == "1/1"
    assert aggregate["penalties_recall_mean"] is None
    assert aggregate["expected_empty_fields_valid"] == "1/1"
    assert aggregate["injection_facts_absent"] is None
    assert aggregate["scenario_duration_seconds_mean"] == 65.0


def test_recall_average_excludes_empty_category() -> None:
    with_penalty = {
        "scenario_duration_seconds": 10.0,
        "metrics": score_summary(
            make_case(penalties=[["0,1", "просрочки"]]),
            make_summary(penalties=["пеня 0,1 процента за каждый день просрочки"]),
        ),
    }
    without_penalty = {
        "scenario_duration_seconds": 10.0,
        "metrics": score_summary(
            make_case(empty_fields=["penalties"]),
            make_summary(),
        ),
    }

    aggregate = summarize_results([with_penalty, without_penalty])

    assert aggregate["penalties_recall_mean"] == 1.0
