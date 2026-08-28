import json

import pytest
from pydantic import ValidationError

from app.schemas.tender import TenderSummary


def test_serializes_tender_summary() -> None:
    summary = TenderSummary(
        contract_amount="14 850 000 рублей",
        deadlines=["до 30 ноября 2026 года"],
        contractor_requirements=["наличие действующей лицензии"],
        penalties=["пеня за каждый день просрочки"],
    )

    assert json.loads(summary.model_dump_json()) == {
        "contract_amount": "14 850 000 рублей",
        "deadlines": ["до 30 ноября 2026 года"],
        "contractor_requirements": ["наличие действующей лицензии"],
        "penalties": ["пеня за каждый день просрочки"],
    }


def test_accepts_explicit_absent_values() -> None:
    summary = TenderSummary(
        contract_amount=None,
        deadlines=[],
        contractor_requirements=[],
        penalties=[],
    )

    assert summary.model_dump() == {
        "contract_amount": None,
        "deadlines": [],
        "contractor_requirements": [],
        "penalties": [],
    }


@pytest.mark.parametrize(
    "missing_field",
    [
        "contract_amount",
        "deadlines",
        "contractor_requirements",
        "penalties",
    ],
)
def test_rejects_missing_summary_fields(missing_field: str) -> None:
    payload: dict[str, object] = {
        "contract_amount": None,
        "deadlines": [],
        "contractor_requirements": [],
        "penalties": [],
    }
    payload.pop(missing_field)

    with pytest.raises(ValidationError):
        TenderSummary.model_validate(payload)


def test_rejects_extra_fields() -> None:
    payload = {
        "contract_amount": None,
        "deadlines": [],
        "contractor_requirements": [],
        "penalties": [],
        "confidence": 0.95,
    }

    with pytest.raises(ValidationError):
        TenderSummary.model_validate(payload)


def test_strips_surrounding_whitespace() -> None:
    summary = TenderSummary(
        contract_amount="  14 850 000 рублей  ",
        deadlines=["  до 30 ноября 2026 года  "],
        contractor_requirements=["  наличие действующей лицензии  "],
        penalties=["  пеня за каждый день просрочки  "],
    )

    assert summary.model_dump() == {
        "contract_amount": "14 850 000 рублей",
        "deadlines": ["до 30 ноября 2026 года"],
        "contractor_requirements": ["наличие действующей лицензии"],
        "penalties": ["пеня за каждый день просрочки"],
    }


@pytest.mark.parametrize("contract_amount", ["", "   "])
def test_rejects_empty_contract_amount(contract_amount: str) -> None:
    with pytest.raises(ValidationError):
        TenderSummary(
            contract_amount=contract_amount,
            deadlines=[],
            contractor_requirements=[],
            penalties=[],
        )


@pytest.mark.parametrize(
    "invalid_field",
    ["deadlines", "contractor_requirements", "penalties"],
)
def test_rejects_empty_list_items(invalid_field: str) -> None:
    payload: dict[str, object] = {
        "contract_amount": None,
        "deadlines": [],
        "contractor_requirements": [],
        "penalties": [],
    }
    payload[invalid_field] = ["   "]

    with pytest.raises(ValidationError):
        TenderSummary.model_validate(payload)
