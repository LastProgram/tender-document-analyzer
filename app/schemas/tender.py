from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class TenderSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_amount: NonEmptyText | None
    deadlines: list[NonEmptyText]
    contractor_requirements: list[NonEmptyText]
    penalties: list[NonEmptyText]
