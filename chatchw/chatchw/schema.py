from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


Sex = Literal["m", "f"]
ObservationId = Literal["temp", "resp_rate", "muac_mm"]
Op = Literal["eq", "lt", "le", "gt", "ge"]
TriageLevel = Literal["hospital", "clinic", "home"]


class Observation(BaseModel):
    id: ObservationId
    value: float

    model_config = {"extra": "forbid"}


class Symptoms(BaseModel):
    feels_very_hot: Optional[bool] = None
    blood_in_stool: Optional[bool] = None
    diarrhea_days: Optional[int] = None
    convulsion: Optional[bool] = None
    edema_both_feet: Optional[bool] = None

    model_config = {"extra": "forbid"}


class ContextFlags(BaseModel):
    malaria_present: bool
    cholera_present: bool
    stockout: Optional[dict[str, bool]] = None

    model_config = {"extra": "forbid"}


class EncounterInput(BaseModel):
    age_months: int = Field(ge=0)
    sex: Sex
    observations: List[Observation]
    symptoms: Symptoms
    context: ContextFlags

    model_config = {"extra": "forbid"}


class ObservationCondition(BaseModel):
    obs: ObservationId
    op: Op
    value: float

    model_config = {"extra": "forbid"}


class SymCondition(BaseModel):
    sym: str  # Dynamic - accepts any symptom variable
    eq: bool | int | float

    model_config = {"extra": "forbid"}


Condition = Union["AnyOfCondition", "AllOfCondition", ObservationCondition, SymCondition]


class AnyOfCondition(BaseModel):
    any_of: List[Condition]

    @model_validator(mode="after")
    def validate_nonempty(self) -> "AnyOfCondition":
        if not self.any_of:
            raise ValueError("any_of must be non-empty")
        return self

    model_config = {"extra": "forbid"}


class AllOfCondition(BaseModel):
    all_of: List[Condition]

    @model_validator(mode="after")
    def validate_nonempty(self) -> "AllOfCondition":
        if not self.all_of:
            raise ValueError("all_of must be non-empty")
        return self

    model_config = {"extra": "forbid"}


AnyOfCondition.model_rebuild()
AllOfCondition.model_rebuild()


class Action(BaseModel):
    id: str
    if_available: bool = False

    model_config = {"extra": "forbid"}


class ThenClause(BaseModel):
    set_flags: Optional[List[str]] = None
    propose_triage: Optional[TriageLevel] = None
    reasons: Optional[List[str]] = None
    actions: Optional[List[Action]] = None
    guideline_ref: str
    priority: Optional[int] = None

    model_config = {"extra": "forbid"}


class Rule(BaseModel):
    rule_id: str
    when: List[Condition]
    then: ThenClause
    priority: int = 0

    @model_validator(mode="after")
    def set_priority_from_then(self) -> "Rule":
        if self.then and self.then.priority is not None:
            self.priority = int(self.then.priority)
        return self

    model_config = {"extra": "forbid"}


class TraceEntry(BaseModel):
    rule_id: str
    guideline_ref: str
    timestamp: str

    model_config = {"extra": "forbid"}


class Decision(BaseModel):
    triage: TriageLevel
    reasons: List[str] = Field(default_factory=list)
    trace: List[TraceEntry] = Field(default_factory=list)

    model_config = {"extra": "forbid"}
