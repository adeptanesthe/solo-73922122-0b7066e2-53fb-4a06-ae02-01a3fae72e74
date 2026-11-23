"""Request/response schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from .stats import DIM

#: Bound on integer speed/pressure endpoints and gears.  Generous for real
#: transducer counts (even 1e12 base values) but keeps every exact
#: Fraction-to-float conversion and response number in range.
MAX_INT_MAGNITUDE = 10**18


class FactoryPayload(BaseModel):
    """One factory's integer summary.

    sums  = [sum speed, sum pressure, sum power]   (integers)
    outer = 3x3 integer matrix of sum(x_i * x_j) for x=(speed, pressure, power)
    correction = integer zero-point correction added to the pressure.
    """

    factory_id: str = Field(min_length=1, max_length=64)
    n: int = Field(ge=1)
    sums: list[int] = Field(min_length=DIM, max_length=DIM)
    outer: list[list[int]] = Field(min_length=DIM, max_length=DIM)
    correction: int = 0

    @field_validator("outer")
    @classmethod
    def _square(cls, v):
        if any(len(row) != DIM for row in v):
            raise ValueError("outer must be a 3x3 matrix")
        return v


class CreateVersionRequest(BaseModel):
    factories: list[FactoryPayload] = Field(min_length=3, max_length=8)

    @field_validator("factories")
    @classmethod
    def _unique_ids(cls, v):
        ids = [f.factory_id for f in v]
        if len(set(ids)) != len(ids):
            raise ValueError("factory_id values must be unique")
        return v


class CorrectionUpdate(BaseModel):
    factory_id: str = Field(min_length=1, max_length=64)
    correction: int


class PredictRequest(BaseModel):
    factory_id: str = Field(min_length=1, max_length=64)
    speed: float
    pressure: float


class GearRequest(BaseModel):
    """Conditions under which a usable integer gear is requested.

    ``pressure_min``/``pressure_max`` are *raw* integer pressure readings;
    the factory's current correction is applied by the service.  The
    pressure interval may be given in either order.
    """

    factory_id: str = Field(min_length=1, max_length=64)
    pressure_min: int = Field(ge=-MAX_INT_MAGNITUDE, le=MAX_INT_MAGNITUDE)
    pressure_max: int = Field(ge=-MAX_INT_MAGNITUDE, le=MAX_INT_MAGNITUDE)
    speed_min: int = Field(ge=-MAX_INT_MAGNITUDE, le=MAX_INT_MAGNITUDE)
    speed_max: int = Field(ge=-MAX_INT_MAGNITUDE, le=MAX_INT_MAGNITUDE)
    power_limit: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _ranges(self):
        if self.speed_min > self.speed_max:
            raise ValueError("speed_min must be <= speed_max")
        return self


class GearReviewRequest(GearRequest):
    """Original conditions plus the gear chosen by the dispatcher."""

    gear: int = Field(ge=-MAX_INT_MAGNITUDE, le=MAX_INT_MAGNITUDE)


class GearSelectionResponse(BaseModel):
    factory_id: str
    available: bool
    gear: int | None
    version: int
    correction: int
    power_limit: float
    mahalanobis_limit: int
    max_power: float | None
    max_distance: float | None


class GearSuggestion(BaseModel):
    available: bool
    gear: int | None
    max_power: float | None
    max_distance: float | None


class GearReviewResponse(BaseModel):
    factory_id: str
    accepted: bool
    gear: int
    version: int
    correction: int
    power_limit: float
    mahalanobis_limit: int
    max_power: float | None
    max_distance: float | None
    reasons: list[str]
    suggestion: GearSuggestion | None = None
