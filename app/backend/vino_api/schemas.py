"""Public request and response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class FeedbackRequest(BaseModel):
    verdict: Literal["correct", "incorrect", "not_in_catalog"]
    correct_slug: str | None = Field(default=None, max_length=240)
    note: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_correct_slug(self) -> "FeedbackRequest":
        if self.verdict == "incorrect" and not self.correct_slug:
            raise ValueError("Для неверного результата нужен correct_slug")
        if self.verdict != "incorrect" and self.correct_slug:
            raise ValueError("correct_slug допустим только для verdict=incorrect")
        return self


class FlatRecognition(BaseModel):
    slug: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_ready: bool
    catalog_size: int
    queue_limit: int
    device: str
