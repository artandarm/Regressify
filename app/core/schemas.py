from pydantic import BaseModel
from typing import Literal, Optional


class PipelineStep(BaseModel):
    name: str
    message: str
    verdict: Literal["ok", "warn", "error", "info"]
    p_value: Optional[float] = None
