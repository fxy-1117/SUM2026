from dataclasses import dataclass
from typing import Optional


# Parameter grid used in the final project tables and figures.
TAU_M_VALUES = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
TAU_C_VALUES = [80, 90, 100]
FIX_NUMBER = 150
SHUFFLE_SEED = 1129


@dataclass(frozen=True)
class Setting:
    dataset: str
    variant: str
    source: Optional[str]
    steps: int

    @property
    def key(self) -> str:
        return f"{self.dataset}_{self.variant}"
