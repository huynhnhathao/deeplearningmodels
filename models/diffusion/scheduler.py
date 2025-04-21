from typing import List, Dict, Optional, Literal
from dataclasses import dataclass
import numpy as np


@dataclass
class Scheduler:
    beta_start: float
    beta_end: float
    steps: int
    scheduler_type: Literal["linear", "cosine"]
    beta_t: List[float] = []
    alpha_t: List[float] = []
    alpha_bar_t: List[float] = []

    def __init__(
        self,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        steps: int = 1000,
        scheduler_type: Literal["linear", "cosine"] = "linear",
    ) -> None:
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.steps = steps

        if scheduler_type == "linear":
            self.beta_t = np.linspace(beta_start, beta_end, steps)
            # it should broadcast the first array to match the self.beta_t array and do the subtraction per element
            self.alpha_t = np.array([1]) - self.beta_t
            self.alpha_bar_t = np.cumulative_prod(self.alpha_t, axis=-1)
        elif scheduler_type == "cosine":
            raise RuntimeError("not supported yet")
        else:
            raise RuntimeError(f"unsupported scheduler type {scheduler_type}")

    def get_beta(self, t: int) -> float:
        """
        t counts from 1 to steps
        """
        assert t > 0 and t <= self.steps, "t out of range"
        return self.beta_t[t]
