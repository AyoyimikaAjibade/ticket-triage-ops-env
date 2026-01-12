from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .utils import EnvConfig


@dataclass
class Ticket:
    active: bool = False
    severity: int = 0
    age: int = 0
    component: int = 0
    sla_due_in: int = 0
    deps: int = 0
    needs_info: bool = False
    blocked_steps: int = 0
    remaining_work: float = 0.0

    def as_features(self, horizon: int, info_delay: int) -> np.ndarray:
        return np.array(
            [
                1.0 if self.active else 0.0,
                float(self.severity),
                float(self.age),
                float(self.component),
                float(self.sla_due_in),  # can go negative after breach
                float(self.deps),
                1.0 if self.needs_info else 0.0,
                float(min(max(self.blocked_steps, 0), info_delay)),
                float(max(self.remaining_work, 0.0)),
            ],
            dtype=np.float32,
        )


@dataclass
class Engineer:
    busy: bool = False
    ticket_idx: Optional[int] = None
    time_on_ticket: int = 0

    def as_features(self, max_tickets: int, horizon: int) -> np.ndarray:
        # encode None as max_tickets (a valid “sentinel” within space bounds)
        idx = self.ticket_idx if self.ticket_idx is not None else max_tickets
        return np.array([1.0 if self.busy else 0.0, float(idx), float(self.time_on_ticket)], dtype=np.float32)


def spawn_ticket(cfg: EnvConfig, rng: np.random.Generator) -> Ticket:
    sev = int(rng.integers(0, 4))  # 0..3
    comp = int(rng.integers(0, cfg.n_components))
    deps = int(rng.integers(0, 6))
    needs_info = bool(rng.random() < cfg.p_needs_info)
    sla = int(rng.integers(cfg.min_sla, cfg.max_sla + 1))
    work = float(rng.uniform(cfg.min_work, cfg.max_work))
    return Ticket(
        active=True,
        severity=sev,
        age=0,
        component=comp,
        sla_due_in=sla,
        deps=deps,
        needs_info=needs_info,
        blocked_steps=0,
        remaining_work=work,
    )
