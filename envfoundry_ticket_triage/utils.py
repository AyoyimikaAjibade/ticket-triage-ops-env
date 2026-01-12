from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class EnvConfig:
    # Core sizes
    n_engineers: int = 3
    max_tickets: int = 10
    n_components: int = 6

    # Episode length
    horizon: int = 200

    # Ticket arrival + work dynamics
    p_new_ticket: float = 0.35
    min_work: float = 2.0
    max_work: float = 8.0
    base_work_rate: float = 1.0  # per engineer per step
    swarm_bonus: float = 0.6     # extra efficiency when swarming (diminishing returns)

    # Regression
    p_regression: float = 0.05
    regression_work: float = 1.5

    # SLA / aging
    min_sla: int = 12
    max_sla: int = 40

    # Info requests + deferrals
    p_needs_info: float = 0.2
    info_delay_steps: int = 3     # how long ticket is blocked after request_info
    defer_sla_push: int = 6

    # Rewards
    r_resolve: float = 1.0
    r_sla_breach: float = -0.2
    r_idle_with_backlog: float = -0.05
    r_context_switch: float = -0.05
    r_escalate_cost: float = -0.01
    r_request_info_cost: float = -0.02
    r_defer_cost: float = -0.02
    r_swarm_cost: float = -0.03

    # Termination safety
    backlog_fail_threshold: int = 30  # if arrivals > processing, treat as failure


def make_rng(seed: Optional[int]) -> np.random.Generator:
    if seed is None:
        # gymnasium supports random seeding too, but we keep it explicit
        seed = np.random.SeedSequence().entropy  # type: ignore[attr-defined]
    return np.random.default_rng(int(seed))


def write_jsonl(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
