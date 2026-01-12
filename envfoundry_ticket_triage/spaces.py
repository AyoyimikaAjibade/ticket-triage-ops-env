from __future__ import annotations

import gymnasium as gym
import numpy as np

from .utils import EnvConfig


def make_observation_space(cfg: EnvConfig) -> gym.spaces.Dict:
    """
    Observation is a dict to keep it readable and “engineering-like”:
      - global: vector of global backlog signals
      - tickets: matrix (max_tickets x ticket_features)
      - engineers: matrix (n_engineers x engineer_features)
    """
    # Global features: time, backlog_count, avg_age, sla_breaches, resolved_count
    global_low = np.array([0, 0, 0, 0, 0], dtype=np.float32)
    global_high = np.array([cfg.horizon, cfg.max_tickets, cfg.horizon, 999, 999], dtype=np.float32)

    # Ticket features:
    # 0 active (0/1)
    # 1 severity (0..3)
    # 2 age (0..horizon)
    # 3 component (0..n_components-1)
    # 4 sla_remaining (-horizon..horizon)
    # 5 deps (0..5)
    # 6 needs_info (0/1)
    # 7 blocked_steps (0..info_delay_steps)
    # 8 remaining_work (0..max_work+regression)
    ticket_low = np.array([0, 0, 0, 0, -cfg.horizon, 0, 0, 0, 0], dtype=np.float32)
    ticket_high = np.array(
        [1, 3, cfg.horizon, cfg.n_components - 1, cfg.horizon, 5, 1, cfg.info_delay_steps, cfg.max_work + 10],
        dtype=np.float32,
    )

    # Engineer features:
    # 0 busy (0/1)
    # 1 ticket_index (-1..max_tickets-1) -> we clamp to [0..max_tickets] by encoding -1 as max_tickets
    # 2 time_on_ticket (0..horizon)
    eng_low = np.array([0, 0, 0], dtype=np.float32)
    eng_high = np.array([1, cfg.max_tickets, cfg.horizon], dtype=np.float32)

    # Create full arrays with the correct shape for tickets and engineers
    # Gymnasium requires low/high to match the shape exactly (no broadcasting)
    ticket_low_full = np.tile(ticket_low, (cfg.max_tickets, 1))
    ticket_high_full = np.tile(ticket_high, (cfg.max_tickets, 1))
    eng_low_full = np.tile(eng_low, (cfg.n_engineers, 1))
    eng_high_full = np.tile(eng_high, (cfg.n_engineers, 1))

    return gym.spaces.Dict(
        {
            "global": gym.spaces.Box(global_low, global_high, dtype=np.float32),
            "tickets": gym.spaces.Box(
                low=ticket_low_full,
                high=ticket_high_full,
                dtype=np.float32,
            ),
            "engineers": gym.spaces.Box(
                low=eng_low_full,
                high=eng_high_full,
                dtype=np.float32,
            ),
        }
    )
