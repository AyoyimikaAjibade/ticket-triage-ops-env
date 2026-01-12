from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .env import TicketTriageEnv
from .utils import EnvConfig, write_jsonl


def rollout(seed: int, steps: int, out: Path) -> None:
    env = TicketTriageEnv(EnvConfig())
    obs, info = env.reset(seed=seed)

    rows: List[Dict[str, Any]] = []
    total_reward = 0.0

    for _ in range(steps):
        # baseline policy: simple heuristic
        # - if backlog exists, assign idle engineers to oldest ticket
        # - otherwise noop
        action = 0  # noop

        tickets = obs["tickets"]
        engineers = obs["engineers"]

        active_idxs = [i for i in range(tickets.shape[0]) if tickets[i, 0] > 0.5]
        idle_engs = [j for j in range(engineers.shape[0]) if engineers[j, 0] < 0.5]

        if active_idxs and idle_engs:
            # pick oldest ticket (max age)
            ages = [(i, tickets[i, 2]) for i in active_idxs]
            ticket_i = max(ages, key=lambda x: x[1])[0]
            engineer_j = idle_engs[0]
            # encode ASSIGN
            K = env.cfg.max_tickets
            N = env.cfg.n_engineers
            action = 1 + ticket_i * N + engineer_j

        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += float(reward)

        rows.append(
            {
                "t": step_info["t"],
                "action": int(action),
                "reward": float(reward),
                "reward_breakdown": step_info["reward_breakdown"],
                "backlog_count": int(step_info["backlog_count"]),
                "total_resolved": int(step_info["total_resolved"]),
            }
        )

        if terminated or truncated:
            break

    write_jsonl(out, rows)
    print(f"Saved trace to {out}")
    print(f"Total reward: {total_reward:.3f}")
    print(env.render())


def validate_determinism(seed: int, steps: int) -> None:
    env1 = TicketTriageEnv()
    env2 = TicketTriageEnv()

    o1, _ = env1.reset(seed=seed)
    o2, _ = env2.reset(seed=seed)

    rng = np.random.default_rng(seed + 123)

    for t in range(steps):
        a = int(rng.integers(0, env1.action_space.n))
        o1, r1, term1, trunc1, _ = env1.step(a)
        o2, r2, term2, trunc2, _ = env2.step(a)

        # exact equality expected because we use deterministic float operations in same order
        assert np.array_equal(o1["global"], o2["global"])
        assert np.array_equal(o1["tickets"], o2["tickets"])
        assert np.array_equal(o1["engineers"], o2["engineers"])
        assert float(r1) == float(r2)
        assert term1 == term2
        assert trunc1 == trunc2

        if term1 or trunc1:
            break

    print("Determinism check: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(prog="ticket-triage")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_roll = sub.add_parser("rollout", help="Run a rollout with a simple heuristic policy and save JSONL trace.")
    p_roll.add_argument("--seed", type=int, default=7)
    p_roll.add_argument("--steps", type=int, default=200)
    p_roll.add_argument("--out", type=Path, default=Path("trace.jsonl"))

    p_val = sub.add_parser("validate", help="Validate determinism (same seed + actions => same trajectory).")
    p_val.add_argument("--seed", type=int, default=7)
    p_val.add_argument("--steps", type=int, default=120)

    args = parser.parse_args()

    if args.cmd == "rollout":
        rollout(seed=args.seed, steps=args.steps, out=args.out)
    elif args.cmd == "validate":
        validate_determinism(seed=args.seed, steps=args.steps)
    else:
        raise SystemExit("Unknown command")
