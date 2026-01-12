from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from .dynamics import Engineer, Ticket, spawn_ticket
from .spaces import make_observation_space
from .utils import EnvConfig, make_rng


class TicketTriageEnv(gym.Env):
    """
    A deterministic, testable Gymnasium environment modeling ticket triage operations.

    Actions supported:
      - NOOP
      - ASSIGN(ticket_i, engineer_j)
      - ESCALATE(ticket_i)
      - REQUEST_INFO(ticket_i)
      - DEFER(ticket_i)
      - SWARM(ticket_i)  (all engineers focus one ticket for one step)

    Notes:
      - This environment is about software engineering / simulation quality, not “training an RL agent”.
      - Determinism: same seed + same actions => same trajectory.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config: Optional[EnvConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode

        self.observation_space = make_observation_space(self.cfg)

        # action_space is one big Discrete with encoding:
        # 0: NOOP
        # 1..(K*N): ASSIGN
        # next K: ESCALATE
        # next K: REQUEST_INFO
        # next K: DEFER
        # next K: SWARM
        K = self.cfg.max_tickets
        N = self.cfg.n_engineers
        self._assign_count = K * N
        self._base_assign = 1
        self._base_escalate = self._base_assign + self._assign_count
        self._base_request = self._base_escalate + K
        self._base_defer = self._base_request + K
        self._base_swarm = self._base_defer + K
        self._action_n = self._base_swarm + K
        self.action_space = gym.spaces.Discrete(self._action_n)

        # state
        self._rng: Optional[np.random.Generator] = None
        self.t: int = 0
        self.total_resolved: int = 0
        self.total_created: int = 0
        self.backlog_total_seen: int = 0  # includes dropped due to capacity
        self.tickets = [Ticket() for _ in range(self.cfg.max_tickets)]
        self.engineers = [Engineer() for _ in range(self.cfg.n_engineers)]

    # ---------------------------
    # Gymnasium API
    # ---------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._rng = make_rng(seed)

        self.t = 0
        self.total_resolved = 0
        self.total_created = 0
        self.backlog_total_seen = 0

        self.tickets = [Ticket() for _ in range(self.cfg.max_tickets)]
        self.engineers = [Engineer() for _ in range(self.cfg.n_engineers)]

        # spawn a few initial tickets
        initial = int(self._rng.integers(2, min(5, self.cfg.max_tickets) + 1))
        for _ in range(initial):
            self._maybe_add_ticket(force=True)

        obs = self._get_obs()
        info = {"config": asdict(self.cfg)}
        return obs, info

    def step(self, action: int):
        assert self._rng is not None, "Call reset(seed=...) before step()."
        self.t += 1

        reward_breakdown = {
            "resolved": 0.0,
            "sla_breach": 0.0,
            "idle_with_backlog": 0.0,
            "context_switch": 0.0,
            "escalate_cost": 0.0,
            "request_info_cost": 0.0,
            "defer_cost": 0.0,
            "swarm_cost": 0.0,
        }

        # 1) apply the chosen action
        self._apply_action(int(action), reward_breakdown)

        # 2) advance time: age tickets, SLA countdown, unblock, arrivals, regressions, work progress
        sla_breaches = 0
        active_idxs = self._active_ticket_indices()

        for i in active_idxs:
            tk = self.tickets[i]
            tk.age += 1
            tk.sla_due_in -= 1
            if tk.sla_due_in < 0:
                sla_breaches += 1
            if tk.blocked_steps > 0:
                tk.blocked_steps -= 1

        # stochastic new ticket arrival
        if self._rng.random() < self.cfg.p_new_ticket:
            self._maybe_add_ticket(force=False)

        # regressions: small chance to add work to random active ticket
        if active_idxs and (self._rng.random() < self.cfg.p_regression):
            j = int(self._rng.choice(active_idxs))
            self.tickets[j].remaining_work += self.cfg.regression_work

        # work progress
        self._do_work()

        # resolve tickets whose remaining_work <= 0
        resolved_now = 0
        for i in self._active_ticket_indices():
            tk = self.tickets[i]
            if tk.remaining_work <= 0.0:
                resolved_now += 1
                self._resolve_ticket(i)

        if resolved_now > 0:
            reward_breakdown["resolved"] += self.cfg.r_resolve * resolved_now

        # SLA penalty
        if sla_breaches > 0:
            reward_breakdown["sla_breach"] += self.cfg.r_sla_breach * sla_breaches

        # idle penalty if backlog exists
        backlog_count = len(self._active_ticket_indices())
        if backlog_count > 0:
            idle = sum(1 for e in self.engineers if not e.busy)
            if idle > 0:
                reward_breakdown["idle_with_backlog"] += self.cfg.r_idle_with_backlog * idle

        reward = float(sum(reward_breakdown.values()))

        # termination logic
        terminated = False
        truncated = self.t >= self.cfg.horizon

        # “fail fast” if we’ve seen too many tickets overall (backlog explosion proxy)
        if self.backlog_total_seen >= self.cfg.backlog_fail_threshold:
            terminated = True

        obs = self._get_obs()
        info = {
            "t": self.t,
            "backlog_count": backlog_count,
            "total_resolved": self.total_resolved,
            "total_created": self.total_created,
            "reward_breakdown": reward_breakdown,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        # simple ANSI render (prints a small summary)
        active = self._active_ticket_indices()
        breaches = sum(1 for i in active if self.tickets[i].sla_due_in < 0)
        lines = []
        lines.append(f"t={self.t} backlog={len(active)} breaches={breaches} resolved={self.total_resolved}")
        lines.append("Engineers:")
        for idx, e in enumerate(self.engineers):
            if e.busy and e.ticket_idx is not None:
                lines.append(f"  E{idx}: ticket={e.ticket_idx} time_on={e.time_on_ticket}")
            else:
                lines.append(f"  E{idx}: idle")
        lines.append("Top tickets:")
        for i in active[: min(5, len(active))]:
            tk = self.tickets[i]
            lines.append(
                f"  T{i}: sev={tk.severity} age={tk.age} sla={tk.sla_due_in} "
                f"work={tk.remaining_work:.1f} blocked={tk.blocked_steps}"
            )
        return "\n".join(lines)

    # ---------------------------
    # Internals
    # ---------------------------
    def _get_obs(self) -> Dict[str, np.ndarray]:
        active = self._active_ticket_indices()
        backlog_count = len(active)
        avg_age = float(np.mean([self.tickets[i].age for i in active])) if active else 0.0
        sla_breaches = float(sum(1 for i in active if self.tickets[i].sla_due_in < 0))
        global_vec = np.array([self.t, backlog_count, avg_age, sla_breaches, self.total_resolved], dtype=np.float32)

        tickets_mat = np.zeros((self.cfg.max_tickets, 9), dtype=np.float32)
        for i in range(self.cfg.max_tickets):
            tickets_mat[i] = self.tickets[i].as_features(self.cfg.horizon, self.cfg.info_delay_steps)

        engineers_mat = np.zeros((self.cfg.n_engineers, 3), dtype=np.float32)
        for j in range(self.cfg.n_engineers):
            engineers_mat[j] = self.engineers[j].as_features(self.cfg.max_tickets, self.cfg.horizon)

        return {"global": global_vec, "tickets": tickets_mat, "engineers": engineers_mat}

    def _active_ticket_indices(self) -> list[int]:
        return [i for i, tk in enumerate(self.tickets) if tk.active]

    def _maybe_add_ticket(self, force: bool) -> None:
        assert self._rng is not None
        # find first inactive slot
        slot = None
        for i, tk in enumerate(self.tickets):
            if not tk.active:
                slot = i
                break

        self.backlog_total_seen += 1
        if slot is None:
            # queue overflow: drop ticket (still counts as “seen”)
            return

        self.tickets[slot] = spawn_ticket(self.cfg, self._rng)
        self.total_created += 1

    def _resolve_ticket(self, idx: int) -> None:
        # free engineers on this ticket
        for e in self.engineers:
            if e.ticket_idx == idx:
                e.busy = False
                e.ticket_idx = None
                e.time_on_ticket = 0
        self.tickets[idx] = Ticket()  # reset slot
        self.total_resolved += 1

    # -------- Actions --------
    def _apply_action(self, action: int, rb: Dict[str, float]) -> None:
        K = self.cfg.max_tickets
        N = self.cfg.n_engineers

        if action == 0:
            return

        # ASSIGN range
        if self._base_assign <= action < self._base_escalate:
            a = action - self._base_assign
            ticket_i = a // N
            engineer_j = a % N
            self._assign(engineer_j, ticket_i, rb)
            return

        # ESCALATE
        if self._base_escalate <= action < self._base_request:
            ticket_i = action - self._base_escalate
            self._escalate(ticket_i, rb)
            return

        # REQUEST_INFO
        if self._base_request <= action < self._base_defer:
            ticket_i = action - self._base_request
            self._request_info(ticket_i, rb)
            return

        # DEFER
        if self._base_defer <= action < self._base_swarm:
            ticket_i = action - self._base_defer
            self._defer(ticket_i, rb)
            return

        # SWARM
        if self._base_swarm <= action < self._action_n:
            ticket_i = action - self._base_swarm
            self._swarm(ticket_i, rb)
            return

        # out of range => noop
        return

    def _assign(self, engineer_j: int, ticket_i: int, rb: Dict[str, float]) -> None:
        if engineer_j < 0 or engineer_j >= self.cfg.n_engineers:
            return
        if ticket_i < 0 or ticket_i >= self.cfg.max_tickets:
            return
        tk = self.tickets[ticket_i]
        if not tk.active:
            return
        if tk.blocked_steps > 0:
            return

        e = self.engineers[engineer_j]
        # context switch penalty if switching off an active ticket
        if e.busy and e.ticket_idx is not None and e.ticket_idx != ticket_i:
            rb["context_switch"] += self.cfg.r_context_switch

        e.busy = True
        e.ticket_idx = ticket_i
        e.time_on_ticket = 0

    def _escalate(self, ticket_i: int, rb: Dict[str, float]) -> None:
        if 0 <= ticket_i < self.cfg.max_tickets and self.tickets[ticket_i].active:
            self.tickets[ticket_i].severity = min(3, self.tickets[ticket_i].severity + 1)
            # bring SLA closer slightly to reflect urgency pressure (optional realism)
            self.tickets[ticket_i].sla_due_in = max(-self.cfg.horizon, self.tickets[ticket_i].sla_due_in - 1)
            rb["escalate_cost"] += self.cfg.r_escalate_cost

    def _request_info(self, ticket_i: int, rb: Dict[str, float]) -> None:
        if 0 <= ticket_i < self.cfg.max_tickets and self.tickets[ticket_i].active:
            tk = self.tickets[ticket_i]
            tk.blocked_steps = self.cfg.info_delay_steps
            tk.needs_info = False
            rb["request_info_cost"] += self.cfg.r_request_info_cost

    def _defer(self, ticket_i: int, rb: Dict[str, float]) -> None:
        if 0 <= ticket_i < self.cfg.max_tickets and self.tickets[ticket_i].active:
            self.tickets[ticket_i].sla_due_in += self.cfg.defer_sla_push
            rb["defer_cost"] += self.cfg.r_defer_cost

    def _swarm(self, ticket_i: int, rb: Dict[str, float]) -> None:
        # For one step, all engineers focus this ticket (regardless of current assignment).
        # We implement it as a special “boost” in _do_work by setting a flag.
        if 0 <= ticket_i < self.cfg.max_tickets and self.tickets[ticket_i].active:
            self._swarm_target = ticket_i
            rb["swarm_cost"] += self.cfg.r_swarm_cost

    # -------- Work --------
    def _do_work(self) -> None:
        # swarm flag default
        swarm_target = getattr(self, "_swarm_target", None)
        if hasattr(self, "_swarm_target"):
            delattr(self, "_swarm_target")

        # increment time_on_ticket + apply work
        # If swarming, we ignore individual ticket assignments for one step.
        if swarm_target is not None and self.tickets[swarm_target].active and self.tickets[swarm_target].blocked_steps == 0:
            # total effective rate with diminishing returns
            n = self.cfg.n_engineers
            effective = self.cfg.base_work_rate * (1.0 + self.cfg.swarm_bonus * np.log1p(n))
            self.tickets[swarm_target].remaining_work -= float(effective)
            for e in self.engineers:
                e.busy = True
                e.ticket_idx = swarm_target
                e.time_on_ticket += 1
            return

        # normal assignments
        for e in self.engineers:
            if e.busy and e.ticket_idx is not None:
                idx = e.ticket_idx
                if 0 <= idx < self.cfg.max_tickets:
                    tk = self.tickets[idx]
                    if tk.active and tk.blocked_steps == 0:
                        # severity makes work slightly more “valuable” (or faster)
                        sev_bonus = 1.0 + 0.1 * tk.severity
                        tk.remaining_work -= float(self.cfg.base_work_rate * sev_bonus)
                        e.time_on_ticket += 1
                    else:
                        # ticket became inactive or blocked; engineer is idle
                        e.busy = False
                        e.ticket_idx = None
                        e.time_on_ticket = 0
            else:
                # idle
                e.busy = False
                e.ticket_idx = None
                e.time_on_ticket = 0
