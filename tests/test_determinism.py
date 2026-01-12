import numpy as np
from envfoundry_ticket_triage import TicketTriageEnv


def test_determinism_same_seed_same_actions():
    seed = 123
    env1 = TicketTriageEnv()
    env2 = TicketTriageEnv()

    o1, _ = env1.reset(seed=seed)
    o2, _ = env2.reset(seed=seed)

    rng = np.random.default_rng(999)

    for _ in range(80):
        a = int(rng.integers(0, env1.action_space.n))
        o1, r1, term1, trunc1, _ = env1.step(a)
        o2, r2, term2, trunc2, _ = env2.step(a)

        assert np.array_equal(o1["global"], o2["global"])
        assert np.array_equal(o1["tickets"], o2["tickets"])
        assert np.array_equal(o1["engineers"], o2["engineers"])
        assert float(r1) == float(r2)
        assert term1 == term2
        assert trunc1 == trunc2

        if term1 or trunc1:
            break
