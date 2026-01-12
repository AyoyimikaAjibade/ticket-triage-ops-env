import numpy as np
from envfoundry_ticket_triage import TicketTriageEnv


def test_step_contract():
    env = TicketTriageEnv()
    obs, info = env.reset(seed=0)

    assert "global" in obs and "tickets" in obs and "engineers" in obs
    assert env.observation_space.contains(obs)

    a = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(a)

    assert env.observation_space.contains(obs2)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "reward_breakdown" in info2

    # basic sanity: arrays are finite
    assert np.isfinite(obs2["global"]).all()
    assert np.isfinite(obs2["tickets"]).all()
    assert np.isfinite(obs2["engineers"]).all()
