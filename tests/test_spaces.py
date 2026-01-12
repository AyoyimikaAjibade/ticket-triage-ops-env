from envfoundry_ticket_triage import TicketTriageEnv


def test_observation_space_contains_reset_obs():
    env = TicketTriageEnv()
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs)
