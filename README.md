# 🎫 Ticket Triage Operations Environment

> **A reinforcement learning environment for optimizing engineering ticket management similar to Sentry (excluding RL)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

## 🚀 Executive Summary

**Ticket Triage Ops Environment** is a sophisticated simulation platform that models real-world engineering support operations. Built with production-quality standards, it enables organizations to:

- **Optimize resource allocation** through AI/ML-driven ticket assignment strategies
- **Reduce SLA breaches** by testing policies before deployment
- **Improve team efficiency** by modeling context switching, regressions, and realistic work dynamics
- **Enable reproducible research** with deterministic, fully-tested simulation

This is a **production-ready environment** designed for serious RL research and operational optimization.

---

## 💡 Why This Matters

### The Problem
Most reinforcement learning environments are either:
- **Too simplistic** (an uneffective problems that don't reflect reality)
- **Non-reproducible** (results can't be verified or replicated)
- **Poorly tested** (bugs go undetected, breaking research)

### My Solution
A **deterministic, testable, production-grade** environment that:
- ✅ Models realistic engineering operations (SLAs, dependencies, regressions, context switching)
- ✅ Guarantees reproducibility (same seed + actions = identical results)
- ✅ Includes comprehensive test coverage (determinism, API contracts, space validation)
- ✅ Provides professional tooling (CLI, logging, reward breakdowns)

---

## 🎯 Key Features

### 🏗️ **Production Quality**
- **Deterministic simulation** with seeded random number generation
- **Comprehensive unit tests** for correctness and reproducibility
- **Clean Gymnasium API** compatible with all major RL frameworks
- **Explicit reward breakdowns** for transparency and debugging

### 🎮 **Realistic Dynamics**
- **Stochastic ticket arrivals** with configurable rates
- **SLA deadlines** with breach penalties
- **Work regressions** (tickets can become harder)
- **Context switching costs** (reassigning engineers is expensive)
- **Blocked tickets** (information requests delay progress)
- **Team swarming** (all engineers focus on critical tickets)

### 🛠️ **Developer Experience**
- **Command-line interface** for quick rollouts and validation
- **Comprehensive documentation** with examples
- **Type hints** throughout for better IDE support
- **Modular design** for easy extension and customization

---

## 📊 What It Simulates

### State (What the Agent Sees)
- **Global metrics**: Time step, backlog size, average age, SLA breaches, resolved count
- **Ticket matrix**: 10 tickets × 9 features (severity, age, component, SLA, dependencies, blocked status, work remaining)
- **Engineer matrix**: 3 engineers × 3 features (busy/idle, current assignment, time on ticket)

### Actions (What the Agent Can Do)
- **ASSIGN**: Assign a ticket to an engineer
- **ESCALATE**: Increase ticket priority
- **REQUEST_INFO**: Block ticket for information gathering
- **DEFER**: Push back SLA deadline
- **SWARM**: All engineers focus on one critical ticket
- **NOOP**: Do nothing

### Rewards (What the Agent Optimizes)
- ✅ **+1.0** per resolved ticket
- ❌ **-0.2** per SLA breach
- ❌ **-0.05** per idle engineer (when backlog exists)
- ❌ **-0.05** per context switch (reassignment)
- Small costs for escalate/request_info/defer/swarm actions

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd ticket-triage-ops-env

# Create virtual environment (recommended)
python3.10 -m venv .venv_ticket_triage
source .venv_ticket_triage/bin/activate  # On Windows: .venv_ticket_triage\Scripts\activate

# Install package with development dependencies
pip install -e ".[dev]"
```

### Basic Usage

```python
from envfoundry_ticket_triage import TicketTriageEnv

# Create environment
env = TicketTriageEnv()

# Reset and start episode
obs, info = env.reset(seed=42)

# Run simulation
for step in range(200):
    # Your policy here (e.g., assign oldest ticket to idle engineer)
    action = your_policy(obs)
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Step {step}: Reward = {reward:.2f}, Backlog = {info['backlog_count']}")
    
    if terminated or truncated:
        break

print(f"Total resolved: {info['total_resolved']}")
```

### Command-Line Interface

```bash
# Run a rollout with default heuristic policy
ticket-triage rollout --seed 7 --steps 200 --out trace.jsonl

# Validate determinism (critical for research)
ticket-triage validate --seed 7 --steps 120
```

---

## 📈 Example Output

```
t=200 backlog=3 breaches=0 resolved=12
Engineers:
  E0: ticket=5 time_on=8
  E1: ticket=7 time_on=2
  E2: idle
Top tickets:
  T5: sev=2 age=15 sla=5 work=2.3 blocked=0
  T7: sev=1 age=8 sla=12 work=4.1 blocked=0
  T9: sev=3 age=3 sla=2 work=1.8 blocked=0
```

---

## 🧪 Testing & Quality Assurance

This project maintains **production-grade quality standards**:

```bash
# Run all tests
pytest

# Run specific test suites
pytest tests/test_determinism.py    # Verify reproducibility
pytest tests/test_spaces.py         # Validate observation spaces
pytest tests/test_step_contract.py  # Check API compliance
```

**Test Coverage:**
- ✅ Determinism verification (same seed = same results)
- ✅ Observation space validation
- ✅ Gymnasium API contract compliance
- ✅ Numerical stability checks

---

## 🎛️ Configuration

Customize the environment to match your use case:

```python
from envfoundry_ticket_triage import TicketTriageEnv
from envfoundry_ticket_triage.utils import EnvConfig

# Custom configuration
config = EnvConfig(
    n_engineers=5,              # Team size
    max_tickets=20,             # Backlog capacity
    horizon=500,               # Episode length
    p_new_ticket=0.5,          # Arrival rate
    r_resolve=2.0,             # Reward for resolution
    r_sla_breach=-0.5,         # Penalty for SLA breach
)

env = TicketTriageEnv(config)
```

See `envfoundry_ticket_triage/utils.py` for all configuration options.

---

## 🔗 Integration with RL Frameworks

Works seamlessly with popular RL libraries:

```python
# Stable-Baselines3
from stable_baselines3 import PPO
from envfoundry_ticket_triage import TicketTriageEnv

env = TicketTriageEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Ray RLlib
from ray.rllib.algorithms.ppo import PPOConfig
from envfoundry_ticket_triage import TicketTriageEnv

config = PPOConfig().environment(TicketTriageEnv)
algo = config.build()
```

---

## 📚 Documentation

- **[API Reference](envfoundry_ticket_triage/)** - Source code with docstrings
- **[Examples](tests/)** - Test files demonstrate usage patterns

---

## 🏆 Production Readiness Checklist

- ✅ **Deterministic** - Reproducible results with seeded RNG
- ✅ **Tested** - Comprehensive unit test coverage
- ✅ **Documented** - Clear API and usage examples
- ✅ **Type-hinted** - Better IDE support and error detection
- ✅ **Modular** - Easy to extend and customize
- ✅ **Standard API** - Compatible with Gymnasium ecosystem
- ✅ **CLI Tools** - Professional command-line interface
- ✅ **Reward Transparency** - Explicit breakdown for debugging

---

## 🤝 Contributing

Contributions are welcome! This project follows best practices:
- Type hints throughout
- Comprehensive test coverage
- Clear documentation
- Code formatting with Ruff

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

Built with:
- [Gymnasium](https://gymnasium.farama.org/) - Standard RL environment API
- [NumPy](https://numpy.org/) - Numerical computing
- [Pytest](https://pytest.org/) - Testing framework

---

**Ready to optimize your ticket triage operations?** Start with `pip install -e ".[dev]"` and run your first rollout!
