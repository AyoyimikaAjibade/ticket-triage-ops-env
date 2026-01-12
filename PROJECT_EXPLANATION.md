# Ticket Triage Ops Environment - Deep Dive Explanation

## 🎯 What This Project Does (Simple Terms)

This project simulates a **support/engineering ticket management system** as a reinforcement learning (RL) environment. Think of it like a video game where:

- **You are the manager** trying to assign tickets to engineers
- **Tickets** arrive randomly with different priorities, deadlines (SLAs), and work requirements
- **Engineers** can work on tickets, but you need to decide who works on what
- **Your goal**: Resolve as many tickets as possible, avoid missing deadlines, and keep engineers busy

The environment is designed to be:
- **Deterministic**: Same seed + same actions = same results (reproducible)
- **Realistic**: Models real-world issues like context switching, regressions, blocked tickets
- **Testable**: Has unit tests to ensure it works correctly

---

## 📁 Project Structure & Configuration

### 1. **pyproject.toml** - The Project Configuration File

This is the "blueprint" of your Python package:

```toml
[build-system]
requires = ["setuptools>=75.0", "wheel", "packaging>=24.2"]
build-backend = "setuptools.build_meta"
```
- **What it does**: Tells Python how to build/install your package
- **Why it matters**: Ensures compatible build tools are used

```toml
[project]
name = "envfoundry-ticket-triage"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "gymnasium>=0.29.1",  # RL environment framework
  "numpy>=1.24",         # Numerical computing
]
```
- **Package metadata**: Name, version, Python version requirement
- **Dependencies**: Libraries your code needs to run

```toml
[project.optional-dependencies]
dev = [
  "pytest>=7.4",   # Testing framework
  "ruff>=0.4.0",   # Code linter/formatter
]
```
- **Dev dependencies**: Tools needed for development/testing, not runtime

```toml
[project.scripts]
ticket-triage = "envfoundry_ticket_triage.cli:main"
```
- **CLI command**: Creates a `ticket-triage` command you can run from terminal
- **Maps to**: The `main()` function in `cli.py`

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```
- **Pytest config**: Tells pytest where to find tests

```toml
[tool.ruff]
line-length = 100
```
- **Code style**: Max line length for formatting

---

## 🏗️ Code Architecture

### 2. **Package Structure**

```
envfoundry_ticket_triage/
├── __init__.py          # Exports TicketTriageEnv (main entry point)
├── env.py               # Main environment class (the "game engine")
├── spaces.py            # Defines observation/action spaces
├── dynamics.py          # Ticket and Engineer data models
├── utils.py             # Configuration and helper functions
└── cli.py               # Command-line interface
```

### 3. **Key Components Explained**

#### **utils.py** - Configuration & Helpers

```python
@dataclass(frozen=True)
class EnvConfig:
    n_engineers: int = 3          # How many engineers
    max_tickets: int = 10         # Max tickets in backlog
    horizon: int = 200            # Episode length (time steps)
    p_new_ticket: float = 0.35    # Probability of new ticket per step
    # ... many more parameters
```

- **EnvConfig**: All the "knobs" you can tune (engineers, ticket arrival rate, rewards, etc.)
- **make_rng()**: Creates a random number generator (for determinism)
- **write_jsonl()**: Helper to save data

#### **dynamics.py** - Data Models

```python
@dataclass
class Ticket:
    active: bool = False
    severity: int = 0        # 0-3 (how urgent)
    age: int = 0             # How old the ticket is
    sla_due_in: int = 0      # Deadline countdown
    remaining_work: float = 0.0  # How much work left
    # ... more fields
```

- **Ticket**: Represents a support ticket with all its properties
- **Engineer**: Represents an engineer (busy/idle, what ticket they're on)
- **spawn_ticket()**: Creates a new random ticket

#### **spaces.py** - Observation & Action Spaces

**Observation Space** (what the agent "sees"):
```python
{
    "global": [time, backlog_count, avg_age, sla_breaches, resolved_count],
    "tickets": matrix of shape (10, 9)  # 10 tickets × 9 features each
    "engineers": matrix of shape (3, 3)  # 3 engineers × 3 features each
}
```

**Action Space** (what the agent can "do"):
- Single discrete space with encoded actions:
  - `0`: NOOP (do nothing)
  - `1..30`: ASSIGN(ticket_i, engineer_j)  # 10 tickets × 3 engineers
  - `31..40`: ESCALATE(ticket_i)
  - `41..50`: REQUEST_INFO(ticket_i)
  - `51..60`: DEFER(ticket_i)
  - `61..70`: SWARM(ticket_i)

#### **env.py** - The Main Environment

This is the "game engine" that implements the Gymnasium API:

**Key Methods:**
1. **`__init__()`**: Sets up observation/action spaces
2. **`reset(seed)`**: Starts a new episode, returns initial observation
3. **`step(action)`**: 
   - Applies the action
   - Advances time (tickets age, engineers work, new tickets arrive)
   - Calculates reward
   - Returns new observation, reward, done flags, info
4. **`render()`**: Prints a human-readable summary

**Internal Logic:**
- **`_apply_action()`**: Decodes action number → actual operation
- **`_do_work()`**: Engineers make progress on tickets
- **`_maybe_add_ticket()`**: Randomly spawns new tickets
- **`_resolve_ticket()`**: Marks ticket as done, frees engineers

**Reward System:**
```python
reward_breakdown = {
    "resolved": +1.0 per ticket,           # Good!
    "sla_breach": -0.2 per breach,         # Bad!
    "idle_with_backlog": -0.05 per idle,   # Wasteful
    "context_switch": -0.05 per switch,     # Inefficient
    "escalate_cost": -0.01,                # Small cost
    # ... more costs
}
```

#### **cli.py** - Command-Line Interface

Two main commands:

1. **`rollout`**: Run the environment with a simple heuristic policy
   ```bash
   ticket-triage rollout --seed 7 --steps 200 --out trace.jsonl
   ```
   - Uses a simple policy: assign idle engineers to oldest tickets
   - Saves trajectory to JSONL file

2. **`validate`**: Test determinism
   ```bash
   ticket-triage validate --seed 7 --steps 120
   ```
   - Runs same seed + actions twice
   - Verifies identical results

---

## 🔄 How Everything Works Together

### **Installation Flow:**

1. **Setup**: `pip install -e ".[dev]"`
   - Reads `pyproject.toml`
   - Installs dependencies (gymnasium, numpy)
   - Installs dev tools (pytest, ruff)
   - Installs package in "editable" mode (changes take effect immediately)
   - Creates `ticket-triage` CLI command

### **Usage Flow:**

1. **Import & Create Environment:**
   ```python
   from envfoundry_ticket_triage import TicketTriageEnv
   env = TicketTriageEnv(EnvConfig())
   ```

2. **Reset (Start Episode):**
   ```python
   obs, info = env.reset(seed=42)
   # obs = {"global": [...], "tickets": [...], "engineers": [...]}
   ```

3. **Step Loop:**
   ```python
   for step in range(200):
       action = your_policy(obs)  # Choose action
       obs, reward, terminated, truncated, info = env.step(action)
       # Use reward to train your agent
       if terminated or truncated:
           break
   ```

4. **CLI Usage:**
   ```bash
   # Run a rollout
   ticket-triage rollout --seed 7 --steps 200 --out trace.jsonl
   
   # Validate determinism
   ticket-triage validate --seed 7 --steps 120
   ```

---

## 🧪 Testing Strategy

### **Test Files:**

1. **`test_determinism.py`**: 
   - Creates two environments with same seed
   - Runs same actions
   - Verifies identical observations/rewards
   - **Why**: Ensures reproducibility (critical for RL research)

2. **`test_spaces.py`**:
   - Checks that `reset()` returns valid observation
   - Verifies observation is in the defined space
   - **Why**: Catches bugs in observation construction

3. **`test_step_contract.py`**:
   - Tests that `step()` returns correct types
   - Verifies observation space consistency
   - Checks for NaN/Inf values
   - **Why**: Ensures Gymnasium API contract is followed

### **Running Tests:**

```bash
pytest -q                    # Run all tests quietly
pytest tests/test_determinism.py  # Run specific test
```

---

## 🎮 Example: What Happens in One Step

1. **Agent chooses action**: `ASSIGN(ticket_3, engineer_1)`
   - Encoded as action number `1 + 3*3 + 1 = 11`

2. **Environment processes:**
   - Engineer 1 is assigned to ticket 3
   - If engineer was on another ticket → context switch penalty
   - Ticket 3 must be active and not blocked

3. **Time advances:**
   - All tickets age by 1
   - SLA countdowns decrease
   - Blocked tickets unblock (if delay expired)
   - 35% chance: new ticket arrives
   - 5% chance: regression (work increases on random ticket)
   - Engineers make progress on assigned tickets

4. **Work progress:**
   - Engineer 1 works on ticket 3
   - Work rate = 1.0 per step (modified by severity)
   - `ticket_3.remaining_work -= 1.0 * (1 + 0.1 * severity)`

5. **Resolution:**
   - If `remaining_work <= 0`: ticket resolved
   - Engineer freed, reward given

6. **Reward calculation:**
   - +1.0 for resolved tickets
   - -0.2 for each SLA breach
   - -0.05 for idle engineers (if backlog exists)
   - -0.05 for context switches
   - Small costs for escalate/request_info/defer/swarm

7. **Return:**
   - New observation (updated state)
   - Reward (scalar)
   - `terminated` (true if backlog exploded)
   - `truncated` (true if reached horizon)
   - `info` (reward breakdown, stats)

---

## 🔑 Key Design Principles

1. **Determinism**: Same seed + actions = same results
   - Critical for reproducible research
   - Uses seeded random number generators

2. **Explicit Rewards**: `info["reward_breakdown"]` shows why reward changed
   - Helps debug agent behavior
   - Makes reward shaping transparent

3. **Realistic Dynamics**: 
   - Context switching costs
   - Regressions (work can increase)
   - Blocked tickets (need info)
   - SLA deadlines

4. **Clean API**: Follows Gymnasium standard
   - Easy to use with existing RL libraries
   - Standard `reset()` and `step()` interface

5. **Testability**: Unit tests ensure correctness
   - Determinism tests
   - Space contract tests
   - Step contract tests

---

## 📊 Configuration Tuning

You can customize the environment by modifying `EnvConfig`:

```python
from envfoundry_ticket_triage import TicketTriageEnv
from envfoundry_ticket_triage.utils import EnvConfig

# Custom config
config = EnvConfig(
    n_engineers=5,           # More engineers
    max_tickets=20,          # Larger backlog
    p_new_ticket=0.5,       # Faster arrival rate
    r_resolve=2.0,          # Higher reward for resolving
)

env = TicketTriageEnv(config)
```

---

## 🚀 Next Steps for Using This

1. **Train an RL agent**: Use with Stable-Baselines3, Ray RLlib, etc.
2. **Experiment with policies**: Try different assignment strategies
3. **Tune parameters**: Adjust `EnvConfig` for different scenarios
4. **Extend functionality**: Add new actions, ticket types, etc.

This environment is designed to be a **production-quality RL environment** that you can actually use for research and development!
