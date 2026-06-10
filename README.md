# shengji-ai

AI agents for six-player 拖拉机. Each agent connects to `shengji-server` as a WebSocket client, receives game state, and responds with actions — identical to how a human browser client works.

## What This Is

A collection of increasingly capable AI agents:

| Agent | Approach | Strength | Status |
|---|---|---|---|
| `RandomAgent` | Random legal move | Baseline | Implemented |
| `RuleBasedAgent` | Hand-coded heuristics | Beginner | Implemented |
| `MCTSAgent` | Monte Carlo Tree Search | Intermediate | Planned |
| `DouZeroAgent` | Deep RL (DouZero architecture) | Strong | Planned |

## What This Is Not

- Not the game rules (see `shengji-engine`)
- Not the server (see `shengji-server`)
- Not the UI (see `shengji-app`)

## Dependencies

- `shengji-engine` (local, installed as editable)
- `websockets` (async WebSocket client)
- `torch` (for DouZeroAgent)
- `numpy`

## Installation

```bash
git clone https://github.com/jimyuhaowu/shengji-ai
cd shengji-ai
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pip install -e "../shengji-engine"
```

## Running an Agent

```bash
# Connect a RuleBasedAgent as player 3 in room "abc123"
python -m shengji_ai.run \
  --agent rule_based \
  --room abc123 \
  --player 3 \
  --server ws://localhost:8000
```

## Running a Full AI Game (all 6 seats)

```bash
python -m shengji_ai.run_game \
  --agents random random rule_based rule_based mcts mcts \
  --room test_room \
  --episodes 100
```

## Project Structure

```
shengji-ai/
├── shengji_ai/
│   ├── __init__.py
│   ├── base_agent.py        # BaseAgent abstract class + WebSocket client loop
│   ├── random_agent.py      # RandomAgent
│   ├── rule_based_agent.py  # RuleBasedAgent
│   ├── mcts/
│   │   ├── __init__.py
│   │   ├── mcts_agent.py    # MCTSAgent
│   │   └── mcts_core.py     # MCTS tree search, determinization
│   ├── douzero/
│   │   ├── __init__.py
│   │   ├── douzero_agent.py # DouZeroAgent (inference only)
│   │   └── model.py         # Neural network architecture
│   ├── training/
│   │   ├── trainer.py       # Self-play training loop
│   │   ├── actor.py         # Actor process (game simulation)
│   │   └── learner.py       # Learner process (gradient updates)
│   ├── run.py               # Single agent runner
│   └── run_game.py          # Full game runner (multiple agents)
├── models/                  # Saved model checkpoints
├── tests/
│   ├── test_random_agent.py
│   └── test_rule_based_agent.py
├── pyproject.toml
├── README.md
└── CLAUDE.md
```

## Agent Interface

All agents implement:

```python
class BaseAgent:
    def act(self, state: GameState, legal_actions: list[Action]) -> Action:
        """Given game state and legal actions, return chosen action."""
        raise NotImplementedError
```

Agents do not need to know about WebSockets — `base_agent.py` handles the connection loop and calls `act()` when it's the agent's turn.

## Training (DouZeroAgent)

```bash
# Start self-play training on M1 (uses MPS backend)
python -m shengji_ai.training.trainer \
  --num_actors 4 \
  --device mps \
  --episodes 10_000_000 \
  --checkpoint_dir models/douzero_v1
```

Training produces model checkpoints that `DouZeroAgent` loads for inference.

## Running Tests

```bash
pytest tests/ -v
```
