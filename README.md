# shengji-ai

AI agents for six-player 拖拉机. Each agent connects to `shengji-server` as a WebSocket client, receives game state, and responds with actions — identical to how a human browser client works.

## What This Is

A collection of increasingly capable AI agents:

| Agent | Approach | Strength | Status |
|---|---|---|---|
| `RandomAgent` | Random legal move | Baseline | Implemented & verified end-to-end |
| `RuleBasedAgent` | Hand-coded heuristics | Beginner | Implemented (basic heuristics) |
| `MCTSAgent` | Monte Carlo Tree Search | Intermediate | Placeholder |
| `DouZeroAgent` | Deep RL (DouZero architecture) | Strong | Placeholder |

`RandomAgent` and `RuleBasedAgent` have been validated playing complete six-player
games (DEALING → SCORING) against a live `shengji-server`. `MCTSAgent` and
`DouZeroAgent` currently return the first legal action and are not yet real.

## What This Is Not

- Not the game rules (see `shengji-engine`)
- Not the server (see `shengji-server`)
- Not the UI (see `shengji-app`)

## Dependencies

- `shengji-engine` (local, installed as editable — imported as the `shengji` package)
- `websockets` (async WebSocket client)
- `numpy`
- `torch` (only needed for `DouZeroAgent`)

## Installation

```bash
git clone https://github.com/JimYuhaoWu/shengji-ai
cd shengji-ai
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -e "../shengji-engine"  # the engine, imported as `shengji`
pip install -e ".[dev]"
```

## Running an Agent

The room must already exist on the server (create one with `POST /rooms`). Connect
an agent to a seat (0–5):

```bash
# Connect a RuleBasedAgent as player 3 in an existing room
python -m shengji_ai.run \
  --agent rule_based \
  --room abc123 \
  --player 3 \
  --server ws://localhost:8000
```

## Running a Full AI Game (all 6 seats)

`run_game` creates a fresh room on the server for each episode, then connects six
agents and plays to completion:

```bash
python -m shengji_ai.run_game \
  --agents random random rule_based rule_based random rule_based \
  --server ws://localhost:8000 \
  --episodes 100
```

`--agents` takes exactly six agent types: `random`, `rule_based`, `mcts`, `douzero`.

## Project Structure

```
shengji-ai/
├── shengji_ai/
│   ├── __init__.py
│   ├── base_agent.py        # BaseAgent: WebSocket client loop, act() + choose_kitty_bury()
│   ├── protocol.py          # (de)serialize GameState / Action / Card to-from the wire
│   ├── random_agent.py      # RandomAgent
│   ├── rule_based_agent.py  # RuleBasedAgent
│   ├── mcts/
│   │   ├── __init__.py
│   │   └── mcts_agent.py    # MCTSAgent (placeholder)
│   ├── douzero/
│   │   ├── __init__.py
│   │   ├── douzero_agent.py # DouZeroAgent (placeholder, inference only)
│   │   └── model.py         # Neural network architecture (placeholder)
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py       # Self-play training entry point (placeholder)
│   ├── run.py               # Single agent runner
│   └── run_game.py          # Full game runner (creates room, connects 6 agents)
├── tests/
│   ├── test_random_agent.py
│   └── test_rule_based_agent.py
├── e2e_test.py              # Manual harness: drive 6 real agents against a live server
├── pyproject.toml
├── README.md
└── CLAUDE.md
```

## Agent Interface

All agents subclass `BaseAgent` and implement `act()`:

```python
class BaseAgent:
    def act(self, state: GameState, legal_actions: list[Action]) -> Action:
        """Given game state and legal actions, return one of the legal actions."""
        raise NotImplementedError

    def choose_kitty_bury(self, state: GameState, hand: list[Card]) -> list[Card]:
        """Return 6 cards to bury during the KITTY phase.

        KITTY has C(32,6) ≈ 906k legal actions, so the server never sends them
        over the wire. The agent receives the dealer's full hand instead and
        picks 6 to bury. Default buries the first 6; override for smarter play.
        """
        return hand[:6]
```

Agents do not need to know about WebSockets — `base_agent.py` handles the connection
loop and calls `act()` (or `choose_kitty_bury()` in the KITTY phase) when it's the
agent's turn. The loop returns once a `game_over` message is received.

## Training (DouZeroAgent)

> Not yet implemented — `training/trainer.py` is a placeholder. The intended design
> (actor–learner self-play, see `CLAUDE.md`) would be invoked like:

```bash
python -m shengji_ai.training.trainer \
  --num_actors 4 \
  --device mps \
  --episodes 10_000_000 \
  --checkpoint_dir models/douzero_v1
```

## Running Tests

Unit tests run the agents against the real engine (no server needed):

```bash
pytest tests/ -v
```

End-to-end test against a live server (start `shengji-server` first):

```bash
# In shengji-server:  python -m uvicorn main:app --port 8000
python e2e_test.py --server ws://127.0.0.1:8000 --rest http://127.0.0.1:8000
```
