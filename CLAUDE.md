# CLAUDE.md — shengji-ai

## What You Are Building

A collection of AI agents for 拖拉机. Each agent is a Python WebSocket client that connects to `shengji-server`, receives game state, and replies with an action. The server treats AI agents identically to human browser clients.

## Cardinal Rules

1. **All agents share the same `BaseAgent` interface.** `act(state, legal_actions) -> Action`. Never break this contract.
2. **Agents must only choose from `legal_actions`.** Never construct an action from scratch — pick from the list the engine provides.
3. **`act()` must be synchronous and fast.** The WebSocket loop in `base_agent.py` is async; `act()` is called with `await loop.run_in_executor()`. Keep it under 5 seconds even for MCTS.
4. **No game logic in agents.** If you need to reason about what's legal, use `shengji-engine` functions — import them, don't reimplement them.
5. **Training code lives in `training/` only.** Inference agents in `*_agent.py` files must work without training infrastructure.

## Coding Standards

### 1. Simplicity First
- **Minimum code that solves the problem.** No speculative abstractions or features beyond what's asked.
- **No error handling for impossible scenarios.** Trust internal code and framework guarantees; validate only at system boundaries (user input, external APIs).
- **Three similar lines = time to extract.** One-off code stays inline.
- **Ask: "Is this overcomplicated?"** If yes, rewrite it.

### 2. Surgical Changes
- **Touch only what you must.** Don't improve adjacent code unless requested.
- **Match existing style.** Even if you'd do it differently.
- **Remove only YOUR orphans.** If your changes make an import/variable/function unused, delete it. Don't clean up pre-existing dead code.
- **Every changed line traces to the user's request.** No drive-by refactoring.

### 3. Think Before Coding
- **State assumptions explicitly.** Uncertain about interpretation? Ask before implementing.
- **Surface tradeoffs.** Don't pick silently between equally valid approaches.
- **Don't hide confusion.** If something is unclear, stop and name what's confusing.
- **Simplify when possible.** If 50 lines can do what 200 does, rewrite it.

### 4. Goal-Driven Execution
- **Define success criteria first.** Transform tasks into verifiable checks:
  - "Add WebSocket endpoint" → test connection, message receipt, broadcast
  - "Fix disconnect bug" → write test for reconnect, make it pass
  - "Implement room expiry" → test stale room cleanup, pass it
- **State brief plans for multi-step work.** Format: `1. [Step] → verify: [check]`
- **Loop until verified.** Success = tests pass + behavior matches spec.

## Base Agent WebSocket Loop (as implemented, verified against the real server)

> The server sends NO `your_turn` message. It broadcasts a `state_update` to every
> player on every change; it's your turn when `msg["current_player"] == player_id`.
> Everything below was verified by playing full games against a live `shengji-server`.

```python
# base_agent.py
class BaseAgent:
    def act(self, state: GameState, legal_actions: list[Action]) -> Action:
        raise NotImplementedError

    def choose_kitty_bury(self, state, hand):   # KITTY only; default buries first 6
        return hand[:6]

    async def run(self, server_url, room_id, player_id, timeout=5.0):
        uri = f"{server_url}/ws/{room_id}/{player_id}"
        async with websockets.connect(uri) as ws:
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "state_update":
                    if msg["current_player"] != player_id:
                        continue
                    legal = msg.get("legal_actions")
                    if legal:                           # normal turn (non-empty list)
                        state = deserialize_state(msg)
                        actions = [deserialize_action(a) for a in legal]
                        action = await loop.run_in_executor(None, self.act, state, actions)
                        await ws.send(json.dumps(action_to_dict(action)))
                    elif msg.get("legal_actions_truncated"):   # KITTY (~906k options)
                        state = deserialize_state(msg)
                        bury = self.choose_kitty_bury(state, list(state.hands[player_id]))
                        await ws.send(json.dumps({"type": "take_kitty", "cards": [...]}))
                    # empty legal_actions (e.g. SCORING) → nothing to do
                elif msg.get("type") == "game_over":
                    return                              # stop; let callers proceed
```

### Wire protocol gotchas (each one bit us; all now handled)

1. **No `your_turn` message.** Drive off `state_update` + `current_player`.
2. **KITTY's `legal_actions` is truncated** (`legal_actions_truncated: true`, the list
   omitted). Don't call `act()` — call `choose_kitty_bury()` and send a semantic
   `take_kitty` message with 6 card dicts.
3. **Empty `legal_actions` (`[]`) in SCORING** is not actionable. Guard on truthiness,
   not `is not None`, or `random.choice([])` will crash.
4. **`run()` must return on `game_over`,** otherwise `asyncio.gather()` over six agents
   hangs forever and multi-episode runners stall after game one.
5. **Rooms must exist before connecting.** The server closes unknown rooms with code
   4004, so `run_game` creates each room via `POST /rooms` first.

### Message/card encodings (from shengji-server)

- Card dict: `{"suit": "H", "rank": "7", "deck_id": 0}` (suit/rank are the enum `.value`s).
- `Card` is imported from `shengji.card`, NOT the top-level `shengji` package.
- Actions sent to the server use semantic messages: `play_cards`, `bid_trump`,
  `pass_trump`, `take_kitty`, `call_helper` (see `protocol.action_to_dict`).
- The server's privacy filter means only our own hand is populated; opponents'
  hands come back empty, so `deserialize_state` leaves them as empty tuples.

## Agent Build Order

Build agents in this order — each is tested against the previous:

**Current status:** `RandomAgent` and `RuleBasedAgent` are implemented and verified
end-to-end (full six-player games to SCORING against a live server). `RuleBasedAgent`
plays legally but its heuristics are still shallow (passes trump, buries the first 6
kitty cards, simple lead/follow). `MCTSAgent` and `DouZeroAgent` are placeholders that
return the first legal action. Training (`training/`) is a placeholder.

### 1. RandomAgent (start here)
```python
class RandomAgent(BaseAgent):
    def act(self, state, legal_actions):
        return random.choice(legal_actions)
```
Simple but critical for testing the connection loop and training infrastructure.

### 2. RuleBasedAgent
Hard-coded heuristics. Implement rules in this priority order:
1. If leading: play highest tractor if available; else highest pair; else highest single
2. If partner is winning the trick: play lowest card to avoid waste
3. If opponents are winning: play highest trump to take the trick if it wins; else play lowest card
4. Protect red fives: never play ♥5 or ♦5 unless forced
5. Helper identity: if you're the helper (revealed), coordinate with dealer; if unknown, blend with farmers

```python
class RuleBasedAgent(BaseAgent):
    def act(self, state, legal_actions):
        if self._is_leading(state):
            return self._lead_action(state, legal_actions)
        else:
            return self._follow_action(state, legal_actions)
```

### 3. MCTSAgent
Uses Perfect Information Monte Carlo (PIMC):

```python
class MCTSAgent(BaseAgent):
    def __init__(self, num_simulations=500, num_worlds=20):
        self.num_simulations = num_simulations
        self.num_worlds = num_worlds

    def act(self, state, legal_actions):
        # Sample N possible "worlds" (what could other players hold)
        worlds = self._sample_worlds(state, n=self.num_worlds)
        # For each world, run MCTS and get action value estimates
        action_values = defaultdict(float)
        for world in worlds:
            values = self._mcts_search(world, legal_actions)
            for action, value in values.items():
                action_values[action] += value
        # Return action with highest average value
        return max(legal_actions, key=lambda a: action_values[a])

    def _sample_worlds(self, state, n):
        # Given our hand + cards we've seen, sample possible full deals
        # consistent with what we know
        ...
```

### 4. DouZeroAgent
Neural network agent. Architecture based on DouZero (Zha et al., 2021) adapted for 6-player 拖拉机.

```python
class DouZeroAgent(BaseAgent):
    def __init__(self, model_path: str, role: str, device: str = "mps"):
        self.model = ShengJiNet(role=role).to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.device = device

    def act(self, state, legal_actions):
        obs = encode_state(state)  # → torch.Tensor
        with torch.no_grad():
            action_values = self.model(obs.to(self.device))
        # Mask illegal actions, return argmax
        return legal_actions[action_values.argmax().item()]
```

## State Encoding for DouZero (encode_state)

The neural network needs a fixed-size vector. Encode the game state as:

```
My hand:          162 bits  (one bit per card in 3-deck set)
Cards seen:       162 bits  (cards played so far)
Current trick:    162 bits  (cards in current trick)
Trump info:       5 bits    (suit one-hot) + 11 bits (level one-hot)
My role:          3 bits    (dealer/helper/farmer one-hot)
My level:         ~60 bits  (level key one-hot over full LEVEL_SEQ)
Score so far:     1 float   (normalized 0–1)
Tricks remaining: 1 float   (normalized 0–1)
─────────────────────────────────────────
Total:            ~567 dimensions
```

## Training Architecture (training/)

Based on DouZero's actor-learner design:

```
┌─────────────────┐     shared     ┌──────────────────┐
│  Actor Process  │ → replay buffer → │ Learner Process  │
│  (CPU, 4-8x)   │                │  (MPS/CUDA, 1x)  │
│  self-play games│                │  gradient updates │
└─────────────────┘                └──────────────────┘
```

### Actor (actor.py)
```python
def run_actor(replay_buffer, model_weights, config):
    """Runs self-play games and pushes (state, action, reward) tuples to buffer."""
    agents = [DouZeroAgent(weights=model_weights, role=role) for role in ROLES]
    game = Game(num_players=6)
    while True:
        trajectory = run_episode(game, agents)
        for transition in trajectory:
            replay_buffer.put(transition)
```

### Learner (learner.py)
```python
def run_learner(replay_buffer, config):
    """Pulls batches from buffer, computes loss, updates model."""
    device = torch.device("mps")  # M1 GPU
    model = ShengJiNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    while True:
        batch = replay_buffer.get(batch_size=256)
        loss = compute_td_loss(model, batch, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Periodically push weights to actors
```

### Reward Function
```python
def compute_reward(player_id, level_changes: dict[int, int]) -> float:
    """
    Reward = level change for this player at end of game.
    Range: typically -3 to +3 per game.
    No intermediate rewards — only terminal.
    """
    return float(level_changes[player_id])
```

## M1 PyTorch Setup

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
tensor = tensor.to(device)

# Check MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"
```

Known MPS limitations:
- Some ops fall back to CPU silently — check with `PYTORCH_ENABLE_MPS_FALLBACK=1`
- For training stability, keep batch sizes ≤ 512
- Use `float32` not `float16` (MPS half-precision support is limited)

## Testing Agents

The engine is functional/immutable: `game.step(state, action)` returns `(state, info)`
and never mutates its input. There is no 4-tuple and no `done` flag — check
`info.get("game_over")` or `state.phase`.

```python
# test_random_agent.py
def test_random_agent_returns_legal_action():
    from shengji import Game
    game = Game(num_players=6)
    state = game.reset()
    agent = RandomAgent()
    for _ in range(1000):
        action = agent.act(state, state.legal_actions)
        assert action in state.legal_actions
        state, info = game.step(state, action)
        if info.get("game_over"):
            state = game.reset()

# test_rule_based_agent.py
def test_never_plays_red_five_voluntarily():
    # Set up a state where the agent has ♥5 and other options
    # Verify RuleBasedAgent doesn't play ♥5 unless forced

# End-to-end (manual): e2e_test.py drives six real agents through their run()
# loop against a live shengji-server and confirms a full game reaches SCORING.
```

## Common Mistakes to Avoid

- **Do not hardcode player roles** — in each game, any player can be dealer/helper/farmer
- **MCTS worlds must be consistent** — if you've seen player 2 play ♥7, don't sample worlds where they still hold it
- **DouZero needs separate models per role** — dealer strategy differs from farmer strategy; train three networks
- **Reward is per-player, not per-team** — even helpers get their own level changes
- **Never block the async event loop** — wrap `act()` in `run_in_executor` as shown in base_agent.py
