"""Tests for RandomAgent."""

from shengji import Game, ActionType

from shengji_ai.random_agent import RandomAgent


def test_random_agent_returns_legal_action():
    """RandomAgent should always return a legal action."""
    agent = RandomAgent()
    game = Game(num_players=6)
    state = game.reset()

    # Play 1000 random actions to test
    for _ in range(1000):
        action = agent.act(state, state.legal_actions)
        assert action in state.legal_actions

        # Step the game
        state, info = game.step(state, action)

        # If game is over, reset
        if info.get("game_over"):
            state = game.reset()


def test_random_agent_handles_all_phases():
    """RandomAgent should work in all game phases."""
    agent = RandomAgent()
    game = Game(num_players=6)
    state = game.reset()

    phases_seen = set()

    for _ in range(100):
        phases_seen.add(state.phase.name)
        if state.legal_actions:
            action = agent.act(state, state.legal_actions)
            assert action in state.legal_actions
            state, info = game.step(state, action)

            if info.get("game_over"):
                state = game.reset()

    # Should have seen multiple phases
    assert len(phases_seen) > 1
