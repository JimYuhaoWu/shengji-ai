"""Tests for RuleBasedAgent."""

from shengji import Game

from shengji_ai.rule_based_agent import RuleBasedAgent


def test_rule_based_agent_returns_legal_action():
    """RuleBasedAgent should always return a legal action."""
    agent = RuleBasedAgent()
    game = Game(num_players=6)
    state = game.reset()

    for _ in range(100):
        if state.legal_actions:
            action = agent.act(state, state.legal_actions)
            assert action in state.legal_actions
            state, info = game.step(state, action)

            if info.get("game_over"):
                state = game.reset()


def test_rule_based_agent_avoids_red_five():
    """RuleBasedAgent should avoid playing red 5s when possible."""
    agent = RuleBasedAgent()
    # This test requires setting up a specific game state
    # TODO: Implement once we have better GameState manipulation
    pass
