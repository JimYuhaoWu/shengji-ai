"""Random agent that picks a legal action uniformly at random."""

import random

from shengji import GameState, Action

from shengji_ai.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Selects a uniformly random legal action."""

    def act(self, state: GameState, legal_actions: list[Action]) -> Action:
        """Return a random legal action."""
        return random.choice(legal_actions)
