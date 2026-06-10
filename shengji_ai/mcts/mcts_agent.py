"""MCTS agent (placeholder)."""

from shengji import GameState, Action

from shengji_ai.base_agent import BaseAgent


class MCTSAgent(BaseAgent):
    """Monte Carlo Tree Search agent (TODO: implement)."""

    def __init__(self, num_simulations: int = 500, num_worlds: int = 20):
        self.num_simulations = num_simulations
        self.num_worlds = num_worlds

    def act(self, state: GameState, legal_actions: list[Action]) -> Action:
        """Use MCTS to select best action (placeholder)."""
        # TODO: Implement MCTS
        return legal_actions[0]
