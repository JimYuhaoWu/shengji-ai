"""DouZero agent (placeholder)."""

from shengji import GameState, Action

from shengji_ai.base_agent import BaseAgent


class DouZeroAgent(BaseAgent):
    """Deep RL agent based on DouZero architecture (TODO: implement)."""

    def __init__(self, model_path: str, role: str, device: str = "cpu"):
        self.model_path = model_path
        self.role = role
        self.device = device
        # TODO: Load model

    def act(self, state: GameState, legal_actions: list[Action]) -> Action:
        """Use neural network to select action (placeholder)."""
        # TODO: Implement
        return legal_actions[0]
