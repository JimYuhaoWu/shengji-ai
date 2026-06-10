"""Base agent class with WebSocket client loop."""

import asyncio
import json
import logging
from abc import ABC, abstractmethod

import websockets

from shengji import GameState, Action
from shengji_ai.protocol import deserialize_state, deserialize_action, action_to_dict

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    @abstractmethod
    def act(self, state: GameState, legal_actions: list[Action]) -> Action:
        """Choose an action given the game state and legal moves.

        Args:
            state: Current game state
            legal_actions: List of legal actions available to this player

        Returns:
            One of the legal actions
        """
        raise NotImplementedError

    async def run(
        self, server_url: str, room_id: str, player_id: int, timeout: float = 5.0
    ):
        """Connect to shengji-server and play a game.

        Args:
            server_url: WebSocket server URL (e.g., "ws://localhost:8000")
            room_id: Room to join
            player_id: Player seat (0-5)
            timeout: Max seconds for act() to complete
        """
        uri = f"{server_url}/ws/{room_id}/{player_id}"
        logger.info(f"Connecting to {uri}")

        try:
            async with websockets.connect(uri) as ws:
                logger.info(f"Connected. Listening for messages...")
                async for raw_msg in ws:
                    msg = json.loads(raw_msg)
                    msg_type = msg.get("type")
                    logger.debug(f"Received: {msg_type}")

                    if msg_type == "joined":
                        logger.info(
                            f"Joined room {msg['room_id']} as player {msg['player_id']}"
                        )
                    elif msg_type == "state_update":
                        is_our_turn = msg["current_player"] == player_id
                        legal_actions_list = msg.get("legal_actions")
                        legal_actions_truncated = msg.get("legal_actions_truncated", False)

                        if is_our_turn and legal_actions_list is not None:
                            # We have legal actions - normal case
                            state = deserialize_state(msg)
                            actions = [deserialize_action(a) for a in legal_actions_list]

                            try:
                                loop = asyncio.get_event_loop()
                                action = await asyncio.wait_for(
                                    loop.run_in_executor(None, self.act, state, actions),
                                    timeout=timeout,
                                )
                                logger.info(f"Sending action: {action}")
                                await ws.send(json.dumps(action_to_dict(action)))
                            except asyncio.TimeoutError:
                                logger.error(f"act() exceeded {timeout}s timeout")
                                raise

                        elif is_our_turn and legal_actions_truncated:
                            # KITTY phase: legal_actions is truncated, can't use act()
                            # For now, just skip - this requires special handling
                            logger.warning("KITTY phase: legal_actions truncated, skipping act()")
                            # TODO: Handle KITTY with semantic message
                    elif msg_type == "player_connected":
                        logger.info(f"Player {msg['player_id']} connected")
                    elif msg_type == "player_disconnected":
                        logger.info(f"Player {msg['player_id']} disconnected")
                    elif msg_type == "game_over":
                        logger.info(
                            f"Game over. Farmer score: {msg.get('farmer_score')}"
                        )
                    elif msg_type == "error":
                        logger.error(f"Server error: {msg.get('message')}")
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket error: {e}")
            raise
