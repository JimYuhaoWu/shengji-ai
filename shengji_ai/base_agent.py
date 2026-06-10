"""Base agent class with WebSocket client loop."""

import asyncio
import json
import logging
from abc import ABC, abstractmethod

import websockets

from shengji import GameState, Action
from shengji.card import Card
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

    def choose_kitty_bury(self, state: GameState, hand: list[Card]) -> list[Card]:
        """Choose 6 cards to bury during the KITTY phase.

        The KITTY phase has C(32,6) ≈ 906k legal actions, so the server never
        sends them over the wire (legal_actions is truncated). Instead the agent
        is given the dealer's full 32-card hand and must return 6 cards to bury.

        Default: bury the first 6 cards. Subclasses should override with smarter
        logic (e.g. bury low non-scoring cards).

        Args:
            state: Current game state (KITTY phase)
            hand: The dealer's full 32-card hand

        Returns:
            Exactly 6 cards from the hand to bury.
        """
        return hand[:6]

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

                        if is_our_turn and legal_actions_list:
                            # We have legal actions - normal case (non-empty list).
                            # An empty list (e.g. SCORING phase) means nothing to do.
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
                            # KITTY phase: legal_actions is truncated (~906k options).
                            # The server gives us the dealer's full hand instead; we
                            # pick 6 cards to bury and send a semantic take_kitty message.
                            state = deserialize_state(msg)
                            hand = list(state.hands[player_id])
                            loop = asyncio.get_event_loop()
                            bury = await asyncio.wait_for(
                                loop.run_in_executor(
                                    None, self.choose_kitty_bury, state, hand
                                ),
                                timeout=timeout,
                            )
                            logger.info(f"Burying {len(bury)} cards (KITTY)")
                            await ws.send(json.dumps({
                                "type": "take_kitty",
                                "cards": [
                                    {
                                        "suit": c.suit.value,
                                        "rank": c.rank.value,
                                        "deck_id": c.deck_id,
                                    }
                                    for c in bury
                                ],
                            }))
                    elif msg_type == "player_connected":
                        logger.info(f"Player {msg['player_id']} connected")
                    elif msg_type == "player_disconnected":
                        logger.info(f"Player {msg['player_id']} disconnected")
                    elif msg_type == "game_over":
                        logger.info(
                            f"Game over. Farmer score: {msg.get('farmer_score')}"
                        )
                        # The game is finished; stop listening so callers (e.g.
                        # run_game) can proceed instead of blocking forever.
                        return
                    elif msg_type == "error":
                        logger.error(f"Server error: {msg.get('message')}")
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket error: {e}")
            raise
