"""CLI for running a full game with multiple AI agents."""

import asyncio
import argparse
import json
import logging
import urllib.request

from shengji_ai.random_agent import RandomAgent
from shengji_ai.rule_based_agent import RuleBasedAgent
from shengji_ai.mcts import MCTSAgent
from shengji_ai.douzero import DouZeroAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

AGENTS = {
    "random": RandomAgent,
    "rule_based": RuleBasedAgent,
    "mcts": MCTSAgent,
    "douzero": DouZeroAgent,
}


def _rest_base(server_url: str) -> str:
    """Derive the HTTP REST base URL from a ws:// server URL."""
    if server_url.startswith("wss://"):
        return "https://" + server_url[len("wss://"):]
    if server_url.startswith("ws://"):
        return "http://" + server_url[len("ws://"):]
    return server_url


def _create_room(server_url: str) -> str:
    """Create a room via POST /rooms and return its id.

    The server closes WebSocket connections to nonexistent rooms (code 4004),
    so the room must be created before any agent connects.
    """
    req = urllib.request.Request(f"{_rest_base(server_url)}/rooms", method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["room_id"]


async def main():
    parser = argparse.ArgumentParser(description="Run a full game with multiple agents")
    parser.add_argument(
        "--agents",
        nargs=6,
        required=True,
        help="6 agent types (random, rule_based, mcts, douzero)",
    )
    parser.add_argument(
        "--server",
        default="ws://localhost:8000",
        help="WebSocket server URL",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of games to play",
    )

    args = parser.parse_args()

    if len(args.agents) != 6:
        parser.error("Must specify exactly 6 agents")

    # Validate agent types
    for agent_type in args.agents:
        if agent_type not in AGENTS:
            parser.error(f"Unknown agent type: {agent_type}")

    logger.info(f"Starting {args.episodes} game(s) with agents: {args.agents}")

    # Create agent instances
    agents = []
    for agent_type in args.agents:
        agent_class = AGENTS[agent_type]
        if agent_type == "douzero":
            # DouZeroAgent needs model path
            agents.append(agent_class(model_path="models/douzero.pth", role="farmer"))
        else:
            agents.append(agent_class())

    # Run games
    for episode in range(args.episodes):
        # Create a fresh room server-side; agents can't connect otherwise.
        room_id = _create_room(args.server)
        logger.info(
            f"Starting episode {episode + 1}/{args.episodes} in room {room_id}"
        )

        # Connect all agents concurrently
        tasks = [
            agent.run(
                server_url=args.server,
                room_id=room_id,
                player_id=i,
            )
            for i, agent in enumerate(agents)
        ]

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in episode {episode + 1}: {e}")

    logger.info("All episodes completed")


if __name__ == "__main__":
    asyncio.run(main())
