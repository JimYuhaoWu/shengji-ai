"""CLI for running a single agent."""

import asyncio
import argparse
import logging

from shengji_ai.random_agent import RandomAgent
from shengji_ai.rule_based_agent import RuleBasedAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


AGENTS = {
    "random": RandomAgent,
    "rule_based": RuleBasedAgent,
}


async def main():
    parser = argparse.ArgumentParser(description="Run a single AI agent")
    parser.add_argument(
        "--agent",
        choices=AGENTS.keys(),
        default="random",
        help="Agent type to run",
    )
    parser.add_argument("--room", required=True, help="Room ID to join")
    parser.add_argument(
        "--player", type=int, required=True, help="Player seat (0-5)"
    )
    parser.add_argument(
        "--server",
        default="ws://localhost:8000",
        help="WebSocket server URL",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Timeout for act() in seconds",
    )

    args = parser.parse_args()

    agent_class = AGENTS[args.agent]
    agent = agent_class()

    await agent.run(
        server_url=args.server,
        room_id=args.room,
        player_id=args.player,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    asyncio.run(main())
