"""CLI for running a full game with multiple AI agents."""

import asyncio
import argparse
import logging

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


async def main():
    parser = argparse.ArgumentParser(description="Run a full game with multiple agents")
    parser.add_argument(
        "--agents",
        nargs=6,
        required=True,
        help="6 agent types (random, rule_based, mcts, douzero)",
    )
    parser.add_argument("--room", required=True, help="Room ID to create/join")
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
        logger.info(f"Starting episode {episode + 1}/{args.episodes}")

        # Connect all agents concurrently
        tasks = [
            agent.run(
                server_url=args.server,
                room_id=f"{args.room}_{episode}",
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
