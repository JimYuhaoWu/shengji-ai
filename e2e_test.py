"""End-to-end test: drive the REAL base_agent.run() loop for 6 agents.

Not a unit test — a manual harness. Run with the server already up:
    python e2e_test.py --server ws://127.0.0.1:8001 --rest http://127.0.0.1:8001

Creates a room, connects six agents via their shipped run() loop, and polls the
room's REST status until it reaches SCORING (or times out). This exercises the
actual wire protocol, serialization, and KITTY handling in base_agent.py.
"""

import argparse
import asyncio
import logging

import httpx

from shengji_ai.random_agent import RandomAgent
from shengji_ai.rule_based_agent import RuleBasedAgent

AGENT_TYPES = {"random": RandomAgent, "rule_based": RuleBasedAgent}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("e2e")


async def poll_until_scoring(rest: str, room_id: str, timeout: float) -> dict:
    """Poll room status until phase==SCORING or timeout. Returns last status."""
    deadline = asyncio.get_event_loop().time() + timeout
    last = {}
    seen_phases = set()
    async with httpx.AsyncClient() as client:
        while asyncio.get_event_loop().time() < deadline:
            r = await client.get(f"{rest}/rooms/{room_id}")
            last = r.json()
            phase = last.get("game_phase")
            if phase not in seen_phases:
                seen_phases.add(phase)
                log.info(f"phase: {phase} (current_player={last.get('current_player')})")
            if phase == "SCORING":
                last["_phases_seen"] = sorted(seen_phases)
                return last
            await asyncio.sleep(0.3)
    last["_phases_seen"] = sorted(seen_phases)
    last["_timed_out"] = True
    return last


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="ws://127.0.0.1:8001")
    ap.add_argument("--rest", default="http://127.0.0.1:8001")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument(
        "--agents", nargs=6, default=["random"] * 6,
        help="6 agent types: random | rule_based",
    )
    args = ap.parse_args()

    async with httpx.AsyncClient() as client:
        r = await client.post(f"{args.rest}/rooms")
        room_id = r.json()["room_id"]
    log.info(f"Created room {room_id}")

    # Launch the REAL shipped agent loops.
    agents = [AGENT_TYPES[name]() for name in args.agents]
    tasks = [
        asyncio.create_task(agents[pid].run(args.server, room_id, pid))
        for pid in range(6)
    ]

    status = await poll_until_scoring(args.rest, room_id, args.timeout)

    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    print("\n==== E2E RESULT ====")
    if status.get("_timed_out"):
        print(f"Status: TIMED OUT at phase={status.get('game_phase')}")
    else:
        print(f"Status: REACHED SCORING")
    print(f"Phases seen: {status.get('_phases_seen')}")


if __name__ == "__main__":
    asyncio.run(main())
