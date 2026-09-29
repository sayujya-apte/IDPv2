"""End-to-end demonstration of the round-based routing model.

Run with:

    .venv/bin/python examples/demo.py

This uses a local stub agent so it needs no API key and no network. It is here to make the
round semantics and the router's enforcement visible in a single trace. Swap `StubAgent` for
`PurposeAgent` (see `live_demo()`) to run the same topology against Groq.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from vibe_idp import Message, OutboundMessage, PurposeAgent, Topology, run_network, summarize_log

PURPOSES = {
    "planner": "break the goal into concrete subtasks",
    "researcher": "gather facts for the subtasks you are given",
    "critic": "find weaknesses in proposed plans",
}


class StubAgent:
    """A deterministic stand-in for `PurposeAgent`, speaking the same protocol."""

    def __init__(self, node_id: str, peers: Sequence[str] = ()) -> None:
        self.node_id = node_id
        self.peers = list(peers)
        self.history: list[tuple[int, list[str]]] = []

    async def step(self, cycle: int, inbox: Sequence[Message]) -> list[OutboundMessage]:
        senders = [m.sender for m in inbox]
        self.history.append((cycle, senders))

        if not inbox:
            return []
        # Broadcast to every permitted peer (multicast), and also try to address a node we
        # may not reach plus one that does not exist -- the router must drop both.
        return [
            OutboundMessage(
                intended_recipients=[*self.peers, "ghost_node"],
                payload=f"{self.node_id} responding to {','.join(senders)}",
            )
        ]


def demo_topology() -> Topology:
    """Three fully connected specialists. Undirected, so every pair may address each other."""
    return Topology(
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
        node_ids=["planner", "researcher", "critic"],
    )


async def run_offline_demo() -> None:
    topo = demo_topology()
    agents = [StubAgent(n, sorted(topo.permitted_recipients(n))) for n in topo.node_ids]

    print("Topology (undirected):")
    for node, peers in topo.to_dict()["permitted"].items():
        print(f"  {node:12} -> {peers}")
    print()

    final = await run_network(
        agents,
        topo,
        max_rounds=4,
        seed_messages=[
            OutboundMessage(
                intended_recipients=["planner", "researcher"],
                payload="Goal: design a router",
            )
        ],
    )

    print("Per-cycle inbox contents (what each agent could actually read):")
    for agent in agents:
        trace = ", ".join(f"c{c}:{s or '-'}" for c, s in agent.history)
        print(f"  {agent.node_id:12} {trace}")
    print()

    final_cycle = max(e.cycle for e in final["log"] if isinstance(e, Message))
    print("Routed messages (produced in cycle n, readable from n+1):")
    for entry in final["log"]:
        if isinstance(entry, Message):
            note = "  <- final cycle: routed but never read" if entry.cycle == final_cycle else ""
            print(
                f"  cycle {entry.cycle}: {entry.sender:12} -> {entry.actual_recipients} "
                f"| {entry.payload[:40]}{note}"
            )
        else:
            print(
                f"  cycle {entry.cycle}: {entry.sender:12} -> {entry.recipient!r} "
                f"BLOCKED ({entry.reason})"
            )
    print()
    print(f"Completed {final['cycle'] - 1} cycles of {4} requested.")
    print("Summary:", summarize_log(final["log"]))


async def live_demo() -> None:  # pragma: no cover - requires GROQ_API_KEY
    """The same network driven by real Groq models."""
    topo = demo_topology()
    agents = [
        PurposeAgent(
            node_id=n,
            purpose=p,
            permitted_peers=sorted(topo.permitted_recipients(n)),
        )
        for n, p in PURPOSES.items()
    ]
    final = await run_network(agents, topo, max_rounds=3)
    print(summarize_log(final["log"]))


if __name__ == "__main__":
    asyncio.run(run_offline_demo())
