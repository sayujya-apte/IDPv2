"""Driving the compiled network graph."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from .agent import Agent
from .graph import build_network_graph, recursion_limit_for
from .topology import Topology
from .types import Message, OutboundMessage, Rejection

logger = logging.getLogger(__name__)


async def run_network(
    agents: Sequence[Agent],
    topology: Topology,
    max_rounds: int,
    *,
    seed_messages: Sequence[OutboundMessage] | None = None,
    checkpointer: Any | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Run the network for `max_rounds` cycles and return the final state.

    Args:
        seed_messages: Optional first messages. These are placed in the outbox of node 0's
            channel and are delivered by the first router pass, so they are readable by their
            recipients in cycle 1 -- the same one-cycle delay that governs every other message.
        thread_id: Required only when a `checkpointer` is supplied.

    Returns:
        The final state, including `log` (every delivered `Message` and every `Rejection`),
        `cycle`, and the terminal `mailboxes`.
    """
    if not agents:
        raise ValueError("run_network requires at least one agent")

    graph = build_network_graph(agents, topology, max_rounds, checkpointer=checkpointer)

    initial: dict[str, Any] = {
        "cycle": 0,
        "mailboxes": {},
        "log": [],
    }
    for index in range(len(agents)):
        initial[f"outbox_{index}"] = []

    if seed_messages:
        initial["outbox_0"] = list(seed_messages)

    config: dict[str, Any] = {"recursion_limit": recursion_limit_for(max_rounds)}
    if checkpointer is not None and thread_id is not None:
        config["configurable"] = {"thread_id": thread_id}

    logger.info("running %d-agent network for %d rounds", len(agents), max_rounds)
    return await graph.ainvoke(initial, config)


def summarize_log(log: Sequence[Message | Rejection]) -> dict[str, Any]:
    """Condense a run's log into counts and a readable per-cycle timeline."""
    delivered = [e for e in log if isinstance(e, Message)]
    rejected = [e for e in log if isinstance(e, Rejection)]

    by_cycle: dict[int, dict[str, int]] = {}
    for entry in delivered:
        bucket = by_cycle.setdefault(entry.cycle, {"messages": 0, "copies": 0})
        bucket["messages"] += 1
        bucket["copies"] += len(entry.actual_recipients)
    for entry in rejected:
        by_cycle.setdefault(entry.cycle, {"messages": 0, "copies": 0})

    return {
        "delivered_messages": len(delivered),
        "delivered_copies": sum(len(e.actual_recipients) for e in delivered),
        "rejected_destinations": len(rejected),
        "rejections_by_reason": {
            reason: sum(1 for r in rejected if r.reason == reason)
            for reason in {r.reason for r in rejected}
        },
        "by_cycle": dict(sorted(by_cycle.items())),
    }
