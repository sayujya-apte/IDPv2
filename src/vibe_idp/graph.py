"""Assembles a LangGraph network from a topology and a set of agents.

Cycle structure
---------------
A LangGraph **superstep boundary is a communication cycle boundary**. The Pregel runtime runs
each superstep transactionally: writes made during a superstep are not visible to any actor
until the next superstep begins. That is precisely the guarantee the design spec asks for --
a message produced in cycle `n` becomes available at the start of cycle `n+1` -- so the round
model is enforced by the runtime rather than by application code.

    START -> router -> agent_0  |  one superstep, all agents in parallel
              ^        agent_1  |
              |        ...      |
              |        agent_N  |
              +------------------  conditional edge to END once cycle > max_rounds

Why the adjacency matrix is *not* encoded as LangGraph edges
-------------------------------------------------------------
LangGraph edges express **control flow**. The adjacency matrix is a **validation
constraint**. If `A[0][2] == 0` then node 2 must simply never receive a message from node 0 --
it must not be "triggered" by an edge from node 0, because an edge would schedule node 2 to
run regardless of whether node 0 actually sent anything. The spec is explicit that the router
decides delivery, not the graph scheduler, so the topology lives in `router.py` and the only
edges here encode the round cycle.

Channel ownership
-----------------
Every state channel has exactly **one** writer and uses full-overwrite semantics. This is not
stylistic: two agents writing the same channel in one superstep raises `InvalidUpdateError`
("can receive only one value per step"), and adding a reducer to paper over it would
reintroduce the concurrent-write problem the design is trying to avoid. Per-node `outbox_*`
channels plus a router-only `mailboxes` channel make `InvalidUpdateError` structurally
impossible.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .agent import Agent
from .router import CentralRouter
from .topology import Topology
from .types import Message, Rejection, RouteResult

ROUTER_NODE = "router"
_SAFE_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class GraphBuildError(ValueError):
    """Raised when the topology and the agent set cannot form a network."""


def _outbox_key(index: int) -> str:
    return f"outbox_{index}"


def build_state_schema(node_ids: Sequence[str]) -> type:
    """Construct the state schema for a given node set.

    The schema is dynamic because the node count is data: the adjacency matrix decides how
    many agents exist. One `outbox_<i>` channel per agent index keeps every channel
    single-writer.
    """
    annotations: dict[str, Any] = {
        "cycle": int,
        "mailboxes": dict[str, list[Message]],
        "log": list[Message | Rejection],
    }
    annotations.update({_outbox_key(i): list[Any] for i in range(len(node_ids))})
    return TypedDict("NetworkState", annotations)  # type: ignore[operator]


def _validate(agents: Sequence[Agent], topology: Topology, max_rounds: int) -> None:
    if max_rounds < 1:
        raise GraphBuildError(f"max_rounds must be >= 1, got {max_rounds}")

    agent_ids = [a.node_id for a in agents]
    if len(set(agent_ids)) != len(agent_ids):
        dupes = sorted({i for i in agent_ids if agent_ids.count(i) > 1})
        raise GraphBuildError(f"agent node_ids must be unique; duplicates: {dupes}")

    bad = [i for i in agent_ids if not _SAFE_ID.match(i)]
    if bad:
        raise GraphBuildError(
            f"node ids must match {_SAFE_ID.pattern} to be usable as graph node names; got {bad}"
        )

    missing = sorted(set(topology.node_ids) - set(agent_ids))
    if missing:
        raise GraphBuildError(f"topology has nodes with no agent: {missing}")
    extra = sorted(set(agent_ids) - set(topology.node_ids))
    if extra:
        raise GraphBuildError(f"agents reference nodes absent from the topology: {extra}")


def build_network_graph(
    agents: Sequence[Agent],
    topology: Topology,
    max_rounds: int,
    *,
    checkpointer: Any | None = None,
) -> Any:
    """Build the compiled round-based network graph.

    Args:
        agents: One agent per node in `topology`. They may be `PurposeAgent` instances or any
            object satisfying the `Agent` protocol.
        topology: The communication topology, built from the adjacency matrix.
        max_rounds: How many cycles agents participate in. The run stops when the cycle
            counter exceeds this, decided by a conditional edge rather than by the recursion
            limit.
        checkpointer: Optional LangGraph checkpointer for per-superstep persistence.
    """
    _validate(agents, topology, max_rounds)

    node_ids = [a.node_id for a in agents]
    router = CentralRouter(topology)
    state_schema = build_state_schema(node_ids)
    outbox_keys = {agent.node_id: _outbox_key(i) for i, agent in enumerate(agents)}

    async def router_node(state: Mapping[str, Any]) -> dict[str, Any]:
        """Validate the previous cycle's outboxes and refill every inbox.

        The router is the sole writer of `mailboxes` and `log`. It returns a *fresh* mailbox
        mapping rather than mutating the previous one, so "inboxes are refilled, not
        accumulated" is structural rather than a rule to remember.
        """
        cycle: int = state["cycle"]
        messages: list[Message] = []
        rejections: list[Rejection] = []

        for node_id in node_ids:
            result = router.route(node_id, state[outbox_keys[node_id]], cycle)
            messages.extend(result.messages)
            rejections.extend(result.rejections)

        mailboxes = router.deliver(RouteResult(messages=messages, rejections=rejections))

        update: dict[str, Any] = {
            "cycle": cycle + 1,
            "mailboxes": mailboxes,
            "log": [*state["log"], *messages, *rejections],
        }
        # Clear every outbox in the same step that consumes it.
        update.update({key: [] for key in outbox_keys.values()})
        return update

    def route_fn(state: Mapping[str, Any]) -> str | list[str]:
        """Stop once the cycle counter has passed `max_rounds`.

        The router increments `cycle` on entry, so `cycle` already counts the activation that
        is about to happen. Comparing with `>` therefore yields exactly `max_rounds`
        activations per agent.
        """
        if state["cycle"] > max_rounds:
            return END
        return list(node_ids)

    builder = StateGraph(state_schema)
    builder.add_node(ROUTER_NODE, router_node)

    for agent in agents:
        key = outbox_keys[agent.node_id]

        async def agent_node(
            state: Mapping[str, Any], agent: Agent = agent, key: str = key
        ) -> dict[str, Any]:
            # An agent may only ever read its own inbox.
            inbox = state["mailboxes"].get(agent.node_id, [])
            outbounds = await agent.step(state["cycle"], inbox)
            return {key: list(outbounds)}

        builder.add_node(agent.node_id, agent_node)

    builder.add_edge(START, ROUTER_NODE)
    # A conditional edge, not a static edge from the router: the destination depends on state.
    builder.add_conditional_edges(ROUTER_NODE, route_fn, {**{n: n for n in node_ids}, END: END})
    # Fan-in: the router waits until every agent has finished the round.
    for node_id in node_ids:
        builder.add_edge(node_id, ROUTER_NODE)

    return builder.compile(checkpointer=checkpointer)


def recursion_limit_for(max_rounds: int) -> int:
    """A safety backstop only. Termination is the conditional edge, not this limit.

    One round costs two supersteps (router, then all agents in parallel), plus a final router
    call to exit. The headroom here exists so that a genuine bug loops into a clear
    `GraphRecursionError` instead of a truncated but plausible-looking result.
    """
    return 4 * max_rounds + 10
