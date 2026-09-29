"""Graph assembly: dynamic schema, validation, and structural shape."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from conftest import ScriptedAgent, out
from vibe_idp import (
    GraphBuildError,
    Message,
    Topology,
    build_network_graph,
    recursion_limit_for,
    run_network,
    summarize_log,
)
from vibe_idp.graph import ROUTER_NODE, build_state_schema


def agents_for(n: int = 3) -> list[ScriptedAgent]:
    return [ScriptedAgent(f"node_{i}") for i in range(n)]


# ------------------------------------------------------------------ dynamic schema


@pytest.mark.parametrize("n", [1, 2, 3, 7])
def test_schema_has_one_outbox_channel_per_agent(n):
    schema = build_state_schema([f"node_{i}" for i in range(n)])

    assert "cycle" in schema.__annotations__
    assert "mailboxes" in schema.__annotations__
    assert "log" in schema.__annotations__
    for i in range(n):
        assert f"outbox_{i}" in schema.__annotations__


def test_schema_tracks_the_adjacency_matrix():
    """Node count is data: a 5-node matrix produces a 5-agent schema."""
    topo = Topology([[0 if i == j else 1 for j in range(5)] for i in range(5)])
    graph = build_network_graph(agents_for(5), topo, 2)

    assert len(graph.get_graph().nodes) == 8  # 5 agents + router + __start__ + __end__


# --------------------------------------------------------------------- structure


def test_topology_is_not_encoded_as_graph_edges(ring3):
    """The key structural decision.

    LangGraph edges mean "run this node next". The adjacency matrix means "this node is
    allowed to send there". Conflating them would make every neighbour a trigger, so a node
    would be scheduled even when nothing was sent to it. The topology must therefore live
    only in the router.
    """
    graph = build_network_graph(agents_for(3), ring3, 2)
    edges = {(e.source, e.target) for e in graph.get_graph().edges}

    for a in ("node_0", "node_1", "node_2"):
        for b in ("node_0", "node_1", "node_2"):
            if a != b:
                assert (a, b) not in edges, f"unexpected topology edge {a} -> {b}"

    assert (ROUTER_NODE, "node_0") in edges
    assert ("node_0", ROUTER_NODE) in edges


def test_router_is_the_only_entry_point(ring3):
    graph = build_network_graph(agents_for(3), ring3, 2)
    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    starts = {s for s, _ in edges}

    assert "__start__" in starts
    assert all(s in {ROUTER_NODE, "__start__"} or s.startswith("node_") for s in starts)


# --------------------------------------------------------------------- validation


def test_max_rounds_must_be_positive(ring3):
    with pytest.raises(GraphBuildError, match="max_rounds"):
        build_network_graph(agents_for(3), ring3, 0)


def test_every_node_needs_an_agent(ring3):
    with pytest.raises(GraphBuildError, match="no agent"):
        build_network_graph(agents_for(2), ring3, 2)


def test_agents_must_not_exceed_the_topology(ring3):
    extra = [*agents_for(3), ScriptedAgent("node_9")]
    with pytest.raises(GraphBuildError, match="absent from the topology"):
        build_network_graph(extra, ring3, 2)


def test_duplicate_agent_ids_are_rejected(ring3):
    dupes = [ScriptedAgent("node_0"), ScriptedAgent("node_0"), ScriptedAgent("node_2")]
    with pytest.raises(GraphBuildError, match="unique"):
        build_network_graph(dupes, ring3, 2)


def test_node_ids_must_be_usable_as_graph_names(ring3):
    bad = [ScriptedAgent("node 0"), ScriptedAgent("node_1"), ScriptedAgent("node_2")]
    with pytest.raises(GraphBuildError, match="graph node names"):
        build_network_graph(bad, ring3, 2)


def test_run_network_requires_agents(ring3):
    with pytest.raises(ValueError, match="at least one agent"):
        import asyncio

        asyncio.run(run_network([], ring3, 2))


# ------------------------------------------------------------------- run plumbing


def test_recursion_limit_leaves_headroom():
    """Termination is the conditional edge; the limit is only a backstop."""
    assert recursion_limit_for(1) == 14
    assert recursion_limit_for(10) == 50
    assert recursion_limit_for(10) > 2 * 10 + 2


@pytest.mark.asyncio
async def test_seed_messages_are_delivered_like_any_other(ring3):
    """Seed messages are subject to the same one-cycle delay as generated ones."""
    agents = agents_for(3)
    final = await run_network(agents, ring3, max_rounds=2, seed_messages=[out(["node_1"], "seed")])

    assert agents[1].senders_at(1) == {"node_0"}
    assert agents[0].inbox_at(1) == [], "the sender does not receive its own message"
    assert any(isinstance(e, Message) and e.payload == "seed" for e in final["log"])


@pytest.mark.asyncio
async def test_checkpointer_records_every_superstep(pair):
    saver = InMemorySaver()
    agents = [ScriptedAgent("node_0", script={1: [out(["node_1"], "x")]}), ScriptedAgent("node_1")]

    await run_network(agents, pair, max_rounds=3, checkpointer=saver, thread_id="t1")

    graph = build_network_graph(agents, pair, 3, checkpointer=saver)
    history = list(graph.get_state_history({"configurable": {"thread_id": "t1"}}))

    # get_state_history is newest-first.
    assert history[0].next == (), "the run should have completed"
    assert history[-1].next == ("__start__",), "oldest checkpoint precedes the first superstep"

    # Every checkpoint is a superstep boundary, and each is either a router pass or a round
    # in which both agents ran together.
    router_passes = [s for s in history if s.next == (ROUTER_NODE,)]
    agent_rounds = [s for s in history if set(s.next) == {"node_0", "node_1"}]

    assert len(agent_rounds) == 3, "one agent superstep per round"
    assert len(router_passes) == 4, "max_rounds passes plus the final pass that exits"

    # pre-run + (router -> agents) x max_rounds + the closing router pass
    assert len(history) == 2 * 3 + 3


def test_summarize_log_counts_deliveries_and_rejections():
    topo = Topology([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    agents = [
        ScriptedAgent("node_0", script={1: [out(["node_1", "node_2"], "x")]}),
        ScriptedAgent("node_1"),
        ScriptedAgent("node_2"),
    ]

    import asyncio

    final = asyncio.run(run_network(agents, topo, 2))
    summary = summarize_log(final["log"])

    assert summary["delivered_messages"] == 1
    assert summary["delivered_copies"] == 1
    assert summary["rejections_by_reason"] == {"not_permitted": 1}
    assert 1 in summary["by_cycle"]


@pytest.mark.asyncio
async def test_single_node_network_terminates():
    """A 1x1 matrix with no self-send still runs and completes."""
    topo = Topology([[0]])
    agent = ScriptedAgent("node_0", script={1: [out(["node_0"], "self")]})

    await run_network([agent], topo, 2)

    assert agent.cycles == [1, 2]
    assert all(not inbox for _c, inbox in agent.inbox_history)
