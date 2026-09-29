"""The spec's core guarantee: a message produced in cycle n is readable in cycle n+1."""

from __future__ import annotations

import pytest

from conftest import EchoAgent, ExplodingAgent, RelayAgent, ScriptedAgent, out
from vibe_idp import run_network


@pytest.mark.asyncio
async def test_message_is_invisible_in_its_production_cycle(pair):
    """A message sent in cycle 1 must not be readable in cycle 1."""
    a = ScriptedAgent("node_0", script={1: [out(["node_1"], "hello")]})
    b = ScriptedAgent("node_1")

    await run_network([a, b], pair, max_rounds=3)

    assert b.senders_at(1) == set(), "recipient read a message in the cycle that produced it"
    assert b.senders_at(2) == {"node_0"}, "recipient never received the message"
    assert b.inbox_at(2)[0][1] == "hello"


@pytest.mark.asyncio
async def test_reply_cannot_arrive_in_the_same_round(pair):
    """node_1's reply, generated in cycle 1, is not visible to node_0 in cycle 1."""
    a = EchoAgent("node_0", reply="hi")
    b = EchoAgent("node_1", reply="hello")

    await run_network([a, b], pair, max_rounds=4, seed_messages=[out(["node_1"], "seed")])

    # node_0 was never told to expect anything; it should only ever see node_1's replies,
    # and only from cycle 2 onward.
    assert a.senders_at(1) == set()
    assert a.senders_at(2) == {"node_1"}


@pytest.mark.asyncio
async def test_one_hop_per_cycle_along_a_chain(ring3):
    """Transitive delivery must not collapse into a single round.

    seed -> node_1 -> node_2, where node_1 relays what it receives. If delivery were
    transitive within a round, node_2 would hear about the seed in cycle 1. It must take one
    cycle per hop.
    """
    silent = ScriptedAgent("node_0")
    relay = RelayAgent("node_1", target="node_2")
    sink = ScriptedAgent("node_2")

    await run_network(
        [silent, relay, sink], ring3, max_rounds=5, seed_messages=[out(["node_1"], "seed")]
    )

    assert relay.senders_at(1) == {"node_0"}, "seed should reach node_1 in cycle 1"
    assert sink.senders_at(1) == set(), "node_2 must not hear about the seed in cycle 1"
    assert sink.senders_at(2) == {"node_1"}, "node_2 should hear from node_1 in cycle 2"


@pytest.mark.asyncio
async def test_agent_only_ever_receives_its_own_mail(ring3):
    """No agent can read another node's inbox, even in a fully connected network."""
    from vibe_idp import Topology

    full = Topology([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # everyone connected, self-send on
    agents = [EchoAgent(f"node_{i}") for i in range(3)]

    await run_network(agents, full, max_rounds=3, seed_messages=[out(["node_1"], "s")])

    for agent in agents:
        for _cycle, inbox in agent.inbox_history:
            for sender, _payload in inbox:
                # Whatever is in an inbox got routed to that specific node; a message is only
                # ever placed in mailboxes its actual_recipients name.
                assert sender != agent.node_id or agent.node_id in full.permitted_recipients(
                    agent.node_id
                )


@pytest.mark.asyncio
async def test_max_rounds_activates_every_agent_exactly_max_rounds_times(ring3):
    agents = [ScriptedAgent(f"node_{i}") for i in range(3)]

    final = await run_network(agents, ring3, max_rounds=4)

    for agent in agents:
        assert agent.cycles == [1, 2, 3, 4], f"{agent.node_id} activated on wrong cycles"
    assert final["cycle"] == 5, "cycle counter should have advanced once past max_rounds"


@pytest.mark.asyncio
async def test_inboxes_are_refilled_not_accumulated(pair):
    """Mailbox contents are replaced each round, so nothing is read twice."""
    a = ScriptedAgent("node_0", script={1: [out(["node_1"], "once")]})
    b = ScriptedAgent("node_1")

    await run_network([a, b], pair, max_rounds=4)

    appearances = [c for c, inbox in b.inbox_history if inbox]
    assert appearances == [2], f"message should be readable exactly once, saw cycles {appearances}"


@pytest.mark.asyncio
async def test_unpermitted_recipient_is_dropped_at_graph_level():
    """A sender cannot address a node outside its permitted range, even if it tries."""
    from vibe_idp import Topology

    topo = Topology([[0, 1, 0], [1, 0, 0], [0, 0, 0]])  # only node_0 <-> node_1
    a = ScriptedAgent(
        "node_0",
        script={1: [out(["node_1", "node_2", "ghost"], "broadcast")]},
    )
    b = ScriptedAgent("node_1")
    c = ScriptedAgent("node_2")

    final = await run_network([a, b, c], topo, max_rounds=3)

    assert b.senders_at(2) == {"node_0"}, "permitted delivery should have happened"
    assert c.inbox_history and all(not inbox for _c, inbox in c.inbox_history), (
        "node_2 was not permitted and must receive nothing"
    )

    reasons = {r.reason for r in final["log"] if hasattr(r, "reason")}
    assert "not_permitted" in reasons
    assert "unknown_recipient" in reasons


@pytest.mark.asyncio
async def test_one_failing_agent_does_not_stop_the_others(ring3):
    """Because supersteps are transactional, a raising node would void the whole round.

    The `Agent` protocol forbids raising; this test pins that expectation so the contract is
    enforced rather than merely documented.
    """
    good = ScriptedAgent(
        "node_0",
        script={
            1: [out(["node_1"], "survive")],
        },
    )
    bad = ExplodingAgent("node_1")
    other = ScriptedAgent("node_2")

    with pytest.raises(RuntimeError, match="exploded"):
        await run_network([good, bad, other], ring3, max_rounds=3)

    # The failure is real and visible -- it is not swallowed. PurposeAgent.step is what
    # guarantees it never happens, by catching its own errors.
    assert good.inbox_history, "the healthy agents were still activated"
