"""The routing rule: Actual = Intended n Permitted.

The router has no LangGraph dependency and no network dependency, so the spec's rules are
tested directly.
"""

from __future__ import annotations

import pytest

from conftest import out
from vibe_idp import CentralRouter, RouterError, Topology, TopologyError


def test_unicast_delivery():
    topo = Topology([[0, 1], [1, 0]], node_ids=["a", "b"])
    result = CentralRouter(topo).route("a", [out(["b"], "hi")])

    assert len(result.messages) == 1
    assert result.messages[0].actual_recipients == ["b"]
    assert result.rejections == []


def test_multicast_delivery():
    topo = Topology([[0, 1, 1], [1, 0, 1], [1, 1, 0]], node_ids=["a", "b", "c"])
    result = CentralRouter(topo).route("a", [out(["b", "c"], "everyone")])

    assert sorted(result.messages[0].actual_recipients) == ["b", "c"]


def test_unpermitted_destination_is_dropped_not_raised():
    topo = Topology([[0, 1, 0], [1, 0, 0], [0, 0, 0]], node_ids=["a", "b", "c"])
    result = CentralRouter(topo).route("a", [out(["b", "c"], "partial")])

    assert result.messages[0].actual_recipients == ["b"]
    assert [r.reason for r in result.rejections] == ["not_permitted"]
    assert result.rejections[0].recipient == "c"


def test_hallucinated_recipient_is_dropped():
    """A model inventing a node id must not break the run."""
    topo = Topology([[0, 1], [1, 0]], node_ids=["a", "b"])
    result = CentralRouter(topo).route("a", [out(["b", "node_99"], "x")])

    assert result.messages[0].actual_recipients == ["b"]
    assert [(r.recipient, r.reason) for r in result.rejections] == [
        ("node_99", "unknown_recipient")
    ]


def test_message_with_no_valid_recipients_produces_no_message():
    topo = Topology([[0, 1], [1, 0]], node_ids=["a", "b"])
    result = CentralRouter(topo).route("a", [out(["node_99"], "nobody")])

    assert result.messages == []
    assert [r.reason for r in result.rejections] == ["unknown_recipient"]


def test_empty_recipient_list_is_recorded():
    topo = Topology([[0, 1], [1, 0]], node_ids=["a", "b"])
    result = CentralRouter(topo).route("a", [out([], "thinking out loud")])

    assert result.messages == []
    assert [r.reason for r in result.rejections] == ["no_recipients_requested"]


def test_routing_is_independent_of_payload():
    """The spec: routing must not depend on payload content."""
    topo = Topology([[0, 1, 0], [1, 0, 1], [0, 1, 0]], node_ids=["a", "b", "c"])
    router = CentralRouter(topo)

    short = router.route("a", [out(["b", "c"], "hi")])
    long = router.route("a", [out(["b", "c"], "x" * 5000)])

    assert short.messages[0].actual_recipients == long.messages[0].actual_recipients
    assert [r.reason for r in short.rejections] == [r.reason for r in long.rejections]


def test_duplicate_recipients_are_collapsed_preserving_order():
    topo = Topology([[0, 1, 1], [1, 0, 1], [1, 1, 0]], node_ids=["a", "b", "c"])
    result = CentralRouter(topo).route("a", [out(["c", "b", "c"], "x")])

    assert result.messages[0].actual_recipients == ["c", "b"]


def test_deliver_places_one_copy_per_recipient():
    topo = Topology([[0, 1, 1], [1, 0, 1], [1, 1, 0]], node_ids=["a", "b", "c"])
    result = CentralRouter(topo).route("a", [out(["b", "c"], "x")])
    mailboxes = CentralRouter(topo).deliver(result)

    assert set(mailboxes) == {"b", "c"}
    assert mailboxes["b"] == mailboxes["c"] == result.messages


def test_deliver_returns_a_fresh_mapping_each_call():
    """Inboxes are refilled, never accumulated -- the old mapping must be untouched."""
    topo = Topology([[0, 1], [1, 0]], node_ids=["a", "b"])
    router = CentralRouter(topo)

    first = router.deliver(router.route("a", [out(["b"], "one")]))
    second = router.deliver(router.route("a", [out(["b"], "two")]))

    assert first["b"][0].payload == "one"
    assert second["b"][0].payload == "two"
    assert first is not second


def test_sender_must_be_a_node_in_the_topology():
    topo = Topology([[0, 1], [1, 0]], node_ids=["a", "b"])
    with pytest.raises(RouterError, match="not a node"):
        CentralRouter(topo).route("stranger", [])


def test_isolated_node_can_still_be_routed_for():
    topo = Topology([[0, 1, 0], [1, 0, 0], [0, 0, 0]], node_ids=["a", "b", "c"])
    result = CentralRouter(topo).route("c", [out(["a"], "x")])

    assert result.messages == []
    assert [r.reason for r in result.rejections] == ["not_permitted"]


def test_cycle_is_carried_onto_the_message():
    topo = Topology([[0, 1], [1, 0]], node_ids=["a", "b"])
    result = CentralRouter(topo).route("a", [out(["b"], "x")], cycle=7)

    assert result.messages[0].cycle == 7


def test_router_surfaces_the_topology_it_guards():
    topo = Topology([[0, 1], [1, 0]], node_ids=["a", "b"])
    router = CentralRouter(topo)

    assert router.topology is topo
    assert router.permitted_recipients("a") == frozenset({"b"})
    with pytest.raises(TopologyError):
        router.permitted_recipients("missing")
