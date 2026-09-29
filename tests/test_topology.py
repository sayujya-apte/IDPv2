"""Adjacency matrix handling."""

from __future__ import annotations

import pytest

from vibe_idp import Topology, TopologyError


def test_matrix_is_treated_as_undirected():
    """A one-way edge in the input permits traffic both ways."""
    topo = Topology([[0, 1], [0, 0]], node_ids=["a", "b"])

    assert topo.permitted_recipients("a") == frozenset({"b"})
    assert topo.permitted_recipients("b") == frozenset({"a"})
    assert topo.asymmetric_pairs == (("a", "b"),)
    assert topo.is_symmetric_input() is False


def test_symmetric_input_reports_no_asymmetry():
    topo = Topology([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    assert topo.is_symmetric_input() is True
    assert topo.asymmetric_pairs == ()


def test_permitted_set_is_the_union_of_both_directions():
    topo = Topology([[0, 1, 0], [0, 0, 1], [0, 0, 0]], node_ids=["a", "b", "c"])
    # a->b only, b->c only
    assert topo.permitted_recipients("a") == frozenset({"b"})
    assert topo.permitted_recipients("b") == frozenset({"a", "c"})
    assert topo.permitted_recipients("c") == frozenset({"b"})


def test_isolated_node_has_no_permitted_recipients():
    topo = Topology([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    assert topo.permitted_recipients("node_2") == frozenset()


def test_diagonal_is_respected_but_never_added():
    """Symmetrisation must not invent a self-edge where the caller gave none."""
    without = Topology([[0, 1], [1, 0]], node_ids=["a", "b"])
    with_self = Topology([[1, 1], [1, 0]], node_ids=["a", "b"])

    assert "a" not in without.permitted_recipients("a")
    assert "a" in with_self.permitted_recipients("a")


def test_non_square_matrix_is_rejected():
    with pytest.raises(TopologyError, match="square"):
        Topology([[0, 1, 0], [1, 0]])


def test_empty_matrix_is_rejected():
    with pytest.raises(TopologyError, match="empty"):
        Topology([])


def test_duplicate_node_ids_are_rejected():
    with pytest.raises(TopologyError, match="unique"):
        Topology([[0, 1], [1, 0]], node_ids=["a", "a"])


def test_node_id_count_must_match_matrix():
    with pytest.raises(TopologyError, match="node_ids"):
        Topology([[0, 1], [1, 0]], node_ids=["a", "b", "c"])


def test_non_integer_entries_are_rejected():
    with pytest.raises(TopologyError, match="bool or int"):
        Topology([[0, 0.5], [0, 0]])


def test_default_node_ids_are_indexed():
    assert Topology([[0, 1], [1, 0]]).node_ids == ("node_0", "node_1")


def test_unknown_node_lookup_raises():
    topo = Topology([[0, 1], [1, 0]])
    assert "node_0" in topo
    assert "nope" not in topo
    with pytest.raises(TopologyError, match="unknown node id"):
        topo.index_of("nope")


def test_to_dict_is_serialisable():
    topo = Topology([[0, 1], [1, 0]], node_ids=["a", "b"])
    assert topo.to_dict() == {
        "node_ids": ["a", "b"],
        "permitted": {"a": ["b"], "b": ["a"]},
        "input_was_asymmetric": False,
    }
