"""Communication topology derived from an adjacency matrix.

Per the design decision for this network, the adjacency matrix is treated as **undirected**:
if `A[i][j]` is truthy then `i` may send to `j` *and* `j` may send to `i`. The matrix is
symmetrised on construction, and any asymmetry in the caller's input is recorded rather than
silently discarded.

This module is pure: no LangGraph, no network, no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

MatrixValue = bool | int


class TopologyError(ValueError):
    """Raised when an adjacency matrix cannot be interpreted as a topology."""


class Topology:
    """An undirected, symmetric view of an adjacency matrix.

    The spec defines actual recipients as the intersection of intended recipients and
    *permitted destinations for that sender*. `permitted_recipients` is the single source of
    truth for that second term.
    """

    def __init__(
        self,
        adjacency: Sequence[Sequence[MatrixValue]],
        node_ids: Sequence[str] | None = None,
    ) -> None:
        self._size = self._validate_shape(adjacency)
        self._node_ids = self._resolve_ids(node_ids, self._size)
        self._edges = self._validate_and_symmetrise(adjacency)

    # ------------------------------------------------------------------ validation

    @staticmethod
    def _validate_shape(adjacency: Sequence[Sequence[MatrixValue]]) -> int:
        if isinstance(adjacency, (str, bytes, Mapping)):
            raise TopologyError("adjacency must be a sequence of rows, not a string or mapping")
        size = len(adjacency)
        if size == 0:
            raise TopologyError("adjacency matrix is empty; need at least one node")
        for i, row in enumerate(adjacency):
            if isinstance(row, (str, bytes, Mapping)):
                raise TopologyError(f"row {i} is not a sequence of values")
            if len(row) != size:
                raise TopologyError(
                    f"adjacency matrix must be square: row {i} has {len(row)} "
                    f"entries, expected {size}"
                )
        return size

    @staticmethod
    def _resolve_ids(node_ids: Sequence[str] | None, size: int) -> tuple[str, ...]:
        if node_ids is None:
            return tuple(f"node_{i}" for i in range(size))
        ids = tuple(str(x) for x in node_ids)
        if len(ids) != size:
            raise TopologyError(
                f"node_ids has {len(ids)} entries but the matrix describes {size} nodes"
            )
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise TopologyError(f"node_ids must be unique; duplicates: {dupes}")
        return ids

    def _validate_and_symmetrise(
        self, adjacency: Sequence[Sequence[MatrixValue]]
    ) -> tuple[frozenset[str], ...]:
        """Coerce to bool, then union each pair so the relation is symmetric.

        The diagonal is respected exactly as given: a truthy `A[i][i]` permits self-send, and
        symmetrisation never *adds* a self-edge where the caller supplied none.
        """
        raw: list[list[bool]] = []
        for i, row in enumerate(adjacency):
            coerced: list[bool] = []
            for j, value in enumerate(row):
                if not isinstance(value, (bool, int)) or isinstance(value, float):
                    raise TopologyError(
                        f"adjacency entry [{i}][{j}] must be a bool or int, got {value!r}"
                    )
                coerced.append(bool(value))
            raw.append(coerced)

        self.asymmetric_pairs: tuple[tuple[str, str], ...] = tuple(
            (self._node_ids[i], self._node_ids[j])
            for i in range(self._size)
            for j in range(i + 1, self._size)
            if raw[i][j] != raw[j][i]
        )

        edges: list[frozenset[str]] = []
        for i in range(self._size):
            neighbours: set[str] = set()
            for j in range(self._size):
                if raw[i][j] or raw[j][i]:
                    neighbours.add(self._node_ids[j])
            edges.append(frozenset(neighbours))
        return tuple(edges)

    # -------------------------------------------------------------------- queries

    @property
    def node_ids(self) -> tuple[str, ...]:
        return self._node_ids

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._node_ids

    def index_of(self, node_id: str) -> int:
        try:
            return self._node_ids.index(node_id)
        except ValueError:
            raise TopologyError(f"unknown node id: {node_id!r}") from None

    def permitted_recipients(self, sender: str) -> frozenset[str]:
        """Destinations `sender` is allowed to address, per the communication topology."""
        return self._edges[self.index_of(sender)]

    def is_symmetric_input(self) -> bool:
        """False if the caller's matrix was not already symmetric before unioning."""
        return not self.asymmetric_pairs

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_ids": list(self._node_ids),
            "permitted": {
                nid: sorted(peers) for nid, peers in zip(self._node_ids, self._edges, strict=True)
            },
            "input_was_asymmetric": not self.is_symmetric_input(),
        }

    def __repr__(self) -> str:
        return f"Topology(nodes={self._size}, ids={list(self._node_ids)!r})"
