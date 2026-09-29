"""Shared test fixtures.

The suite never calls Groq. Tests substitute `ScriptedAgent` for `PurposeAgent`, so routing and
round semantics are verified deterministically, with no API key and no cost.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from vibe_idp.types import Message, OutboundMessage


def out(recipients: Sequence[str], payload: str = "") -> OutboundMessage:
    """Terse constructor for tests (Pydantic v2 forbids positional args)."""
    return OutboundMessage(intended_recipients=list(recipients), payload=payload)


class ScriptedAgent:
    """An `Agent` whose output is canned and whose view of the network is recorded.

    Satisfies the `Agent` protocol, so it can be dropped straight into
    `build_network_graph` in place of a live `PurposeAgent`.
    """

    def __init__(
        self,
        node_id: str,
        script: dict[int, list[OutboundMessage]] | None = None,
        default: Sequence[OutboundMessage] = (),
    ) -> None:
        self.node_id = node_id
        self._script = dict(script or {})
        self._default = list(default)
        #: (cycle, [(sender, payload), ...]) for every cycle this agent was activated.
        self.inbox_history: list[tuple[int, list[tuple[str, str]]]] = []

    @property
    def cycles(self) -> list[int]:
        return [c for c, _ in self.inbox_history]

    def inbox_at(self, cycle: int) -> list[tuple[str, str]]:
        """(sender, payload) pairs this agent actually received in `cycle`."""
        for recorded_cycle, inbox in self.inbox_history:
            if recorded_cycle == cycle:
                return inbox
        raise AssertionError(f"{self.node_id} was not activated in cycle {cycle}")

    def senders_at(self, cycle: int) -> set[str]:
        return {s for s, _ in self.inbox_at(cycle)}

    async def step(self, cycle: int, inbox: Sequence[Message]) -> list[OutboundMessage]:
        self.inbox_history.append((cycle, [(m.sender, m.payload) for m in inbox]))
        return list(self._script.get(cycle, self._default))


class EchoAgent(ScriptedAgent):
    """Replies to everyone who wrote, and only to those. One hop per cycle."""

    def __init__(self, node_id: str, reply: str = "ack") -> None:
        super().__init__(node_id)
        self.reply = reply

    async def step(self, cycle: int, inbox: Sequence[Message]) -> list[OutboundMessage]:
        self.inbox_history.append((cycle, [(m.sender, m.payload) for m in inbox]))
        return [
            OutboundMessage(intended_recipients=[m.sender], payload=f"{self.reply}->{m.sender}")
            for m in inbox
        ]


class RelayAgent(ScriptedAgent):
    """Forwards whatever it receives on to a fixed next hop.

    Used to build a chain, so that the number of cycles a message takes to travel a given
    distance can be measured directly.
    """

    def __init__(self, node_id: str, target: str, prefix: str = "relay") -> None:
        super().__init__(node_id)
        self.target = target
        self.prefix = prefix

    async def step(self, cycle: int, inbox: Sequence[Message]) -> list[OutboundMessage]:
        self.inbox_history.append((cycle, [(m.sender, m.payload) for m in inbox]))
        return [
            OutboundMessage(intended_recipients=[self.target], payload=f"{self.prefix}:{m.payload}")
            for m in inbox
        ]


class ExplodingAgent(ScriptedAgent):
    """Raises, to prove the graph's behaviour when a node violates the Agent protocol."""

    async def step(self, cycle: int, inbox: Sequence[Message]) -> list[OutboundMessage]:
        raise RuntimeError("agent exploded")


@pytest.fixture
def ring3():
    """Three nodes in a line: node_0 - node_1 - node_2."""
    from vibe_idp import Topology

    return Topology([[0, 1, 0], [1, 0, 1], [0, 1, 0]])


@pytest.fixture
def pair():
    """Two mutually connected nodes."""
    from vibe_idp import Topology

    return Topology([[0, 1], [1, 0]])
