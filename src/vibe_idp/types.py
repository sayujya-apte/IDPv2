"""Core data types for the routing network.

These mirror the three routing concepts in the design spec:

* **Sender** -- the agent that generated the message.
* **Intended recipients** -- the nodes the sender asked to reach.
* **Payload** -- the information being transmitted.

`Message` additionally records `actual_recipients`, the intersection the router computed
between intended recipients and permitted destinations. The spec requires routing to depend
only on sender and intended recipients, never on payload content, so the payload is carried
as an opaque string and is never inspected during routing.
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RejectionReason = Literal[
    "unknown_recipient",
    "not_permitted",
    "no_recipients_requested",
]


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class OutboundMessage(BaseModel):
    """A message an agent wants to send: who it intends to reach, and what it wants to say.

    An agent chooses its own intended recipients but cannot bypass the topology: the router
    intersects this list with the sender's permitted destinations before anything is
    delivered. A recipient id that is not in the network is treated as invalid and dropped
    rather than raising, so a hallucinated id cannot break a run.

    The class docstring is serialized into the tool definition on every model call, so it is
    kept short on purpose. The rest of this reasoning lives in `router.py` and `DOC.md`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    intended_recipients: list[str] = Field(
        default_factory=list,
        description="Ids of the agents you want this message delivered to.",
    )
    payload: str = Field(default="", description="The information to transmit.")


class Message(BaseModel):
    """A message the router has validated and accepted for delivery.

    `intended_recipients` is preserved verbatim alongside `actual_recipients` so that a log
    consumer can see exactly what the sender asked for versus what the topology allowed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    message_id: str = Field(default_factory=_new_id)
    sender: str
    intended_recipients: list[str] = Field(default_factory=list)
    actual_recipients: list[str] = Field(default_factory=list)
    payload: str = ""
    cycle: int = 0


class Rejection(BaseModel):
    """A record of a requested destination the router refused to deliver to.

    Rejections are retained rather than silently discarded: they are the observable evidence
    that a sender could not address nodes outside its permitted communication range.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sender: str
    recipient: str
    reason: RejectionReason
    cycle: int = 0
    payload_preview: str = ""


class AgentDecision(BaseModel):
    """An agent's output for one cycle.

    Deliberately minimal: an agent only expresses who it wants to reach and what it wants to
    say. It has no way to express a delivery decision, because delivery belongs to the router.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    messages: list[OutboundMessage] = Field(
        default_factory=list,
        description="Messages to send this cycle. Empty if you have nothing to add.",
    )
    reasoning: str | None = Field(
        default=None, description="Brief justification for these choices."
    )


class RouteResult(BaseModel):
    """The router's output for one sender in one cycle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    messages: list[Message] = Field(default_factory=list)
    rejections: list[Rejection] = Field(default_factory=list)
