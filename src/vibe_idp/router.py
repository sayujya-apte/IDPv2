"""Centralized message routing, per the design spec.

This module implements the routing rule and nothing else:

    Actual Recipients = Intended Recipients n Permitted Destinations

Design constraints taken directly from the spec:

* The sending node never delivers a message itself. It only names intended recipients.
* The router validates destinations against the communication relationships permitted for that
  sender, then forwards to every valid recipient (unicast when there is one, multicast when
  there are several).
* The router places messages into mailboxes. It never executes the receiving node.
* Routing is independent of payload content. `route` never reads `OutboundMessage.payload`
  except to record a truncated preview on rejected messages for debuggability.

**No LangGraph imports here, deliberately.** The spec requires the communication layer to be
independent of node application logic. Keeping this module framework-free means the rules that
actually contain logic are unit-testable with no checkpointer, no event loop, and no API key.
"""

from __future__ import annotations

from collections.abc import Iterable

from .topology import Topology
from .types import Message, OutboundMessage, Rejection, RouteResult

_PREVIEW_CHARS = 120


class RouterError(ValueError):
    """Raised when the router is asked to route for a node outside the topology."""


class CentralRouter:
    """Validates and forwards messages produced by agents in the network.

    One router serves the whole network. Nodes cannot bypass it, because a node has no
    reference to it and no path to a recipient's mailbox.
    """

    def __init__(self, topology: Topology) -> None:
        self._topology = topology

    @property
    def topology(self) -> Topology:
        return self._topology

    def permitted_recipients(self, sender: str) -> frozenset[str]:
        """Expose the permitted set for prompt construction and tests."""
        return self._topology.permitted_recipients(sender)

    def route(
        self, sender: str, outbounds: Iterable[OutboundMessage], cycle: int = 0
    ) -> RouteResult:
        """Validate and forward every message `sender` produced in `cycle`.

        Returns the messages that passed validation together with a record of every requested
        destination that was refused. A message whose recipients are *all* invalid produces no
        `Message` at all -- it appears only as rejections.
        """
        if sender not in self._topology:
            raise RouterError(f"sender {sender!r} is not a node in this topology")

        permitted = self._topology.permitted_recipients(sender)
        messages: list[Message] = []
        rejections: list[Rejection] = []

        for outbound in outbounds:
            accepted: list[str] = []
            seen: set[str] = set()

            for recipient in outbound.intended_recipients:
                # Deduplicate while preserving the sender's ordering.
                if recipient in seen:
                    continue
                seen.add(recipient)

                if recipient not in self._topology:
                    reason = "unknown_recipient"
                elif recipient not in permitted:
                    reason = "not_permitted"
                else:
                    accepted.append(recipient)
                    continue

                rejections.append(
                    Rejection(
                        sender=sender,
                        recipient=recipient,
                        reason=reason,
                        cycle=cycle,
                        payload_preview=outbound.payload[:_PREVIEW_CHARS],
                    )
                )

            if accepted:
                messages.append(
                    Message(
                        sender=sender,
                        intended_recipients=list(outbound.intended_recipients),
                        actual_recipients=accepted,
                        payload=outbound.payload,
                        cycle=cycle,
                    )
                )
            elif not outbound.intended_recipients:
                rejections.append(
                    Rejection(
                        sender=sender,
                        recipient="",
                        reason="no_recipients_requested",
                        cycle=cycle,
                        payload_preview=outbound.payload[:_PREVIEW_CHARS],
                    )
                )
        print(messages)
        return RouteResult(messages=messages, rejections=rejections)

    def deliver(self, result: RouteResult) -> dict[str, list[Message]]:
        """Convert routed messages into the mailbox mapping for the next cycle.

        The router is the sole writer of this structure. It returns a *fresh* dict each call
        rather than mutating the previous one, which is what makes "inboxes are refilled, not
        accumulated" a property of the design instead of a rule that has to be remembered.
        """
        mailboxes: dict[str, list[Message]] = {}
        for message in result.messages:
            for recipient in message.actual_recipients:
                mailboxes.setdefault(recipient, []).append(message)
        return mailboxes
