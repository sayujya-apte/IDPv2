"""A purpose-specific agent backed by the Groq API.

An agent's entire responsibility is message *generation*: it reads its own inbox, decides
what it wants to say, and names who it intends to say it to. It has no ability to deliver
anything -- that belongs to the router.

Reliability notes that materially affect the network as a whole:

* **A LangGraph superstep is transactional.** If any node raises, none of the writes from
  that superstep are applied. One agent whose Groq call gives up would therefore discard
  every other agent's successful output for that round. `PurposeAgent.step` consequently
  never propagates an exception; it degrades to sending nothing and logs why.
* **Retry ownership is explicit.** `ChatGroq` is constructed with `max_retries=0` so that the
  LangChain-level `with_retry` is the single retry policy. Leaving both enabled multiplies
  attempts (2 x 4 = 8 requests per agent per round) and makes the backoff curve a product of
  two independent policies.
* **Only transient failures are retried.** Authentication, permission, and invalid-request
  errors will never succeed on a second attempt, so retrying them just burns the round's time
  budget and rate-limit quota.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from langchain_core.exceptions import ModelError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from .types import AgentDecision, Message, OutboundMessage

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_TIMEOUT_SECONDS = 60.0

#: Backoff curve handed to LangChain's `with_retry` (tenacity `wait_exponential_jitter`).
#: Initial wait 1s, doubling, capped at 20s, plus up to 1s of random jitter so that a fan-out
#: of agents rejected together does not retry in lockstep.
BACKOFF_PARAMS: dict[str, float] = {"initial": 1.0, "max": 20.0, "exp_base": 2.0, "jitter": 1.0}


def retryable_model_errors() -> tuple[type[BaseException], ...]:
    """Model errors worth retrying, derived from langchain-core's own `is_retryable` flag.

    Deriving this instead of hardcoding a tuple keeps the policy correct across langchain-core
    versions, and guarantees authentication and invalid-request errors are excluded -- they
    carry no `is_retryable` flag.
    """
    import langchain_core.exceptions as exc

    return tuple(
        obj
        for obj in vars(exc).values()
        if isinstance(obj, type)
        and issubclass(obj, ModelError)
        and getattr(obj, "is_retryable", False)
    )


@runtime_checkable
class Agent(Protocol):
    """The contract the graph depends on.

    `graph.py` is written against this protocol, not against `PurposeAgent`, so tests can
    substitute a deterministic scripted agent and run the whole network with no API key.
    """

    node_id: str

    async def step(self, cycle: int, inbox: Sequence[Message]) -> list[OutboundMessage]:
        """Read this agent's inbox for `cycle` and return the messages it wants to send.

        Implementations must not raise. A LangGraph superstep is transactional, so an
        exception here discards every sibling agent's writes for the round.
        """
        ...


class PurposeAgent:
    """A single-purpose agent that generates messages using a Groq model."""

    def __init__(
        self,
        node_id: str,
        purpose: str,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        permitted_peers: Sequence[str] = (),
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        temperature: float = 0.0,
        run_when_idle: bool = True,
    ) -> None:
        self.node_id = node_id
        self.purpose = purpose
        self.permitted_peers = tuple(permitted_peers)
        self.max_attempts = max(1, max_attempts)
        self.run_when_idle = run_when_idle
    
        key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError(
                f"agent {node_id!r} has no API key: pass api_key= or set the GROQ_API_KEY env var"
            )

        # max_retries=0 hands retry policy to with_retry below; see module docstring.
        self._client = ChatGroq(
            model=model,
            api_key=key,
            temperature=temperature,
            max_retries=0,
            timeout=timeout,
        )
        self._decider = self._client.with_structured_output(AgentDecision).with_retry(
            stop_after_attempt=self.max_attempts,
            retry_if_exception_type=retryable_model_errors(),
            wait_exponential_jitter=True,
            exponential_jitter_params=BACKOFF_PARAMS,
        )

    # ------------------------------------------------------------------- prompting

    def _build_messages(
        self, cycle: int, inbox: Sequence[Message]
    ) -> list[SystemMessage | HumanMessage]:
        if self.permitted_peers:
            peers = ", ".join(self.permitted_peers)
        else:
            peers = "(none -- you have no permitted peers)"

        lines = [f"Communication cycle {cycle}."]
        if inbox:
            lines.append("Messages in your inbox:")
            for msg in inbox:
                lines.append(f"  - from {msg.sender}: {msg.payload}")
        else:
            lines.append("Your inbox is empty this cycle.")
        lines.append(f"You may address only these nodes: {peers}.")

        return [
            SystemMessage(
                content=(
                    f"You are the {self.node_id} agent in a network of cooperating agents. "
                    f"Your sole purpose: {self.purpose}\n\n"
                    "Decide what, if anything, to send this cycle. Address peers only from "
                    "your permitted list. Send nothing if you have nothing to add -- silence "
                    "is a valid and often correct choice. Keep each payload short and concrete."
                )
            ),
            HumanMessage(content="\n".join(lines)),
        ]

    # ----------------------------------------------------------------------- step

    async def step(self, cycle: int, inbox: Sequence[Message]) -> list[OutboundMessage]:
        """Generate this cycle's outbound messages. Never raises."""
        if not inbox and not self.run_when_idle:
            return []

        try:
            decision: AgentDecision = await self._decider.ainvoke(
                self._build_messages(cycle, inbox)
            )
        except Exception:
            logger.exception(
                "agent %s failed in cycle %d; contributing nothing this round", self.node_id, cycle
            )
            return []

        if decision is None:
            logger.warning("agent %s returned no decision in cycle %d", self.node_id, cycle)
            return []

        outbounds = [
            OutboundMessage(
                intended_recipients=list(m.intended_recipients),
                payload=m.payload,
            )
            for m in decision.messages
            if m.intended_recipients
        ]
        if not outbounds:
            logger.debug("agent %s chose to send nothing in cycle %d", self.node_id, cycle)
        return outbounds

    def __repr__(self) -> str:
        return f"PurposeAgent(node_id={self.node_id!r})"


def build_agents(
    purposes: dict[str, str],
    api_key: str | None = None,
    permitted: dict[str, Sequence[str]] | None = None,
    **kwargs: Any,
) -> list[PurposeAgent]:
    """Convenience factory: one agent per entry in `purposes`.

    `permitted` is only used to tell each model which peers it may address. It reduces wasted
    requests; it is *not* an access control. The router enforces the topology regardless of
    what any model is told.
    """
    return [
        PurposeAgent(
            node_id=node_id,
            purpose=purpose,
            api_key=api_key,
            permitted_peers=(permitted or {}).get(node_id, ()),
            **kwargs,
        )
        for node_id, purpose in purposes.items()
    ]
