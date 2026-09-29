"""Agent behaviour: retry policy, backoff, and the no-raise contract.

No Groq call is made. The model runnable is replaced with a local stub.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.exceptions import (
    ModelAuthenticationError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
)
from langchain_core.runnables import RunnableLambda

from vibe_idp import Message, PurposeAgent
from vibe_idp.agent import BACKOFF_PARAMS, retryable_model_errors
from vibe_idp.types import AgentDecision, OutboundMessage


def make_agent(**kwargs) -> PurposeAgent:
    return PurposeAgent("node_0", "summarise incoming findings", api_key="test-key", **kwargs)


def stub(agent: PurposeAgent, fn, **retry_kwargs):
    """Swap in a local stub, optionally wrapped in a retry policy."""
    runnable = RunnableLambda(fn)
    if retry_kwargs:
        runnable = runnable.with_retry(**retry_kwargs)
    agent._decider = runnable
    return agent


# ------------------------------------------------------------------ retry policy


def test_retry_policy_matches_configuration():
    agent = make_agent(max_attempts=3)
    decider = agent._decider

    assert decider.max_attempt_number == 3
    assert decider.wait_exponential_jitter is True
    assert decider.exponential_jitter_params == BACKOFF_PARAMS
    assert set(decider.retry_exception_types) == set(retryable_model_errors())


def test_backoff_params_are_exponential_and_capped():
    assert BACKOFF_PARAMS["exp_base"] == 2.0
    assert BACKOFF_PARAMS["max"] >= BACKOFF_PARAMS["initial"]
    assert BACKOFF_PARAMS["jitter"] > 0, "jitter avoids agents retrying in lockstep"


def test_transport_layer_does_not_retry_on_its_own():
    """ChatGroq must not add a second, hidden retry policy underneath with_retry."""
    assert make_agent()._client.max_retries == 0


def test_only_transient_errors_are_retryable():
    retryable = set(retryable_model_errors())

    for transient in (ModelRateLimitError,):
        assert transient in retryable
    for permanent in (
        ModelAuthenticationError,
        ModelPermissionDeniedError,
        ModelNotFoundError,
        ModelInvalidRequestError,
    ):
        assert permanent not in retryable, f"{permanent.__name__} would never succeed on retry"


def test_retryable_set_is_non_empty():
    assert len(retryable_model_errors()) >= 4


# -------------------------------------------------------------- no-raise contract


def test_step_returns_empty_when_the_model_keeps_failing(caplog):
    """A raising node would void the whole superstep, so step must swallow."""
    calls: list[int] = []

    def boom(_):
        calls.append(1)
        raise ModelRateLimitError("429 rate limited")

    agent = stub(
        make_agent(),
        boom,
        stop_after_attempt=3,
        retry_if_exception_type=retryable_model_errors(),
        wait_exponential_jitter=False,
    )

    result = asyncio.run(agent.step(1, []))

    assert result == []
    assert len(calls) == 3, "a retryable error should have been retried"
    assert "contributing nothing" in caplog.text


def test_non_retryable_error_is_attempted_once():
    calls: list[int] = []

    def boom(_):
        calls.append(1)
        raise ModelAuthenticationError("bad key")

    agent = stub(
        make_agent(),
        boom,
        stop_after_attempt=4,
        retry_if_exception_type=retryable_model_errors(),
        wait_exponential_jitter=False,
    )

    assert asyncio.run(agent.step(1, [])) == []
    assert len(calls) == 1, "an auth failure must not be retried"


def test_step_never_raises_even_for_unexpected_errors():
    def weird(_):
        raise ZeroDivisionError("not a model error at all")

    agent = stub(
        make_agent(),
        weird,
        stop_after_attempt=2,
        retry_if_exception_type=retryable_model_errors(),
        wait_exponential_jitter=False,
    )

    assert asyncio.run(agent.step(1, [])) == []


def test_step_tolerates_a_none_decision():
    agent = stub(make_agent(), lambda _: None)
    assert asyncio.run(agent.step(1, [])) == []


# ------------------------------------------------------------------- generation


def test_decision_is_mapped_to_outbound_messages():
    agent = stub(
        make_agent(),
        lambda _: AgentDecision(
            messages=[
                OutboundMessage(intended_recipients=["b"], payload="finding one"),
                OutboundMessage(intended_recipients=[], payload="thinking out loud"),
            ]
        ),
    )

    result = asyncio.run(agent.step(3, []))

    assert len(result) == 1, "a message with no recipients should be dropped before routing"
    assert result[0].intended_recipients == ["b"]
    assert result[0].payload == "finding one"


def test_silence_is_a_valid_answer():
    agent = stub(make_agent(), lambda _: AgentDecision(messages=[]))
    assert asyncio.run(agent.step(1, [Message(sender="b", payload="hi")])) == []


def test_run_when_idle_false_skips_the_model_entirely():
    calls: list[int] = []
    agent = stub(make_agent(run_when_idle=False), lambda _: calls.append(1) or AgentDecision())

    assert asyncio.run(agent.step(1, [])) == []
    assert calls == [], "no model call should happen when the inbox is empty"

    # It still acts when it actually has mail.
    asyncio.run(agent.step(2, [Message(sender="b", payload="hi")]))
    assert calls == [1]


def test_run_when_idle_true_calls_the_model_even_with_empty_inbox():
    calls: list[int] = []
    agent = stub(make_agent(), lambda _: calls.append(1) or AgentDecision())

    asyncio.run(agent.step(1, []))
    assert calls == [1]


# ------------------------------------------------------------------- credentials


def test_missing_api_key_is_rejected_clearly(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="no API key"):
        PurposeAgent("node_0", "purpose")


def test_api_key_falls_back_to_environment(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "from-env")
    agent = PurposeAgent("node_0", "purpose")
    assert agent._client.groq_api_key.get_secret_value() == "from-env"


# --------------------------------------------------------------------- prompting


def test_prompt_states_purpose_permitted_peers_and_inbox():
    agent = make_agent(permitted_peers=["b", "c"])
    inbox = [
        Message(
            sender="b",
            intended_recipients=["node_0"],
            actual_recipients=["node_0"],
            payload="a finding",
        )
    ]

    text = "\n".join(m.content for m in agent._build_messages(2, inbox))

    assert "summarise incoming findings" in text
    assert "b, c" in text
    assert "a finding" in text
    assert "Communication cycle 2" in text


def test_prompt_makes_isolation_explicit_when_no_peers():
    agent = make_agent()
    text = "\n".join(m.content for m in agent._build_messages(1, []))
    assert "no permitted peers" in text
