# DOC.md — build log

Implementation of the design in `Centralized Synchronous Message Routing For Agents in a
Graph.pdf`, using LangGraph for the graph machinery and Groq (via LangChain) for agent
reasoning.

This file records what was built, the order it was built in, the decisions made along the
way, and — most usefully — the things that turned out to be traps.

---

## 1. The central design decision

**A LangGraph superstep boundary *is* a communication cycle boundary.**

From the LangGraph Pregel runtime documentation:

> **Execution:** Execute all selected actors in parallel, until all complete, or one fails, or
> a timeout is reached. *During this phase, channel updates are invisible to actors until the
> next step.*
> **Update:** Update the channels with the values written by the actors in this step.

That is a Bulk Synchronous Parallel barrier, which is structurally the same thing as the
spec's communication cycle. The spec's central rule —

> if a message is generated during communication cycle `n`, the router validates and forwards
> it within the same cycle, but it only becomes available at the start of cycle `n+1`

— is therefore enforced by the **runtime**, not by application code. Nothing in
`graph.py` has to implement round ordering because the executor already provides it.

This was the finding that made the rest of the design simple, and it was verified
experimentally before any production code was written (step 3 below).

### The decision that follows from it: the topology is *not* encoded as graph edges

This is the mistake I expect a future contributor to make, so it is stated explicitly in
`graph.py` and pinned by a test.

LangGraph edges mean **"run this node next"**. An adjacency matrix means **"this node is
permitted to send there"**. These are different things. If `A[0][2] == 0` were expressed as
an edge `node_0 -> node_2`, then `node_2` would be *scheduled to run* on every round in which
`node_0` ran — regardless of whether `node_0` actually sent anything. An edge means
"delivery happened", which is exactly wrong.

The spec is also explicit that the router, not the scheduler, decides delivery. So:

- The topology lives **only** in `router.py`, as a validation step.
- LangGraph edges express **only** the round cycle.

Verified edge list for a 3-node ring topology — note that no agent-to-agent edge exists:

```
__start__ -> router
node_0    -> router      node_1 -> router      node_2 -> router
router    -> node_0      router -> node_1      router -> node_2   (all conditional)
router    -> __end__                                                          (conditional)
```

`tests/test_graph_wiring.py::test_topology_is_not_encoded_as_graph_edges` asserts this.

---

## 2. Two ambiguities in the spec, resolved with the user

The spec does not settle these, and both change the code substantially:

| Question | Decision |
|---|---|
| What does `A[i][j] = 1` mean? | **Undirected.** The matrix is symmetrised: `permitted(s) = {j : A[s][j] or A[j][s] truthy}`. Asymmetry in the caller's input is recorded in `Topology.asymmetric_pairs` rather than silently discarded. |
| When does the network stop? | **`max_rounds` only.** No agent-completion signal, no consensus. |

One consequence of the second: on the terminating router pass, messages produced in the final
cycle *are* validated and logged, but are never read, because the run ends. They appear in
`log` with their production `cycle`. `examples/demo.py` labels these explicitly.

---

## 3. Build order, and what was verified at each step

The environment was bare — Python 3.13.5, nothing installed, no commits, no lockfile. So
every design claim below was checked against the real installed library before being built on.

**Step 1 — scaffold and install.** `pyproject.toml` (hatchling, src layout), `.env.example`,
`.gitignore`, `README.md`. Installed into a venv: langgraph 1.2.12, langchain-core 1.6.5,
langchain-groq 1.1.3, groq 0.37.1, pydantic 2.13.5, pytest 9.1.1, ruff 0.16.9.
*First attempt failed* — `readme = "README.md"` pointed at a file that did not exist yet, and
hatchling refused to build. Fixed by writing the README.

**Step 2 — verify the API surface I intended to use.** Confirmed `StateGraph`, `Command`
(fields: `graph`, `update`, `resume`, `goto`), `START`/`END`, and that a dynamically
constructed `TypedDict` works as a state schema. The schema is dynamic because the node count
is *data* — the adjacency matrix decides how many agents exist.

**Step 3 — prototype and falsify the design.** This was the most valuable step. I wrote a
throwaway prototype of the round cycle and ran it. Four things came out of it:

1. **Round semantics work.** A 2-node network where `a` sends to `b` in round 1:
   ```
   a: round=1 saw=[]      b: round=1 saw=['a']
   a: round=2 saw=['b']   b: round=2 saw=[]
   ```
   `b`'s reply is generated in round 1 and is invisible until round 2. The spec's guarantee
   holds.
2. **`Command(goto=...)` does *not* suppress static edges.** My first plan had the router
   return `Command(goto=END)` to terminate, alongside static edges to the agents. It looped
   forever: the static edge still fired after `goto=END`. **Fix: use `add_conditional_edges`
   from the router, with no static outgoing edge at all.**
3. **Supersteps are transactional.** I confirmed that if one node raises, *every* sibling's
   writes in that superstep are discarded — `ok` wrote successfully and the write vanished
   when `bad` raised. This is why `PurposeAgent.step` must never propagate an exception (§5).
4. **Two nodes writing the same `LastValue` channel in one superstep raises
   `InvalidUpdateError`.** This is what forced the one-writer-per-channel design (§4).

**Step 4 — write the modules.** `types.py` → `topology.py` → `router.py` → `agent.py` →
`graph.py` → `runner.py`. `router.py` deliberately imports no LangGraph, following the spec's
requirement that the communication layer stay independent of node application logic. That also
means the rules with real logic are testable with no event loop and no API key.

**Step 5 — tests.** 70 tests. `ScriptedAgent` in `tests/conftest.py` satisfies the `Agent`
protocol, so the whole network runs deterministically with no network access.

**Step 6 — lint, format, demo.** `ruff check` clean, `ruff format` applied, 70/70 passing.
`examples/demo.py` runs the full stack end to end.

---

## 4. Channel ownership: one writer per channel

Every state channel has exactly **one** writer and uses full-overwrite semantics:

| Channel | Writer | Semantics |
|---|---|---|
| `outbox_<i>` | *only* `agent_<i>` | full overwrite |
| `mailboxes` | *only* `router` | full overwrite (a fresh dict each round) |
| `cycle` | *only* `router` | int |
| `log` | *only* `router` | full overwrite of `state["log"] + new` |

This is **not** stylistic. It is load-bearing, for two reasons discovered in step 3:

- `InvalidUpdateError` is raised when two nodes write the same `LastValue` channel in one
  superstep. Per-node `outbox_<i>` channels make that structurally impossible.
- The obvious alternative — a single shared `outbox: Annotated[list[Message], operator.add]`
  — cannot be *cleared* by the router. A reducing channel only appends, so the router has no
  way to reset it between rounds. Full-overwrite single-writer channels avoid needing a
  reducer or a sentinel "clear" value at all.

Because `mailboxes` is rewritten fresh each round, the spec's "inboxes are refilled, not
accumulated" behaviour is structural rather than a rule someone has to remember.

**Cost of a raising node.** Since supersteps are transactional, one agent whose Groq call gave
up would discard *every other agent's* output for that round. Hence `PurposeAgent.step` catches
everything and returns `[]` (§5).

---

## 5. Agent, retry, and backoff

**The retryable set is derived, not hardcoded.** `langchain-core` 1.6.5 exposes an
`is_retryable` class flag:

```
RETRYABLE    : ModelAPIError (5xx), ModelConnectionError, ModelRateLimitError (429), ModelTimeoutError
NOT retryable: ModelAuthenticationError, ModelPermissionDeniedError,
               ModelNotFoundError, ModelInvalidRequestError, ContextOverflowError
```

`retryable_model_errors()` reads that flag, so the policy stays correct across versions and
cannot accidentally come to include auth failures. `langchain-core`'s own docstring agrees:
*"Good exceptions to retry are all server errors (5xx) and selected client errors (4xx) such
as 429 Too Many Requests."*

**Retry ownership is single.** `ChatGroq` is constructed with `max_retries=0` so the LangChain
`with_retry` is the only retry policy. Both defaulting to retry would give 2 × 4 = 8 requests
per agent per round, with a backoff curve that is a product of two independent policies.

**Backoff:** `wait_exponential_jitter` with `initial=1.0, exp_base=2.0, max=20.0, jitter=1.0`.
Jitter matters here specifically: all agents in a round are rejected together by a rate limit,
and without jitter they would retry in lockstep and collide again.

**`step()` never raises.** Verified behaviourally in `test_agent_policy.py`: a rate-limit error
is retried the configured number of times and then yields `[]`; an auth error is attempted
*once*; even a `ZeroDivisionError` is contained.

**Structured output:** `AgentDecision` → `OutboundMessage` → `{intended_recipients, payload}`,
via `with_structured_output` (tool-calling, which is the broadly supported path on Groq;
`method="json_schema"` works only on a model subset). An agent can express *who it wants to
reach* but has no way to express a *delivery decision* — that belongs to the router.

The `permitted_peers` list is injected into the prompt to reduce wasted requests. It is **not**
an access control; the router enforces the topology regardless of what any model is told.

---

## 6. Traps found during the build

Things that would each have caused a confusing failure:

1. **`Command(goto=END)` does not suppress static edges** → infinite loop. Use conditional
   edges with no static outgoing edge from the router.
2. **`RateLimitError` does not exist** in langchain-core 1.x. It is `ModelRateLimitError`.
   Importing the old name would have been an `ImportError` at module import.
3. **Supersteps are transactional** — one raising node voids the whole round. Agents must
   never propagate.
4. **`InvalidUpdateError` on concurrent writes** to a `LastValue` channel. Motivates the
   one-writer-per-channel design.
5. **A reducing channel cannot be cleared.** Motivates full-overwrite over `operator.add`.
6. **`recursion_limit` defaults to 25 and counts *supersteps*, not nodes.** A round costs two
   supersteps, so 25 would have died at ~12 rounds. Termination is the conditional edge;
   `recursion_limit_for()` sets a generous backstop so that a genuine bug raises a clear
   `GraphRecursionError` instead of silently truncating.
7. **Pydantic v2 forbids positional constructor args.** `OutboundMessage(["a"], "x")` is a
   `TypeError`; must be `OutboundMessage(intended_recipients=[...], payload=...)`.
8. **Pydantic class docstrings are serialized into the tool definition on every model call.**
   The first draft put multi-paragraph developer rationale in `AgentDecision`/`OutboundMessage`
   docstrings, which was shipped to Groq on every request. Trimmed to concise model-facing
   text, with the rationale moved to comments and to this file.
9. **`get_state_history` is newest-first**, and the oldest checkpoint's `next` is
   `('__start__',)` rather than the first node.
10. **Checkpoint serde warns on custom types.** LangGraph logs
    `Deserializing unregistered type vibe_idp.types.Message ... will be blocked in a future
    version`. Harmless today; see §8.

---

## 7. What is tested

70 tests, no network access, no API key.

| File | Tests | Covers |
|---|---|---|
| `test_round_semantics.py` | 8 | **The spec's core guarantee.** n→n+1 visibility, replies not arriving same-round, one hop per cycle along a chain, inboxes refilled not accumulated, `max_rounds` activation count, unpermitted recipients dropped, and that a raising agent really does break the round. |
| `test_router.py` | 14 | The intersection rule, multicast, hallucinated recipients, payload-independence, dedup, fresh-mailbox semantics. |
| `test_topology.py` | 13 | Squareness, symmetrisation, diagonal handling, duplicate ids, asymmetry detection. |
| `test_agent_policy.py` | 17 | Retry/backoff configuration, the derived retryable set, the no-raise contract, credential handling, prompt contents. |
| `test_graph_wiring.py` | 18 | Dynamic schema, build-time validation, **the no-topology-edges assertion**, checkpointer superstep counts, log summary. |

The two tests that would catch the most damaging regressions are
`test_message_is_invisible_in_its_production_cycle` and
`test_one_hop_per_cycle_along_a_chain`. A plausible refactor — switching to `Send`-based
fan-out, or moving routing logic inside the agent nodes — breaks both.

---

## 8. Known limitations and suggested next steps

1. **No live Groq run has been performed.** The full stack is exercised end to end via
   `ScriptedAgent` and `examples/demo.py`, and the `PurposeAgent` code path is unit-tested with
   a stubbed runnable, but no real API call has been made. `examples/demo.py::live_demo()` is
   the entry point once `GROQ_API_KEY` is set. Structured output tool-calling behaviour on
   `llama-3.3-70b-versatile` in particular is unverified.
2. **Rate limits vs. fan-out.** All agents in a round run in one superstep, i.e. `N`
   concurrent Groq calls, for `max_rounds` rounds — `N × max_rounds` requests. This will hit
   free-tier limits on a wide network. `run_when_idle=False` on `PurposeAgent` skips the model
   call for agents with empty inboxes, but changes semantics (an idle agent can no longer
   originate a message) and defaults to `True` for spec fidelity.
3. **Checkpoint serde forward-compat.** Custom Pydantic types in state will be rejected in a
   future LangGraph. When that lands, either add `vibe_idp.types` to
   `allowed_msgpack_modules` or store plain dicts in channels at the graph boundary.
4. **Node ids must be valid identifiers** (`^[A-Za-z_][A-Za-z0-9_]*$`) because they are used as
   LangGraph node names. Enforced at build time with a clear error. Node ids *with spaces or
   dashes* — e.g. `fact-checker` is fine, `fact checker` is not — will be rejected.
5. **The `log` grows without bound** across cycles. For long runs, a trimming policy or a
   `DeltaChannel` would be needed; the checkpointer snapshots the full state each superstep.
6. **Concurrency within a superstep is untested against real rate limits** — only the
   round-ordering logic is covered by tests.

---

## 9. Running it

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"

.venv/bin/python -m pytest -q          # 70 tests, no API key needed
.venv/bin/python -m ruff check .
.venv/bin/python examples/demo.py      # end-to-end trace, no API key needed

# for the live Groq path:
cp .env.example .env                   # then set GROQ_API_KEY
# .venv/bin/python -c "import asyncio; from examples.demo import live_demo; asyncio.run(live_demo())"
```

## 10. Layout

```
src/vibe_idp/
  types.py     Message, OutboundMessage, Rejection, AgentDecision, RouteResult
  topology.py  adjacency validation, symmetrisation, permitted_recipients
  router.py    CentralRouter — pure routing logic, no LangGraph imports
  agent.py     PurposeAgent (Groq + LangChain retry/backoff), Agent protocol
  graph.py     build_network_graph — dynamic schema, round cycle, termination
  runner.py    run_network, summarize_log
tests/         70 tests, conftest.py provides ScriptedAgent / EchoAgent / RelayAgent
examples/demo.py
```

Nothing has been committed — the repo had no commits when this build started, so the first
commit would contain the entire implementation.
