# vibe_idp

Round-based, topology-constrained message routing for a network of purpose-specific
agents, built on [LangGraph](https://github.com/langchain-ai/langgraph).

Implements the design in `Centralized Synchronous Message Routing For Agents in a Graph.pdf`:
agents generate messages, a centralized router validates and delivers them into
per-agent inboxes, and delivery advances one communication cycle at a time.

## The round model

A LangGraph **superstep boundary is a communication cycle boundary.** LangGraph's Pregel
runtime runs a superstep transactionally — writes made during a superstep are invisible
until the next one. That gives the spec's core guarantee for free: a message produced in
cycle `n` cannot be read or acted on until cycle `n+1` begins. No agent can trigger a
response chain inside a single round.

```
START -> router -> agent_0  ┐
          ^       agent_1  ├─ one superstep, all agents in parallel
          |       ...      │
          |       agent_N  ┘
          +------------------  router returns Command(goto=END) at max_rounds
```

The adjacency matrix is **not** encoded as LangGraph edges. Edges express control flow; the
topology is a validation constraint applied by the router. LangGraph edges express only the
round cycle.

## Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
cp .env.example .env   # then set GROQ_API_KEY
```

## Run tests

```bash
.venv/bin/python -m pytest -q
```

The test suite never calls Groq. Agents implement the `Agent` protocol, and tests inject a
`ScriptedAgent` with canned decisions, so routing and round semantics are tested
deterministically and for free.

## Layout

| Module | Responsibility |
|---|---|
| `types.py` | `Message`, `OutboundMessage`, `Delivery`, `AgentDecision` |
| `topology.py` | Adjacency matrix validation, symmetrisation, permitted-recipient lookup |
| `router.py` | Pure routing logic. No LangGraph imports, no network. |
| `agent.py` | `PurposeAgent` (Groq + LangChain retry/backoff) and the `Agent` protocol |
| `graph.py` | Assembles the `StateGraph` from a topology and a set of agents |
| `runner.py` | `run_network` — drives the graph for `max_rounds` |

`router.py` is deliberately free of LangGraph imports, following the spec's requirement that
the communication layer stay independent of node application logic.

See `DOC.md` for the build log and design decisions.
