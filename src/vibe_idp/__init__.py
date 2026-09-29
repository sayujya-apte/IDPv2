"""Round-based, topology-constrained message routing for a network of agents.

Implements the design in `Centralized Synchronous Message Routing For Agents in a Graph.pdf`
using LangGraph's superstep model as the round boundary.
"""

from __future__ import annotations

from .agent import Agent, PurposeAgent, build_agents, retryable_model_errors
from .graph import GraphBuildError, build_network_graph, recursion_limit_for
from .router import CentralRouter, RouterError
from .runner import run_network, summarize_log
from .topology import Topology, TopologyError
from .types import (
    AgentDecision,
    Message,
    OutboundMessage,
    Rejection,
    RouteResult,
)

__all__ = [
    "Agent",
    "AgentDecision",
    "CentralRouter",
    "GraphBuildError",
    "Message",
    "OutboundMessage",
    "PurposeAgent",
    "Rejection",
    "RouteResult",
    "RouterError",
    "Topology",
    "TopologyError",
    "build_agents",
    "build_network_graph",
    "recursion_limit_for",
    "retryable_model_errors",
    "run_network",
    "summarize_log",
]
