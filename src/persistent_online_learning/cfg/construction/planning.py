"""Choose a feasible nonterminal-count plan before topology construction.

Unold et al. treat ``S_NT`` as a maximum and decide whether to create a new
nonterminal while rules are inserted. This project deliberately resolves that
uncertainty earlier: it enumerates all feasible total nonterminal counts and the
compatible size range of the initial terminal-only foundation, then randomly
selects one plan.

Preplanning is a project-specific adaptation. It exists to make the later topology
phase reason directly about a fixed number of future node creations and therefore
reserve the correct number of RHS edges for root ancestry. The paper's symbol and
rule capacities remain the mathematical basis of the feasible range.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt

import torch

from ..config import (
    CFGSpawnConfig,
    _connection_edge_count,
    _minimum_plain_nonterminals,
)


@dataclass(frozen=True, slots=True)
class ConstructionPlan:
    """Resolve the node-count uncertainty needed by productive topology building.

    Topology construction needs to know how many future rules must create a new
    LHS, because each such rule consumes one RHS edge to contain the previous root.
    This plan records that chosen total and the allowable size of the initial
    terminal-only foundation. It is transient construction state and never appears
    in the published ``CFG``.

    Attributes:
        target_nonterminals: Exact total number of nonterminal nodes this build
            will create, selected within ``GrammarConfig.max_nonterminals``.
        initial_nonterminal_min: Smallest number of productive foundation nodes
            that can host the requested terminal-pair rules while leaving enough
            later rules to reach ``target_nonterminals``.
        initial_nonterminal_max: Largest foundation size that can still be joined
            into one reachable graph by the remaining nonterminal RHS edges.
    """

    target_nonterminals: int
    initial_nonterminal_min: int
    initial_nonterminal_max: int


def feasible_construction_plans(config: CFGSpawnConfig) -> tuple[ConstructionPlan, ...]:
    """Enumerate every node-count plan compatible with one validated request.

    Request-level capacity checks already ran in ``CFGSpawnConfig``. This function
    performs the construction-specific reduction from those maxima to concrete
    total/foundation count combinations. A missing plan means the request passes
    coarse symbol capacities but cannot satisfy this project's explicit productive
    foundation and connection lifecycle.
    """

    grammar = config.grammar
    terminal_count = config.terminal_vocabularies.terminal_count
    terminal_square = terminal_count**2

    minimum_nonterminals = max(
        1,
        _minimum_plain_nonterminals(config),
        _ceil_sqrt_ratio(grammar.parenthesis_rules, terminal_square),
        _ceil_sqrt_ratio(grammar.iteration_rules, 2 * terminal_count),
        _ceil_cuberoot(grammar.branch_rules),
    )
    remaining_rules = (
        grammar.parenthesis_rules + grammar.iteration_rules + grammar.branch_rules
    )
    total_rules = grammar.terminal_pair_rules + remaining_rules
    maximum_nonterminals = min(
        grammar.max_nonterminals,
        _connection_edge_count(grammar) + 1,
        total_rules,
    )

    plans: list[ConstructionPlan] = []
    for target_nonterminals in range(
        minimum_nonterminals, maximum_nonterminals + 1
    ):
        initial_min = max(
            _minimum_plain_nonterminals(config),
            target_nonterminals - remaining_rules,
        )
        initial_max = min(
            grammar.terminal_pair_rules,
            target_nonterminals,
            _connection_edge_count(grammar) + 1,
        )
        if initial_min <= initial_max:
            plans.append(
                ConstructionPlan(
                    target_nonterminals=target_nonterminals,
                    initial_nonterminal_min=initial_min,
                    initial_nonterminal_max=initial_max,
                )
            )

    if not plans:
        raise ValueError(
            "rule counts cannot create and connect a feasible nonterminal graph"
        )
    return tuple(plans)


def choose_construction_plan(
    config: CFGSpawnConfig,
    generator: torch.Generator,
) -> ConstructionPlan:
    """Randomly select one feasible plan without biasing toward minimum graph size.

    This is the boundary where a request containing a nonterminal maximum becomes
    one concrete construction lifecycle. Keeping selection here lets topology
    consume a fixed contract rather than repeatedly asking whether future rule
    choices might still recover the requested graph.
    """

    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be torch.Generator")
    plans = feasible_construction_plans(config)
    index = int(torch.randint(len(plans), (), generator=generator))
    return plans[index]


def _ceil_div(numerator: int, denominator: int) -> int:
    """Return an exact integer ceiling for capacity-bound calculations."""

    return (numerator + denominator - 1) // denominator


def _ceil_sqrt_ratio(count: int, multiplier: int) -> int:
    """Find the least N satisfying ``count <= multiplier * N**2`` exactly."""

    if count == 0:
        return 0
    quotient = _ceil_div(count, multiplier)
    root = isqrt(quotient)
    return root if root * root == quotient else root + 1


def _ceil_cuberoot(count: int) -> int:
    """Find the least N satisfying ``count <= N**3`` without float rounding."""

    if count == 0:
        return 0
    low, high = 0, 1
    while high**3 < count:
        high *= 2
    while low + 1 < high:
        middle = (low + high) // 2
        if middle**3 >= count:
            high = middle
        else:
            low = middle
    return high
