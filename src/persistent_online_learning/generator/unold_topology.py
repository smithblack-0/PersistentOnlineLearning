"""Construct productive nonterminal topology for an Unold grammar.

Every inserted draft is productive immediately. Plain ``A -> a b`` drafts
create productive roots first. Later drafts use only productive RHS nodes,
connect hanging roots before the remaining edge budget is exhausted, and make
each newly created LHS the new start root by placing the prior root on its RHS.

This module owns topology only. Complete terminal nodes are supplied elsewhere,
and terminal positions are labeled after this phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterator, TypeVar

import torch

from .cfg_config import CFGSpawnConfig, GrammarConfig, _construction_plans
from .grammar import Nonterminal
from .unold_rules import (
    RuleDraft,
    RuleFamily,
    bucket_key,
    child_count,
    terminal_label_capacity,
)

_RANDOM_ATTEMPTS = 128
_Value = TypeVar("_Value")


@dataclass(frozen=True, slots=True)
class SyntaxTopology:
    """Transient phase product containing nodes and unlabeled production drafts."""

    start: Nonterminal
    nonterminals: tuple[Nonterminal, ...]
    drafts: tuple[RuleDraft, ...]


@dataclass(slots=True)
class _SyntaxState:
    """Minimal evolving state required by productive-first topology construction."""

    nonterminals: list[Nonterminal]
    reachable: list[Nonterminal]
    hanging: list[Nonterminal]
    root: Nonterminal
    drafts: list[RuleDraft]
    bucket_counts: dict[tuple[object, ...], int]
    target_nonterminals: int
    generator: torch.Generator


@dataclass(frozen=True, slots=True)
class _TopologyCandidate:
    """One legal next topology change before commitment to construction state."""

    draft: RuleDraft
    creates_lhs: bool


def build_unold_topology(
    config: CFGSpawnConfig,
    terminal_count: int,
    generator: torch.Generator,
) -> SyntaxTopology:
    """Build the productive topology selected by one feasible count plan."""

    state = _create_plain_foundation(config, terminal_count, generator)
    _extend_productive_graph(state, config.grammar, terminal_count)
    return SyntaxTopology(
        start=state.root,
        nonterminals=tuple(state.nonterminals),
        drafts=tuple(state.drafts),
    )


def _create_plain_foundation(
    config: CFGSpawnConfig,
    terminal_count: int,
    generator: torch.Generator,
) -> _SyntaxState:
    """Create productive A -> a b roots and identify the initial hanging set."""

    plans = _construction_plans(config)
    plan = plans[_random_index(len(plans), generator)]
    initial_count = _random_inclusive(
        plan.initial_nonterminal_min,
        plan.initial_nonterminal_max,
        generator,
    )
    nonterminals = [Nonterminal(f"N{index}") for index in range(initial_count)]
    root = nonterminals[-1]
    drafts: list[RuleDraft] = []
    bucket_counts: dict[tuple[object, ...], int] = {}

    rules_per_lhs = [1] * initial_count
    unassigned = config.grammar.terminal_pair_rules - initial_count
    capacity = terminal_count**2
    while unassigned:
        available = [
            index for index, count in enumerate(rules_per_lhs) if count < capacity
        ]
        lhs_index = available[_random_index(len(available), generator)]
        rules_per_lhs[lhs_index] += 1
        unassigned -= 1

    lhs_order = torch.randperm(initial_count, generator=generator).tolist()
    for lhs_index in lhs_order:
        lhs = nonterminals[lhs_index]
        key = (RuleFamily.PLAIN_PARENTHESIS, lhs)
        count = rules_per_lhs[lhs_index]
        bucket_counts[key] = count
        drafts.extend(
            RuleDraft(RuleFamily.PLAIN_PARENTHESIS, lhs) for _ in range(count)
        )

    return _SyntaxState(
        nonterminals=nonterminals,
        reachable=[root],
        hanging=nonterminals[:-1],
        root=root,
        drafts=drafts,
        bucket_counts=bucket_counts,
        target_nonterminals=plan.target_nonterminals,
        generator=generator,
    )


def _extend_productive_graph(
    state: _SyntaxState,
    config: GrammarConfig,
    terminal_count: int,
) -> None:
    """Add remaining topologies while preserving productivity and edge budgets."""

    remaining = {
        RuleFamily.PARENTHESIS: config.parenthesis_rules,
        RuleFamily.ITERATION: config.iteration_rules,
        RuleFamily.BRANCH: config.branch_rules,
    }
    while any(remaining.values()):
        rules_left = sum(remaining.values())
        new_nodes_left = state.target_nonterminals - len(state.nonterminals)
        if new_nodes_left < 0 or new_nodes_left > rules_left:
            raise RuntimeError("construction lost its planned nonterminal count")

        create_modes = (
            (True,)
            if new_nodes_left == rules_left
            else (False,)
            if new_nodes_left == 0
            else (False, True)
        )
        choices: list[tuple[RuleFamily, _TopologyCandidate]] = []
        for family, count in remaining.items():
            if count == 0:
                continue
            remaining_after = dict(remaining)
            remaining_after[family] -= 1
            for creates_lhs in create_modes:
                future_new_nodes = new_nodes_left - int(creates_lhs)
                future_hanging_slots = (
                    _connection_slots(remaining_after) - future_new_nodes
                )
                if future_hanging_slots < 0:
                    continue
                required_hanging = max(
                    0, len(state.hanging) - future_hanging_slots
                )
                max_hanging = child_count(family) - int(creates_lhs)
                if required_hanging > max_hanging:
                    continue
                candidate = _find_topology_candidate(
                    state=state,
                    family=family,
                    creates_lhs=creates_lhs,
                    required_hanging=required_hanging,
                    terminal_count=terminal_count,
                )
                if candidate is not None:
                    choices.append((family, candidate))

        if not choices:
            raise RuntimeError(
                "no topology change can preserve the remaining construction invariants"
            )
        family, candidate = choices[_random_index(len(choices), state.generator)]
        _commit_topology(state, candidate, terminal_count)
        remaining[family] -= 1

    if state.hanging:
        raise RuntimeError("construction ended with unreachable productive roots")
    if len(state.nonterminals) != state.target_nonterminals:
        raise RuntimeError("construction ended before reaching its planned node count")


def _find_topology_candidate(
    *,
    state: _SyntaxState,
    family: RuleFamily,
    creates_lhs: bool,
    required_hanging: int,
    terminal_count: int,
) -> _TopologyCandidate | None:
    """Choose one legal topology, using exhaustive scan only after random misses."""

    for _ in range(_RANDOM_ATTEMPTS):
        candidate = _sample_topology_candidate(
            state=state,
            family=family,
            creates_lhs=creates_lhs,
            required_hanging=required_hanging,
        )
        if _topology_has_capacity(state, candidate.draft, terminal_count):
            return candidate

    for candidate in _iter_topology_candidates(
        state=state,
        family=family,
        creates_lhs=creates_lhs,
        required_hanging=required_hanging,
    ):
        if _topology_has_capacity(state, candidate.draft, terminal_count):
            return candidate
    return None


def _sample_topology_candidate(
    *,
    state: _SyntaxState,
    family: RuleFamily,
    creates_lhs: bool,
    required_hanging: int,
) -> _TopologyCandidate:
    """Sample one topology with required hanging roots and root ancestry."""

    number_of_children = child_count(family)
    lhs = (
        Nonterminal(f"N{len(state.nonterminals)}")
        if creates_lhs
        else state.reachable[_random_index(len(state.reachable), state.generator)]
    )

    children: list[Nonterminal] = [state.root] if creates_lhs else []
    selected_hanging = _sample_without_replacement(
        state.hanging, required_hanging, state.generator
    )
    children.extend(selected_hanging)
    while len(children) < number_of_children:
        children.append(
            state.nonterminals[
                _random_index(len(state.nonterminals), state.generator)
            ]
        )
    children = _shuffle(children, state.generator)

    return _TopologyCandidate(
        draft=RuleDraft(
            family=family,
            lhs=lhs,
            children=tuple(children),
            terminal_first=(
                bool(_random_index(2, state.generator))
                if family is RuleFamily.ITERATION
                else True
            ),
        ),
        creates_lhs=creates_lhs,
    )


def _iter_topology_candidates(
    *,
    state: _SyntaxState,
    family: RuleFamily,
    creates_lhs: bool,
    required_hanging: int,
) -> Iterator[_TopologyCandidate]:
    """Enumerate legal topology shapes only when random sampling finds none."""

    number_of_children = child_count(family)
    lhs_options = (
        (Nonterminal(f"N{len(state.nonterminals)}"),)
        if creates_lhs
        else tuple(state.reachable)
    )
    for lhs in lhs_options:
        for children in product(state.nonterminals, repeat=number_of_children):
            if creates_lhs and state.root not in children:
                continue
            hanging_used = len({child for child in children if child in state.hanging})
            if hanging_used < required_hanging:
                continue
            sides = (False, True) if family is RuleFamily.ITERATION else (True,)
            for terminal_first in sides:
                yield _TopologyCandidate(
                    draft=RuleDraft(
                        family=family,
                        lhs=lhs,
                        children=tuple(children),
                        terminal_first=terminal_first,
                    ),
                    creates_lhs=creates_lhs,
                )


def _topology_has_capacity(
    state: _SyntaxState,
    draft: RuleDraft,
    terminal_count: int,
) -> bool:
    """Check whether terminal labeling can still make this draft unique."""

    used = state.bucket_counts.get(bucket_key(draft), 0)
    return used < terminal_label_capacity(draft, terminal_count)


def _commit_topology(
    state: _SyntaxState,
    candidate: _TopologyCandidate,
    terminal_count: int,
) -> None:
    """Apply one already-legal topology change to the evolving state."""

    draft = candidate.draft
    if not _topology_has_capacity(state, draft, terminal_count):
        raise RuntimeError("attempted to commit an exhausted rule topology")

    if candidate.creates_lhs:
        state.nonterminals.append(draft.lhs)
        state.reachable.append(draft.lhs)
        state.root = draft.lhs

    connected = [child for child in draft.children if child in state.hanging]
    if connected:
        connected_set = set(connected)
        state.hanging = [
            node for node in state.hanging if node not in connected_set
        ]
        for node in connected:
            if node not in state.reachable:
                state.reachable.append(node)

    key = bucket_key(draft)
    state.bucket_counts[key] = state.bucket_counts.get(key, 0) + 1
    state.drafts.append(draft)


def _connection_slots(remaining: dict[RuleFamily, int]) -> int:
    """Count future RHS edges before reserving edges for newly created roots."""

    return (
        remaining[RuleFamily.PARENTHESIS]
        + remaining[RuleFamily.ITERATION]
        + 2 * remaining[RuleFamily.BRANCH]
    )


def _random_index(size: int, generator: torch.Generator) -> int:
    """Draw one index from the explicit topology random stream."""

    if size <= 0:
        raise RuntimeError("cannot sample an empty choice set")
    return int(torch.randint(size, (), generator=generator))


def _random_inclusive(low: int, high: int, generator: torch.Generator) -> int:
    """Draw one integer from an inclusive feasible count interval."""

    return low + _random_index(high - low + 1, generator)


def _shuffle(
    values: list[_Value],
    generator: torch.Generator,
) -> list[_Value]:
    """Return a stream-owned random ordering without mutating the input."""

    if len(values) < 2:
        return values.copy()
    order = torch.randperm(len(values), generator=generator).tolist()
    return [values[index] for index in order]


def _sample_without_replacement(
    values: list[_Value],
    count: int,
    generator: torch.Generator,
) -> list[_Value]:
    """Select distinct list members from the explicit topology stream."""

    if count == 0:
        return []
    if count > len(values):
        raise RuntimeError("cannot sample more unique values than are available")
    order = torch.randperm(len(values), generator=generator)[:count].tolist()
    return [values[index] for index in order]
