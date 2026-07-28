"""Build the productive nonterminal topology of the requested CFG.

This phase implements the productive-first core of Unold et al.: terminal-only
rules establish productive symbols, later RHS nonterminals are already productive,
and a newly created LHS contains the previous root and becomes the new root.

The project makes two deliberate construction adaptations. First, a
``ConstructionPlan`` fixes how many nonterminals will be created before this phase
starts. Second, existing-LHS extensions are restricted to the already reachable
component, while productive but disconnected foundation roots are attached through
future RHS edges. Together these choices make the hanging-root edge budget explicit
instead of relying on candidate creation followed by future-repair checks.

Terminal *identity* is not selected here. This module accounts only for the finite
number of unique terminal labelings available to each fixed topology bucket; the
later terminal-assignment phase chooses the actual ``TerminalVocabulary`` nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterator, TypeVar

import torch

from ..config import GrammarConfig
from ..grammar import Nonterminal
from .planning import ConstructionPlan
from .rules import (
    RuleDraft,
    RuleFamily,
    bucket_key,
    child_count,
    terminal_label_capacity,
)

_RANDOM_ATTEMPTS = 128
_Value = TypeVar("_Value")


# ---------------------------------------------------------------------------
# Phase products and state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SyntaxTopology:
    """Expose the completed topology to the terminal-assignment phase.

    The object exists only across one construction boundary: topology has fixed
    every nonterminal node and RHS edge, while terminal positions are still open.
    Terminal assignment consumes the drafts, populates their LHS nodes, and then
    this transient representation is discarded.

    Attributes:
        start: Final root selected by productive-first construction.
        nonterminals: Every nonterminal node created for this grammar.
        drafts: Rule topologies awaiting concrete terminal-node labels.
    """

    start: Nonterminal
    nonterminals: tuple[Nonterminal, ...]
    drafts: tuple[RuleDraft, ...]


@dataclass(slots=True)
class _SyntaxState:
    """Hold only the evolving facts needed to preserve topology invariants.

    ``reachable`` records nodes already connected beneath the current root;
    ``hanging`` records productive foundation roots that still need an incoming
    path from that component. ``bucket_counts`` reserves enough terminal-label
    capacity for final production uniqueness without choosing terminal identities.
    The state dies when ``SyntaxTopology`` is published.
    """

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
    """Represent one invariant-preserving rule topology before state mutation.

    Separating selection from commitment lets random sampling and deterministic
    fallback share the same capacity check, while ``_commit_topology`` remains a
    simple mutation of already-legal construction state.
    """

    draft: RuleDraft
    creates_lhs: bool


# ---------------------------------------------------------------------------
# Public topology phase
# ---------------------------------------------------------------------------


def build_unold_topology(
    grammar: GrammarConfig,
    plan: ConstructionPlan,
    terminal_count: int,
    generator: torch.Generator,
) -> SyntaxTopology:
    """Construct all nonterminal nodes and rule topologies for one plan.

    The caller has already built the terminal alphabet and selected a feasible
    ``ConstructionPlan``. This phase creates the productive terminal-only
    foundation, then consumes the remaining rule counts while keeping every RHS
    productive, connecting all hanging roots, reaching the planned node count, and
    reserving enough terminal-label capacity for unique final productions.
    """

    state = _create_plain_foundation(grammar, plan, terminal_count, generator)
    _extend_productive_graph(state, grammar, terminal_count)
    return SyntaxTopology(
        start=state.root,
        nonterminals=tuple(state.nonterminals),
        drafts=tuple(state.drafts),
    )


# ---------------------------------------------------------------------------
# Productive foundation
# ---------------------------------------------------------------------------


def _create_plain_foundation(
    grammar: GrammarConfig,
    plan: ConstructionPlan,
    terminal_count: int,
    generator: torch.Generator,
) -> _SyntaxState:
    """Create the productive ``A -> a b`` roots required before later rule types.

    Unold construction begins with terminal-only rules so every nonterminal used by
    later RHS positions is already productive. The selected plan determines how
    many distinct initial LHS nodes to create. The newest becomes the reachable
    root; earlier roots are productive but hanging until later rules connect them.
    """

    initial_count = _random_inclusive(
        plan.initial_nonterminal_min,
        plan.initial_nonterminal_max,
        generator,
    )
    nonterminals = [Nonterminal(f"N{index}") for index in range(initial_count)]
    root = nonterminals[-1]
    drafts: list[RuleDraft] = []
    bucket_counts: dict[tuple[object, ...], int] = {}

    # Give every foundation node one productive rule, then distribute the remaining
    # terminal-only rules without exhausting the finite T^2 labelings for any LHS.
    rules_per_lhs = [1] * initial_count
    unassigned = grammar.terminal_pair_rules - initial_count
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


# ---------------------------------------------------------------------------
# Productive topology extension
# ---------------------------------------------------------------------------


def _extend_productive_graph(
    state: _SyntaxState,
    grammar: GrammarConfig,
    terminal_count: int,
) -> None:
    """Consume all non-plain rules while preserving the planned graph lifecycle.

    At each step the algorithm knows exactly how many later rules must create new
    LHS nodes. Each such creation reserves one future RHS edge for the previous
    root, so only the remaining child edges are available to attach hanging
    productive roots. Candidates are sampled only from topologies that can satisfy
    the currently required attachments and still admit a unique terminal labeling.
    """

    remaining = {
        RuleFamily.PARENTHESIS: grammar.parenthesis_rules,
        RuleFamily.ITERATION: grammar.iteration_rules,
        RuleFamily.BRANCH: grammar.branch_rules,
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

    # These are internal lifecycle assertions, not a second graph audit: if either
    # fails, the topology phase itself violated the plan it was maintaining.
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
    """Find one topology that preserves connectivity and final-label capacity.

    Random sampling supplies the normal stochastic behavior. The deterministic
    enumeration is only a completeness fallback for the finite candidate set when
    those samples miss an available topology; it does not repair an invalid graph
    or relax any construction invariant.
    """

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
    """Sample a candidate with mandatory root ancestry and hanging attachments.

    A new LHS always receives the current root as one RHS child, implementing the
    paper's newest-symbol root progression. ``required_hanging`` reserves any
    additional child positions that must connect still-disconnected productive
    roots now in order for the remaining edge budget to stay feasible.
    """

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
    """Enumerate the finite legal topology space after random sampling misses.

    Enumeration preserves the same root-ancestry and minimum hanging-attachment
    contracts as stochastic sampling. It exists so a feasible finite choice is not
    incorrectly reported impossible merely because 128 random samples collided
    with already-exhausted topology buckets.
    """

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
    """Check that this fixed topology still has an unused terminal labeling.

    Rule uniqueness is required by the paper. Because terminal identities are
    intentionally delayed, topology tracks only how many drafts already compete in
    the same bucket and rejects the next draft once all possible labels are reserved.
    """

    used = state.bucket_counts.get(bucket_key(draft), 0)
    return used < terminal_label_capacity(draft, terminal_count)


def _commit_topology(
    state: _SyntaxState,
    candidate: _TopologyCandidate,
    terminal_count: int,
) -> None:
    """Apply one already-legal candidate to the transient topology state.

    Commitment updates the root/reachability lifecycle, removes newly attached
    hanging roots, and reserves one terminal-label slot for the draft's topology
    bucket. It does not perform candidate search or revisit earlier choices.
    """

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
    """Count future nonterminal RHS edges before reserving new-root ancestry."""

    return (
        remaining[RuleFamily.PARENTHESIS]
        + remaining[RuleFamily.ITERATION]
        + 2 * remaining[RuleFamily.BRANCH]
    )


# ---------------------------------------------------------------------------
# Random-selection helpers
# ---------------------------------------------------------------------------


def _random_index(size: int, generator: torch.Generator) -> int:
    """Draw one index from the explicit syntax random stream."""

    if size <= 0:
        raise RuntimeError("cannot sample an empty choice set")
    return int(torch.randint(size, (), generator=generator))


def _random_inclusive(low: int, high: int, generator: torch.Generator) -> int:
    """Draw one integer from an inclusive feasible construction interval."""

    return low + _random_index(high - low + 1, generator)


def _shuffle(
    values: list[_Value],
    generator: torch.Generator,
) -> list[_Value]:
    """Return a syntax-stream-owned random ordering without mutating the input."""

    if len(values) < 2:
        return values.copy()
    order = torch.randperm(len(values), generator=generator).tolist()
    return [values[index] for index in order]


def _sample_without_replacement(
    values: list[_Value],
    count: int,
    generator: torch.Generator,
) -> list[_Value]:
    """Select distinct list members required by one topology candidate."""

    if count == 0:
        return []
    if count > len(values):
        raise RuntimeError("cannot sample more unique values than are available")
    order = torch.randperm(len(values), generator=generator)[:count].tolist()
    return [values[index] for index in order]
