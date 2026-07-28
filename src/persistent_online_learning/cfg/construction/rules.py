"""Represent the four Unold rule families between topology and terminal labeling.

The rule shapes come directly from sections 2.2-2.3 of:

O. Unold, A. Kaczmarek, and Ł. Culer, "Iterative method of generating
artificial context-free grammars," arXiv:1911.05801, 2019.

Construction separates nonterminal topology from terminal identity so connectivity
can be established without entangling it with finite terminal-label capacity.
``RuleDraft`` is the small transient representation at that phase boundary; it is
not part of the published grammar and does not introduce a second runtime graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from ..grammar import Nonterminal, Production, TerminalVocabulary


class RuleFamily(Enum):
    """Identify the paper rule shape whose remaining slots a draft represents."""

    PLAIN_PARENTHESIS = auto()  # A -> a b
    PARENTHESIS = auto()  # A -> a B b
    ITERATION = auto()  # A -> a B or B a
    BRANCH = auto()  # A -> B C


@dataclass(frozen=True, slots=True)
class RuleDraft:
    """Carry topology decisions from graph construction to terminal assignment.

    ``lhs`` and ``children`` completely determine the nonterminal topology. Terminal
    positions intentionally remain unlabeled until the later assignment phase can
    satisfy exact terminal use and unique-rule capacity across the whole draft set.

    Attributes:
        family: Rule shape that determines terminal and child slot counts.
        lhs: Nonterminal that will own the realized production.
        children: Already-productive RHS nonterminals fixed by topology construction.
        terminal_first: For iteration rules, whether the terminal precedes the
            child; ignored by the other rule families.
    """

    family: RuleFamily
    lhs: Nonterminal
    children: tuple[Nonterminal, ...] = ()
    terminal_first: bool = True


def terminal_arity(family: RuleFamily) -> int:
    """Return how many terminal nodes terminal assignment must supply for a rule."""

    if family in (RuleFamily.PLAIN_PARENTHESIS, RuleFamily.PARENTHESIS):
        return 2
    if family is RuleFamily.ITERATION:
        return 1
    return 0


def child_count(family: RuleFamily) -> int:
    """Return the nonterminal RHS edge count consumed by this rule family."""

    return 2 if family is RuleFamily.BRANCH else 1


def bucket_key(draft: RuleDraft) -> tuple[object, ...]:
    """Identify drafts that compete for the same finite terminal-label namespace.

    Two drafts with the same fixed LHS/topology become duplicate productions if
    they receive the same terminal tuple. Grouping them by this key lets topology
    reserve enough label capacity before terminal identities are chosen.
    """

    if draft.family is RuleFamily.PLAIN_PARENTHESIS:
        return (draft.family, draft.lhs)
    if draft.family is RuleFamily.PARENTHESIS:
        return (draft.family, draft.lhs, draft.children[0])
    if draft.family is RuleFamily.ITERATION:
        return (
            draft.family,
            draft.lhs,
            draft.children[0],
            draft.terminal_first,
        )
    return (draft.family, draft.lhs, *draft.children)


def terminal_label_capacity(draft: RuleDraft, terminal_count: int) -> int:
    """Count distinct final productions available inside one topology bucket."""

    if draft.family in (RuleFamily.PLAIN_PARENTHESIS, RuleFamily.PARENTHESIS):
        return terminal_count**2
    if draft.family is RuleFamily.ITERATION:
        return terminal_count
    return 1


def realize_production(
    draft: RuleDraft,
    terminals: tuple[TerminalVocabulary, ...],
) -> Production:
    """Materialize one finished RHS from a topology draft and chosen terminals.

    This is the only translation between transient ``RuleDraft`` state and the
    published graph representation. After realization, the ordinary ``Nonterminal``
    owns a direct production over graph nodes and the draft can be discarded.
    """

    if draft.family is RuleFamily.PLAIN_PARENTHESIS:
        return terminals
    if draft.family is RuleFamily.PARENTHESIS:
        return (terminals[0], draft.children[0], terminals[1])
    if draft.family is RuleFamily.ITERATION:
        return (
            (terminals[0], draft.children[0])
            if draft.terminal_first
            else (draft.children[0], terminals[0])
        )
    return draft.children
