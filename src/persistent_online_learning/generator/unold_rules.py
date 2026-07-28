"""Rule-family semantics shared by Unold construction phases.

The four shapes follow sections 2.2-2.3 of:

O. Unold, A. Kaczmarek, and Ł. Culer, "Iterative method of generating
artificial context-free grammars," arXiv:1911.05801, 2019.

Drafts contain complete nonterminal topology but leave terminal positions open.
That phase boundary lets connectivity and terminal-label capacity be reasoned
about independently without changing the directly traversable published graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from .grammar import Nonterminal, Production, TerminalVocabulary


class RuleFamily(Enum):
    """The four production shapes admitted by the paper's generator."""

    PLAIN_PARENTHESIS = auto()  # A -> a b
    PARENTHESIS = auto()  # A -> a B b
    ITERATION = auto()  # A -> a B or B a
    BRANCH = auto()  # A -> B C


@dataclass(frozen=True, slots=True)
class RuleDraft:
    """One topology-complete production awaiting terminal-node assignment."""

    family: RuleFamily
    lhs: Nonterminal
    children: tuple[Nonterminal, ...] = ()
    terminal_first: bool = True


def terminal_arity(family: RuleFamily) -> int:
    """Return terminal positions supplied by one production family."""

    if family in (RuleFamily.PLAIN_PARENTHESIS, RuleFamily.PARENTHESIS):
        return 2
    if family is RuleFamily.ITERATION:
        return 1
    return 0


def child_count(family: RuleFamily) -> int:
    """Return nonterminal right-hand-side positions in a non-plain rule."""

    return 2 if family is RuleFamily.BRANCH else 1


def bucket_key(draft: RuleDraft) -> tuple[object, ...]:
    """Identify drafts sharing one finite terminal-label capacity."""

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
    """Count distinct terminal assignments available to one topology bucket."""

    if draft.family in (RuleFamily.PLAIN_PARENTHESIS, RuleFamily.PARENTHESIS):
        return terminal_count**2
    if draft.family is RuleFamily.ITERATION:
        return terminal_count
    return 1


def realize_production(
    draft: RuleDraft,
    terminals: tuple[TerminalVocabulary, ...],
) -> Production:
    """Translate one topology draft into its final graph-node sequence."""

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
