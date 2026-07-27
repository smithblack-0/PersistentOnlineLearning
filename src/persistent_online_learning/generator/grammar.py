"""One coherent representation of a generated context-free grammar.

A grammar owns an indexed set of nonterminal nodes. Each node owns its possible
expansions, and each expansion is one of the rule forms used by the synthetic
language system. Construction rejects malformed, disconnected, or unproductive
grammars so downstream code never has to recreate those invariants.
"""

from collections import deque
from dataclasses import dataclass
from typing import NamedTuple, TypeAlias


class TerminalPairRule(NamedTuple):
    """Emit two terminal categories."""

    left_terminal: int
    right_terminal: int


class ParenthesisRule(NamedTuple):
    """Emit a terminal, expand one child, then emit another terminal."""

    left_terminal: int
    child: int
    right_terminal: int


class IterationRule(NamedTuple):
    """Emit one terminal beside one child expansion."""

    terminal: int
    child: int
    terminal_first: bool


class BranchRule(NamedTuple):
    """Expand two child nonterminals in order."""

    left_child: int
    right_child: int


Rule: TypeAlias = (
    TerminalPairRule
    | ParenthesisRule
    | IterationRule
    | BranchRule
)


@dataclass(frozen=True, slots=True)
class Grammar:
    """A complete recursive grammar whose nodes own their alternatives.

    ``nodes[index]`` is the tuple of alternatives for one nonterminal. Rule
    child fields refer directly to indices in that tuple. The representation is
    immutable after construction and permits recursion, but every node must be
    reachable from ``root`` and have at least one finite terminal derivation.
    """

    root: int
    nodes: tuple[tuple[Rule, ...], ...]

    def __post_init__(self) -> None:
        _require_index("root", self.root)
        if type(self.nodes) is not tuple or not self.nodes:
            raise TypeError("nodes must be a nonempty tuple")
        if self.root >= len(self.nodes):
            raise ValueError("root is outside the grammar")

        children_by_node: list[tuple[int, ...]] = []
        for index, alternatives in enumerate(self.nodes):
            if type(alternatives) is not tuple or not alternatives:
                raise TypeError(f"node {index} alternatives must be a nonempty tuple")
            if not all(isinstance(rule, _RULE_TYPES) for rule in alternatives):
                raise TypeError(f"node {index} contains an invalid rule")
            if len(set(alternatives)) != len(alternatives):
                raise ValueError(f"node {index} contains duplicate alternatives")

            children: list[int] = []
            for rule in alternatives:
                rule_children, terminals = _rule_parts(rule)
                for terminal in terminals:
                    _require_index("terminal category", terminal)
                for child in rule_children:
                    _require_index("child", child)
                    if child >= len(self.nodes):
                        raise ValueError(
                            f"node {index} references child {child} outside the grammar"
                        )
                children.extend(rule_children)
            children_by_node.append(tuple(children))

        reachable = {self.root}
        pending = [self.root]
        while pending:
            node = pending.pop()
            for child in children_by_node[node]:
                if child not in reachable:
                    reachable.add(child)
                    pending.append(child)
        if len(reachable) != len(self.nodes):
            missing = min(set(range(len(self.nodes))) - reachable)
            raise ValueError(f"node {missing} is unreachable from the root")

        unresolved_by_rule = [
            [len(_rule_parts(rule)[0]) for rule in alternatives]
            for alternatives in self.nodes
        ]
        dependents: list[list[tuple[int, int]]] = [[] for _ in self.nodes]
        productive: set[int] = set()
        pending_productive: deque[int] = deque()
        for owner, alternatives in enumerate(self.nodes):
            for alternative, rule in enumerate(alternatives):
                children, _ = _rule_parts(rule)
                if not children and owner not in productive:
                    productive.add(owner)
                    pending_productive.append(owner)
                for child in children:
                    dependents[child].append((owner, alternative))

        while pending_productive:
            child = pending_productive.popleft()
            for owner, alternative in dependents[child]:
                unresolved_by_rule[owner][alternative] -= 1
                if (
                    unresolved_by_rule[owner][alternative] == 0
                    and owner not in productive
                ):
                    productive.add(owner)
                    pending_productive.append(owner)

        if len(productive) != len(self.nodes):
            missing = min(set(range(len(self.nodes))) - productive)
            raise ValueError(f"node {missing} has no finite terminal derivation")

    def alternatives(self, node: int) -> tuple[Rule, ...]:
        """Return the alternatives owned by one nonterminal node."""

        self._require_node(node)
        return self.nodes[node]

    def children(self, node: int) -> frozenset[int]:
        """Return the nonterminal nodes directly referenced by one node."""

        self._require_node(node)
        return frozenset(
            child
            for rule in self.nodes[node]
            for child in _rule_parts(rule)[0]
        )

    @property
    def terminal_categories(self) -> frozenset[int]:
        """Return every terminal-category index used by the grammar."""

        return frozenset(
            terminal
            for alternatives in self.nodes
            for rule in alternatives
            for terminal in _rule_parts(rule)[1]
        )

    @property
    def rule_count(self) -> int:
        """Number of alternatives across all nonterminal nodes."""

        return sum(len(alternatives) for alternatives in self.nodes)

    def _require_node(self, node: int) -> None:
        _require_index("node", node)
        if node >= len(self.nodes):
            raise ValueError("node is outside the grammar")


_RULE_TYPES = (TerminalPairRule, ParenthesisRule, IterationRule, BranchRule)


def _require_index(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must use int")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _rule_parts(rule: Rule) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if isinstance(rule, TerminalPairRule):
        return (), (rule.left_terminal, rule.right_terminal)
    if isinstance(rule, ParenthesisRule):
        return (rule.child,), (rule.left_terminal, rule.right_terminal)
    if isinstance(rule, IterationRule):
        if type(rule.terminal_first) is not bool:
            raise TypeError("terminal_first must use bool")
        return (rule.child,), (rule.terminal,)
    return (rule.left_child, rule.right_child), ()
