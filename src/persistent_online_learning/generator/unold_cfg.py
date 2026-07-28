"""Construct fixed lexicalized CFGs with the productive-first Unold method.

All graph-wide reasoning lives in this module. The returned objects are passive,
sealed runtime data: the syntax graph, the vocabulary, and the mapping between
abstract terminal nodes and concrete token IDs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from itertools import product
from typing import Iterable

import torch

from .grammar import (
    CFG,
    GrammarSymbol,
    LexicalizedCFG,
    Lexicon,
    LexiconEntry,
    Nonterminal,
    Production,
    Terminal,
    Vocabulary,
)

_MAX_RANDOM_ATTEMPTS = 512


def _require_count(name: str, value: int, *, positive: bool = False) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must use int")
    minimum = 1 if positive else 0
    if value < minimum:
        word = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {word}")


@dataclass(frozen=True, slots=True)
class LexiconParameters:
    """Exact lexical coverage requested for the generated language."""

    category_count: int
    vocabulary_size: int
    tokens_per_category: int

    def __post_init__(self) -> None:
        _require_count("category_count", self.category_count, positive=True)
        _require_count("vocabulary_size", self.vocabulary_size, positive=True)
        _require_count(
            "tokens_per_category", self.tokens_per_category, positive=True
        )
        if self.tokens_per_category > self.vocabulary_size:
            raise ValueError(
                "tokens_per_category cannot exceed vocabulary_size when "
                "category membership is unique"
            )
        if self.category_count * self.tokens_per_category < self.vocabulary_size:
            raise ValueError(
                "lexical categories do not provide enough slots to cover the vocabulary"
            )


@dataclass(frozen=True, slots=True)
class UnoldCFGParameters:
    """One exact request for a fixed lexicalized CFG."""

    terminal_pair_rules: int
    parenthesis_rules: int
    iteration_rules: int
    branch_rules: int
    max_nonterminals: int
    lexicon: LexiconParameters

    def __post_init__(self) -> None:
        _require_count("terminal_pair_rules", self.terminal_pair_rules, positive=True)
        _require_count("parenthesis_rules", self.parenthesis_rules)
        _require_count("iteration_rules", self.iteration_rules)
        _require_count("branch_rules", self.branch_rules)
        _require_count("max_nonterminals", self.max_nonterminals, positive=True)
        if not isinstance(self.lexicon, LexiconParameters):
            raise TypeError("lexicon must be LexiconParameters")

        terminal_count = self.lexicon.category_count
        terminal_positions = (
            2 * self.terminal_pair_rules
            + 2 * self.parenthesis_rules
            + self.iteration_rules
        )
        if terminal_count > terminal_positions:
            raise ValueError(
                "syntax rules do not contain enough terminal positions to use every "
                "terminal category"
            )

        terminal_square = terminal_count**2
        if self.terminal_pair_rules > self.max_nonterminals * terminal_square:
            raise ValueError("terminal-pair rule count exceeds symbol capacity")
        minimum_plain_roots = (
            self.terminal_pair_rules + terminal_square - 1
        ) // terminal_square
        connection_slots = (
            self.parenthesis_rules + self.iteration_rules + 2 * self.branch_rules
        )
        if minimum_plain_roots > connection_slots + 1:
            raise ValueError("remaining rules cannot connect the terminal-rule roots")
        if self.parenthesis_rules > self.max_nonterminals**2 * terminal_square:
            raise ValueError("parenthesis rule count exceeds symbol capacity")
        if self.iteration_rules > (
            2 * self.max_nonterminals**2 * terminal_count
        ):
            raise ValueError("iteration rule count exceeds symbol capacity")
        if self.branch_rules > self.max_nonterminals**3:
            raise ValueError("branch rule count exceeds symbol capacity")


class _Family(Enum):
    PARENTHESIS = auto()
    ITERATION = auto()
    BRANCH = auto()


@dataclass(frozen=True, slots=True)
class _Candidate:
    lhs: Nonterminal
    rhs: Production
    creates_lhs: bool


class _Construction:
    """Transient state that builds, verifies, and seals one language."""

    def __init__(self, parameters: UnoldCFGParameters) -> None:
        self.parameters = parameters
        self.nodes: list[Nonterminal] = []
        self.terminals: list[Terminal] = []
        self.current_root: Nonterminal | None = None

    def build(self) -> LexicalizedCFG:
        self._add_terminal_pair_rules()
        assert self.nodes
        self.current_root = self.nodes[-1]

        remaining = {
            _Family.PARENTHESIS: self.parameters.parenthesis_rules,
            _Family.ITERATION: self.parameters.iteration_rules,
            _Family.BRANCH: self.parameters.branch_rules,
        }
        while any(remaining.values()):
            legal: list[tuple[_Family, _Candidate]] = []
            for family, count in remaining.items():
                if not count:
                    continue
                candidate = self._find_random_candidate(family, remaining)
                if candidate is not None:
                    legal.append((family, candidate))
            if not legal:
                raise RuntimeError("no legal rule can complete the requested CFG")
            family, candidate = legal[_random_index(len(legal))]
            self._commit(candidate)
            remaining[family] -= 1

        if self.current_root is None:
            raise RuntimeError("CFG construction ended without a root")
        if len(self.terminals) != self.parameters.lexicon.category_count:
            raise RuntimeError("CFG construction did not use every terminal category")

        entries = _assign_vocabulary(self.terminals, self.parameters.lexicon)
        return _finalize_language(
            start=self.current_root,
            declared_nonterminals=self.nodes,
            declared_terminals=self.terminals,
            entries=entries,
            vocabulary_size=self.parameters.lexicon.vocabulary_size,
        )

    def _add_terminal_pair_rules(self) -> None:
        for completed in range(self.parameters.terminal_pair_rules):
            remaining = self.parameters.terminal_pair_rules - completed - 1
            candidate = self._sample_plain_candidate(remaining)
            if candidate is None:
                candidate = self._find_plain_candidate(remaining)
            if candidate is None:
                raise RuntimeError("no unique terminal-pair rule remains")
            self._commit(candidate)

    def _sample_plain_candidate(self, remaining_plain: int) -> _Candidate | None:
        for _ in range(_MAX_RANDOM_ATTEMPTS):
            lhs, creates_lhs = self._sample_lhs(plain=True)
            if lhs is None:
                return None
            remaining_terminal_positions = (
                2 * remaining_plain
                + 2 * self.parameters.parenthesis_rules
                + self.parameters.iteration_rules
            )
            required_new = max(
                0,
                self.parameters.lexicon.category_count
                - len(self.terminals)
                - remaining_terminal_positions,
            )
            candidate = _Candidate(
                lhs,
                self._sample_terminals(2, required_new=required_new),
                creates_lhs,
            )
            if self._plain_candidate_is_legal(candidate, remaining_plain):
                return candidate
        return None

    def _find_plain_candidate(self, remaining_plain: int) -> _Candidate | None:
        for lhs, creates_lhs in self._lhs_options(plain=True):
            for rhs in self._terminal_options(2):
                candidate = _Candidate(lhs, rhs, creates_lhs)
                if self._plain_candidate_is_legal(candidate, remaining_plain):
                    return candidate
        return None

    def _plain_candidate_is_legal(
        self,
        candidate: _Candidate,
        remaining_plain: int,
    ) -> bool:
        if not candidate.creates_lhs and candidate.rhs in candidate.lhs.alternatives:
            return False

        node_count = len(self.nodes) + int(candidate.creates_lhs)
        terminal_count = self._terminal_count_after(candidate)
        remaining_terminal_positions = (
            2 * remaining_plain
            + 2 * self.parameters.parenthesis_rules
            + self.parameters.iteration_rules
        )
        if terminal_count + remaining_terminal_positions < (
            self.parameters.lexicon.category_count
        ):
            return False

        max_plain_nodes = min(
            self.parameters.max_nonterminals,
            1
            + self.parameters.parenthesis_rules
            + self.parameters.iteration_rules
            + 2 * self.parameters.branch_rules,
        )
        future_nodes = min(max_plain_nodes, node_count + remaining_plain)
        return self.parameters.terminal_pair_rules <= (
            future_nodes * self.parameters.lexicon.category_count**2
        )

    def _find_random_candidate(
        self,
        family: _Family,
        remaining: dict[_Family, int],
    ) -> _Candidate | None:
        for _ in range(_MAX_RANDOM_ATTEMPTS):
            candidate = self._sample_candidate(family, remaining)
            if candidate is not None and self._candidate_is_legal(
                candidate, family, remaining
            ):
                return candidate
        for candidate in self._candidate_options(family):
            if self._candidate_is_legal(candidate, family, remaining):
                return candidate
        return None

    def _sample_candidate(
        self,
        family: _Family,
        remaining: dict[_Family, int],
    ) -> _Candidate | None:
        lhs, creates_lhs = self._sample_lhs(plain=False)
        if lhs is None or self.current_root is None:
            return None

        child_count = 2 if family is _Family.BRANCH else 1
        if creates_lhs:
            children = [self.current_root]
            while len(children) < child_count:
                children.append(self.nodes[_random_index(len(self.nodes))])
            children = _random_order(children)
        else:
            children = [
                self.nodes[_random_index(len(self.nodes))]
                for _ in range(child_count)
            ]

        remaining_after = dict(remaining)
        remaining_after[family] -= 1
        remaining_terminal_positions = (
            2 * remaining_after[_Family.PARENTHESIS]
            + remaining_after[_Family.ITERATION]
        )
        required_new = max(
            0,
            self.parameters.lexicon.category_count
            - len(self.terminals)
            - remaining_terminal_positions,
        )
        terminal_slots = (
            2
            if family is _Family.PARENTHESIS
            else 1
            if family is _Family.ITERATION
            else 0
        )
        if required_new > terminal_slots:
            return None

        if family is _Family.BRANCH:
            rhs: Production = tuple(children)
        elif family is _Family.PARENTHESIS:
            terminals = self._sample_terminals(2, required_new=required_new)
            rhs = (terminals[0], children[0], terminals[1])
        else:
            terminal = self._sample_terminals(1, required_new=required_new)[0]
            rhs = (
                (terminal, children[0])
                if _random_index(2) == 0
                else (children[0], terminal)
            )
        return _Candidate(lhs, rhs, creates_lhs)

    def _candidate_options(self, family: _Family) -> Iterable[_Candidate]:
        if self.current_root is None:
            return
        child_count = 2 if family is _Family.BRANCH else 1
        for lhs, creates_lhs in self._lhs_options(plain=False):
            for children in product(self.nodes, repeat=child_count):
                if creates_lhs and self.current_root not in children:
                    continue
                if family is _Family.BRANCH:
                    yield _Candidate(lhs, tuple(children), creates_lhs)
                elif family is _Family.PARENTHESIS:
                    for terminals in self._terminal_options(2):
                        yield _Candidate(
                            lhs,
                            (terminals[0], children[0], terminals[1]),
                            creates_lhs,
                        )
                else:
                    for terminals in self._terminal_options(1):
                        terminal = terminals[0]
                        yield _Candidate(lhs, (terminal, children[0]), creates_lhs)
                        yield _Candidate(lhs, (children[0], terminal), creates_lhs)

    def _candidate_is_legal(
        self,
        candidate: _Candidate,
        family: _Family,
        remaining: dict[_Family, int],
    ) -> bool:
        if not candidate.creates_lhs and candidate.rhs in candidate.lhs.alternatives:
            return False
        if candidate.creates_lhs:
            if self.current_root not in candidate.rhs:
                return False
            if candidate.lhs in candidate.rhs:
                return False

        remaining_after = dict(remaining)
        remaining_after[family] -= 1
        connection_capacity = (
            remaining_after[_Family.PARENTHESIS]
            + remaining_after[_Family.ITERATION]
            + 2 * remaining_after[_Family.BRANCH]
        )
        root_after = candidate.lhs if candidate.creates_lhs else self.current_root
        if root_after is None:
            return False
        nodes_after = self.nodes + ([candidate.lhs] if candidate.creates_lhs else [])
        if self._hanging_component_count(
            root_after, nodes_after, candidate
        ) > connection_capacity:
            return False

        terminal_count = self._terminal_count_after(candidate)
        remaining_terminal_positions = (
            2 * remaining_after[_Family.PARENTHESIS]
            + remaining_after[_Family.ITERATION]
        )
        if terminal_count + remaining_terminal_positions < (
            self.parameters.lexicon.category_count
        ):
            return False

        rules_left = sum(remaining_after.values())
        future_nodes = min(
            self.parameters.max_nonterminals,
            len(nodes_after) + rules_left,
        )
        terminal_count = self.parameters.lexicon.category_count
        if self.parameters.parenthesis_rules > future_nodes**2 * terminal_count**2:
            return False
        if self.parameters.iteration_rules > 2 * future_nodes**2 * terminal_count:
            return False
        if self.parameters.branch_rules > future_nodes**3:
            return False
        return True

    def _terminal_count_after(self, candidate: _Candidate) -> int:
        candidate_terminals = {
            symbol for symbol in candidate.rhs if isinstance(symbol, Terminal)
        }
        return len(set(self.terminals) | candidate_terminals)

    def _hanging_component_count(
        self,
        root: Nonterminal,
        nodes: list[Nonterminal],
        candidate: _Candidate,
    ) -> int:
        adjacency = {node: [] for node in nodes}
        for node in nodes:
            for alternative in node.alternatives:
                adjacency[node].extend(
                    symbol
                    for symbol in alternative
                    if isinstance(symbol, Nonterminal)
                )
        adjacency[candidate.lhs].extend(
            symbol for symbol in candidate.rhs if isinstance(symbol, Nonterminal)
        )

        reachable = _reachable(root, adjacency)
        unreachable = [node for node in nodes if node not in reachable]
        if not unreachable:
            return 0

        components = _strongly_connected_components(unreachable, adjacency)
        component_of = {
            node: component_index
            for component_index, component in enumerate(components)
            for node in component
        }
        incoming = [False] * len(components)
        for node in unreachable:
            source = component_of[node]
            for child in adjacency[node]:
                if child not in component_of:
                    continue
                target = component_of[child]
                if source != target:
                    incoming[target] = True
        return sum(not value for value in incoming)

    def _sample_lhs(self, *, plain: bool) -> tuple[Nonterminal | None, bool]:
        options = self._lhs_options(plain=plain)
        if not options:
            return None, False
        return options[_random_index(len(options))]

    def _lhs_options(self, *, plain: bool) -> list[tuple[Nonterminal, bool]]:
        options = [(node, False) for node in self.nodes]
        can_create = len(self.nodes) < self.parameters.max_nonterminals
        if plain:
            connection_slots = (
                self.parameters.parenthesis_rules
                + self.parameters.iteration_rules
                + 2 * self.parameters.branch_rules
            )
            can_create = can_create and len(self.nodes) < connection_slots + 1
        if can_create:
            options.append((Nonterminal(f"N{len(self.nodes)}"), True))
        return options

    def _sample_terminals(
        self,
        count: int,
        *,
        required_new: int = 0,
    ) -> tuple[Terminal, ...]:
        if required_new > count:
            raise RuntimeError("remaining terminal positions cannot satisfy the request")
        available = self.terminals.copy()
        initial_count = len(available)
        result: list[Terminal] = []
        for position in range(count):
            can_create = len(available) < self.parameters.lexicon.category_count
            created = len(available) - initial_count
            still_required = max(0, required_new - created)
            positions_left = count - position
            if still_required == positions_left:
                if not can_create:
                    raise RuntimeError("required terminal category cannot be created")
                choice = len(available)
            else:
                choice = _random_index(len(available) + int(can_create))
            if choice == len(available):
                terminal = Terminal(f"T{len(available)}")
                available.append(terminal)
                result.append(terminal)
            else:
                result.append(available[choice])
        return tuple(result)

    def _terminal_options(self, count: int) -> list[tuple[Terminal, ...]]:
        options: list[tuple[Terminal, ...]] = []

        def extend(
            prefix: tuple[Terminal, ...],
            available: tuple[Terminal, ...],
        ) -> None:
            if len(prefix) == count:
                options.append(prefix)
                return
            for terminal in available:
                extend(prefix + (terminal,), available)
            if len(available) < self.parameters.lexicon.category_count:
                terminal = Terminal(f"T{len(available)}")
                extend(prefix + (terminal,), available + (terminal,))

        extend((), tuple(self.terminals))
        return options

    def _commit(self, candidate: _Candidate) -> None:
        if candidate.creates_lhs:
            self.nodes.append(candidate.lhs)
            self.current_root = candidate.lhs
        candidate.lhs.add_alternative(*candidate.rhs)

        for symbol in candidate.rhs:
            if isinstance(symbol, Terminal) and symbol not in self.terminals:
                self.terminals.append(symbol)


def _assign_vocabulary(
    terminals: list[Terminal],
    parameters: LexiconParameters,
) -> tuple[LexiconEntry, ...]:
    """Cover every token ID once, then fill each category independently."""

    assignments: list[list[int]] = [[] for _ in terminals]
    membership: list[set[int]] = [set() for _ in terminals]
    terminal_order = torch.randperm(len(terminals)).tolist()
    vocabulary_order = torch.randperm(parameters.vocabulary_size).tolist()

    for position, token_id in enumerate(vocabulary_order):
        terminal_index = terminal_order[position % len(terminals)]
        assignments[terminal_index].append(token_id)
        membership[terminal_index].add(token_id)

    for terminal_index in range(len(terminals)):
        for token_id in torch.randperm(parameters.vocabulary_size).tolist():
            if len(assignments[terminal_index]) == parameters.tokens_per_category:
                break
            if token_id in membership[terminal_index]:
                continue
            assignments[terminal_index].append(token_id)
            membership[terminal_index].add(token_id)

    return tuple(
        LexiconEntry(terminal, tuple(assignments[index]))
        for index, terminal in enumerate(terminals)
    )


def _finalize_language(
    *,
    start: Nonterminal,
    declared_nonterminals: list[Nonterminal],
    declared_terminals: list[Terminal],
    entries: tuple[LexiconEntry, ...],
    vocabulary_size: int,
) -> LexicalizedCFG:
    """Verify construction output, seal the nodes, and publish passive containers."""

    _require_distinct_declared_nodes(declared_nonterminals, declared_terminals)
    reachable_nonterminals, referenced_terminals = _walk_declared_graph(
        start,
        declared_nonterminals,
        declared_terminals,
    )
    if set(reachable_nonterminals) != set(declared_nonterminals):
        missing = next(
            node for node in declared_nonterminals if node not in reachable_nonterminals
        )
        raise RuntimeError(f"constructed nonterminal {missing.name!r} is unreachable")
    if referenced_terminals != set(declared_terminals):
        missing = next(
            terminal
            for terminal in declared_terminals
            if terminal not in referenced_terminals
        )
        raise RuntimeError(f"constructed terminal {missing.name!r} is unused")

    _require_productive(reachable_nonterminals)
    vocabulary = Vocabulary(vocabulary_size)
    _require_complete_lexicon(declared_terminals, entries, vocabulary)

    for node in (*declared_nonterminals, *declared_terminals):
        node._seal()

    return LexicalizedCFG(
        grammar=CFG(
            start=start,
            nonterminals=tuple(declared_nonterminals),
            terminals=tuple(declared_terminals),
        ),
        lexicon=Lexicon(vocabulary=vocabulary, entries=entries),
    )


def _require_distinct_declared_nodes(
    nonterminals: list[Nonterminal],
    terminals: list[Terminal],
) -> None:
    all_nodes: list[GrammarSymbol] = [*nonterminals, *terminals]
    if len(set(all_nodes)) != len(all_nodes):
        raise RuntimeError("construction declared the same node more than once")
    names: set[str] = set()
    for node in all_nodes:
        if node.name in names:
            raise RuntimeError(f"construction reused node name {node.name!r}")
        names.add(node.name)


def _walk_declared_graph(
    start: Nonterminal,
    declared_nonterminals: list[Nonterminal],
    declared_terminals: list[Terminal],
) -> tuple[list[Nonterminal], set[Terminal]]:
    declared_nonterminal_set = set(declared_nonterminals)
    declared_terminal_set = set(declared_terminals)
    if start not in declared_nonterminal_set:
        raise RuntimeError("CFG start is not one of the declared nonterminals")

    ordered: list[Nonterminal] = []
    seen: set[Nonterminal] = {start}
    terminals: set[Terminal] = set()
    pending: deque[Nonterminal] = deque([start])
    while pending:
        node = pending.popleft()
        ordered.append(node)
        if not node.alternatives:
            raise RuntimeError(f"constructed nonterminal {node.name!r} has no productions")
        for production in node.alternatives:
            for symbol in production:
                if isinstance(symbol, Terminal):
                    if symbol not in declared_terminal_set:
                        raise RuntimeError(
                            f"production references undeclared terminal {symbol.name!r}"
                        )
                    terminals.add(symbol)
                else:
                    if symbol not in declared_nonterminal_set:
                        raise RuntimeError(
                            f"production references undeclared nonterminal {symbol.name!r}"
                        )
                    if symbol not in seen:
                        seen.add(symbol)
                        pending.append(symbol)
    return ordered, terminals


def _require_productive(nonterminals: list[Nonterminal]) -> None:
    unresolved_by_production: dict[Nonterminal, list[int]] = {
        node: [
            sum(isinstance(symbol, Nonterminal) for symbol in production)
            for production in node.alternatives
        ]
        for node in nonterminals
    }
    dependents: dict[Nonterminal, list[tuple[Nonterminal, int]]] = {
        node: [] for node in nonterminals
    }
    productive: set[Nonterminal] = set()
    pending: deque[Nonterminal] = deque()

    for owner in nonterminals:
        for production_index, production in enumerate(owner.alternatives):
            children = [
                symbol for symbol in production if isinstance(symbol, Nonterminal)
            ]
            if not children and owner not in productive:
                productive.add(owner)
                pending.append(owner)
            for child in children:
                dependents[child].append((owner, production_index))

    while pending:
        child = pending.popleft()
        for owner, production_index in dependents[child]:
            unresolved_by_production[owner][production_index] -= 1
            if (
                unresolved_by_production[owner][production_index] == 0
                and owner not in productive
            ):
                productive.add(owner)
                pending.append(owner)

    if len(productive) != len(nonterminals):
        missing = next(node for node in nonterminals if node not in productive)
        raise RuntimeError(
            f"constructed nonterminal {missing.name!r} has no finite terminal derivation"
        )


def _require_complete_lexicon(
    terminals: list[Terminal],
    entries: tuple[LexiconEntry, ...],
    vocabulary: Vocabulary,
) -> None:
    if len(entries) != len(terminals):
        raise RuntimeError("lexicon does not contain one entry per terminal")
    entry_by_terminal: dict[Terminal, LexiconEntry] = {}
    used_token_ids: set[int] = set()
    for entry in entries:
        if entry.terminal in entry_by_terminal:
            raise RuntimeError(
                f"lexicon contains duplicate entry for {entry.terminal.name!r}"
            )
        if entry.terminal not in terminals:
            raise RuntimeError(
                f"lexicon references undeclared terminal {entry.terminal.name!r}"
            )
        entry_by_terminal[entry.terminal] = entry
        for token_id in entry.token_ids:
            if token_id >= vocabulary.size:
                raise RuntimeError(
                    f"lexicon token ID {token_id} is outside vocabulary size "
                    f"{vocabulary.size}"
                )
            used_token_ids.add(token_id)

    missing_terminal = next(
        (terminal for terminal in terminals if terminal not in entry_by_terminal),
        None,
    )
    if missing_terminal is not None:
        raise RuntimeError(
            f"lexicon has no entry for terminal {missing_terminal.name!r}"
        )

    expected = set(range(vocabulary.size))
    if used_token_ids != expected:
        missing_token_id = min(expected - used_token_ids)
        raise RuntimeError(f"lexicon does not use token ID {missing_token_id}")


def _random_index(size: int) -> int:
    if size <= 0:
        raise RuntimeError("cannot sample an empty choice set")
    return int(torch.randint(size, ()))


def _random_order(values: list[Nonterminal]) -> list[Nonterminal]:
    if len(values) < 2:
        return values.copy()
    order = torch.randperm(len(values)).tolist()
    return [values[index] for index in order]


def _reachable(
    root: Nonterminal,
    adjacency: dict[Nonterminal, list[Nonterminal]],
) -> set[Nonterminal]:
    seen = {root}
    pending = [root]
    while pending:
        node = pending.pop()
        for child in adjacency[node]:
            if child not in seen:
                seen.add(child)
                pending.append(child)
    return seen


def _strongly_connected_components(
    nodes: list[Nonterminal],
    adjacency: dict[Nonterminal, list[Nonterminal]],
) -> list[list[Nonterminal]]:
    allowed = set(nodes)
    finish_order: list[Nonterminal] = []
    seen: set[Nonterminal] = set()
    for start in nodes:
        if start in seen:
            continue
        stack: list[tuple[Nonterminal, bool]] = [(start, False)]
        while stack:
            node, finished = stack.pop()
            if finished:
                finish_order.append(node)
                continue
            if node in seen:
                continue
            seen.add(node)
            stack.append((node, True))
            for child in reversed(adjacency[node]):
                if child in allowed and child not in seen:
                    stack.append((child, False))

    reverse_adjacency = {node: [] for node in nodes}
    for node in nodes:
        for child in adjacency[node]:
            if child in allowed:
                reverse_adjacency[child].append(node)

    components: list[list[Nonterminal]] = []
    assigned: set[Nonterminal] = set()
    for start in reversed(finish_order):
        if start in assigned:
            continue
        component: list[Nonterminal] = []
        pending = [start]
        assigned.add(start)
        while pending:
            node = pending.pop()
            component.append(node)
            for parent in reverse_adjacency[node]:
                if parent not in assigned:
                    assigned.add(parent)
                    pending.append(parent)
        components.append(component)
    return components


def generate_unold_cfg(parameters: UnoldCFGParameters) -> LexicalizedCFG:
    """Generate one fixed lexicalized CFG from an exact feasible request."""

    if not isinstance(parameters, UnoldCFGParameters):
        raise TypeError("parameters must be UnoldCFGParameters")
    return _Construction(parameters).build()
