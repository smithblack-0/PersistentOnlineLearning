"""Construct consistent random CFGs using the method of Unold et al.

The implementation preserves the paper's productive-first construction,
hanging terminal-rule components, random remaining-rule-family choice, unique
rules, and final start-root conversion. Random tie-breaking uses the active
PyTorch random stream; seeding and process replacement belong to the caller.
"""

from dataclasses import dataclass, field
from itertools import product

import torch

from .model import CFG, Nonterminal, Production, RuleFamily, Terminal
from .specification import UnoldCFGSpecification
from .validation import validate_unold_cfg

_MAX_RANDOM_ATTEMPTS = 4096


def _random_index(size: int) -> int:
    if size <= 0:
        raise RuntimeError("cannot sample from an empty candidate set")
    return int(torch.randint(size, ()))


def _random_order(values: list[int]) -> list[int]:
    if len(values) < 2:
        return values.copy()
    permutation = torch.randperm(len(values)).tolist()
    return [values[index] for index in permutation]


def _nonterminal_slots(family: RuleFamily) -> int:
    if family in (
        RuleFamily.PARENTHESIS_WITH_NONTERMINAL,
        RuleFamily.ITERATION,
    ):
        return 1
    if family is RuleFamily.BRANCH:
        return 2
    return 0


@dataclass(slots=True)
class _Construction:
    """Build one grammar while preserving productive and connection invariants."""

    specification: UnoldCFGSpecification
    terminals: list[Terminal] = field(default_factory=list)
    nonterminals: list[Nonterminal] = field(default_factory=list)
    productions: list[Production] = field(default_factory=list)
    production_set: set[Production] = field(default_factory=set)
    current_root: Nonterminal | None = None
    connected: set[Nonterminal] = field(default_factory=set)
    hanging: set[Nonterminal] = field(default_factory=set)

    def build(self) -> CFG:
        for _ in range(self.specification.parenthesis_without_nonterminal):
            self._add_plain_parenthesis()

        if not self.nonterminals:
            raise RuntimeError("terminal-rule phase failed to create a nonterminal")
        self.current_root = self.nonterminals[-1]
        self.connected = {self.current_root}
        self.hanging = set(self.nonterminals[:-1])

        remaining = {
            RuleFamily.PARENTHESIS_WITH_NONTERMINAL: (
                self.specification.parenthesis_with_nonterminal
            ),
            RuleFamily.ITERATION: self.specification.iteration_rules,
            RuleFamily.BRANCH: self.specification.branch_rules,
        }
        while any(remaining.values()):
            family, required_hanging = self._choose_remaining_family(remaining)
            self._add_nonterminal_rule(family, required_hanging)
            remaining[family] -= 1

        if self.hanging:
            raise RuntimeError("construction ended with unresolved hanging symbols")
        if self.current_root is None:
            raise RuntimeError("construction ended without a start symbol")

        grammar = CFG(
            start=self.current_root,
            productions=tuple(self.productions),
        )
        validate_unold_cfg(grammar, self.specification)
        return grammar

    def _add_plain_parenthesis(self) -> None:
        for _ in range(_MAX_RANDOM_ATTEMPTS):
            candidate = self._sample_plain_candidate()
            if candidate not in self.production_set:
                self._commit_candidate(candidate)
                return
        candidate = self._find_plain_candidate()
        if candidate is None:
            raise RuntimeError("no unique plain parenthesis rule remains")
        self._commit_candidate(candidate)

    def _sample_plain_candidate(self) -> Production:
        existing_nonterminals = self.nonterminals
        can_create_lhs = self._can_create_plain_lhs()
        if not existing_nonterminals:
            lhs_index = 0
        else:
            candidate_count = len(existing_nonterminals) + int(can_create_lhs)
            choice = _random_index(candidate_count)
            lhs_index = (
                len(existing_nonterminals)
                if choice == len(existing_nonterminals)
                else existing_nonterminals[choice].index
            )

        terminal_indices = self._sample_terminal_indices(2)
        return Production(
            lhs=Nonterminal(lhs_index),
            rhs=tuple(Terminal(index) for index in terminal_indices),
        )

    def _find_plain_candidate(self) -> Production | None:
        lhs_indices = [symbol.index for symbol in self.nonterminals]
        if self._can_create_plain_lhs():
            lhs_indices.append(len(self.nonterminals))
        for lhs_index in lhs_indices:
            for terminal_indices in self._terminal_index_options(2):
                candidate = Production(
                    lhs=Nonterminal(lhs_index),
                    rhs=tuple(Terminal(index) for index in terminal_indices),
                )
                if candidate not in self.production_set:
                    return candidate
        return None

    def _can_create_plain_lhs(self) -> bool:
        if len(self.nonterminals) >= self.specification.max_nonterminals:
            return False
        # Every distinct terminal-only lhs is a separate productive component.
        # Leave at least one future rhs slot for every component except the root.
        available_connection_slots = (
            self.specification.parenthesis_with_nonterminal
            + self.specification.iteration_rules
            + 2 * self.specification.branch_rules
        )
        distinct_after_creation = len(self.nonterminals) + 1
        return distinct_after_creation - 1 <= available_connection_slots

    def _choose_remaining_family(
        self,
        remaining: dict[RuleFamily, int],
    ) -> tuple[RuleFamily, int]:
        legal: list[tuple[RuleFamily, int]] = []
        for family, count in remaining.items():
            if count == 0:
                continue
            slots = _nonterminal_slots(family)
            capacity_after = sum(
                _nonterminal_slots(other_family)
                * (other_count - int(other_family is family))
                for other_family, other_count in remaining.items()
            )
            # Attach enough component roots now that all remaining hanging roots
            # can still fit into later nonterminal rhs positions.
            required_hanging = max(0, len(self.hanging) - capacity_after)
            if required_hanging <= slots:
                legal.append((family, required_hanging))
        if not legal:
            raise RuntimeError("remaining rule quotas cannot connect hanging symbols")
        return legal[_random_index(len(legal))]

    def _add_nonterminal_rule(
        self,
        family: RuleFamily,
        required_hanging: int,
    ) -> None:
        for _ in range(_MAX_RANDOM_ATTEMPTS):
            candidate = self._sample_nonterminal_candidate(family, required_hanging)
            if candidate not in self.production_set:
                self._commit_nonterminal_candidate(candidate)
                return
        # Saturated rule spaces can make rejection sampling arbitrarily slow.
        # Exhaustively locate a remaining legal rule after the ordinary random path.
        candidate = self._find_nonterminal_candidate(family, required_hanging)
        if candidate is None:
            raise RuntimeError(f"no unique {family.value} rule remains")
        self._commit_nonterminal_candidate(candidate)

    def _sample_nonterminal_candidate(
        self,
        family: RuleFamily,
        required_hanging: int,
    ) -> Production:
        if self.current_root is None:
            raise RuntimeError("nonterminal-rule phase has no connected root")
        slots = _nonterminal_slots(family)
        # Existing lhs symbols come from the connected component. Hanging symbols
        # remain component roots to be attached on a rhs, so each forced placement
        # genuinely reduces the number of disconnected roots still outstanding.
        connected = sorted(self.connected, key=lambda symbol: symbol.index)
        can_create_lhs = (
            len(self.nonterminals) < self.specification.max_nonterminals
            and slots >= required_hanging + 1
        )
        lhs_choice = _random_index(len(connected) + int(can_create_lhs))
        create_lhs = lhs_choice == len(connected)
        lhs = (
            Nonterminal(len(self.nonterminals))
            if create_lhs
            else connected[lhs_choice]
        )

        hanging_indices = sorted(symbol.index for symbol in self.hanging)
        mandatory_hanging = _random_order(hanging_indices)[:required_hanging]
        rhs_indices = mandatory_hanging
        if create_lhs:
            rhs_indices.append(self.current_root.index)
        all_indices = [symbol.index for symbol in self.nonterminals]
        while len(rhs_indices) < slots:
            rhs_indices.append(all_indices[_random_index(len(all_indices))])
        rhs_indices = _random_order(rhs_indices)

        if family is RuleFamily.BRANCH:
            rhs = tuple(Nonterminal(index) for index in rhs_indices)
        elif family is RuleFamily.PARENTHESIS_WITH_NONTERMINAL:
            terminal_indices = self._sample_terminal_indices(2)
            rhs = (
                Terminal(terminal_indices[0]),
                Nonterminal(rhs_indices[0]),
                Terminal(terminal_indices[1]),
            )
        elif family is RuleFamily.ITERATION:
            terminal_indices = self._sample_terminal_indices(1)
            terminal = Terminal(terminal_indices[0])
            nonterminal = Nonterminal(rhs_indices[0])
            rhs = (
                (terminal, nonterminal)
                if _random_index(2) == 0
                else (nonterminal, terminal)
            )
        else:
            raise RuntimeError("plain parenthesis rules belong to the first phase")
        return Production(lhs=lhs, rhs=rhs)

    def _find_nonterminal_candidate(
        self,
        family: RuleFamily,
        required_hanging: int,
    ) -> Production | None:
        if self.current_root is None:
            return None
        slots = _nonterminal_slots(family)
        connected_indices = sorted(symbol.index for symbol in self.connected)
        lhs_options: list[tuple[int, bool]] = [
            (index, False) for index in connected_indices
        ]
        if (
            len(self.nonterminals) < self.specification.max_nonterminals
            and slots >= required_hanging + 1
        ):
            lhs_options.append((len(self.nonterminals), True))

        all_indices = [symbol.index for symbol in self.nonterminals]
        hanging_indices = {symbol.index for symbol in self.hanging}
        terminal_options = (
            self._terminal_index_options(2)
            if family is RuleFamily.PARENTHESIS_WITH_NONTERMINAL
            else self._terminal_index_options(1)
            if family is RuleFamily.ITERATION
            else [()]
        )
        cached_terminals = list(terminal_options)

        for lhs_index, create_lhs in lhs_options:
            for rhs_indices in product(all_indices, repeat=slots):
                if len(set(rhs_indices) & hanging_indices) < required_hanging:
                    continue
                if create_lhs and self.current_root.index not in rhs_indices:
                    continue
                for terminal_indices in cached_terminals:
                    if family is RuleFamily.BRANCH:
                        candidates = [
                            Production(
                                lhs=Nonterminal(lhs_index),
                                rhs=tuple(
                                    Nonterminal(index) for index in rhs_indices
                                ),
                            )
                        ]
                    elif family is RuleFamily.PARENTHESIS_WITH_NONTERMINAL:
                        candidates = [
                            Production(
                                lhs=Nonterminal(lhs_index),
                                rhs=(
                                    Terminal(terminal_indices[0]),
                                    Nonterminal(rhs_indices[0]),
                                    Terminal(terminal_indices[1]),
                                ),
                            )
                        ]
                    else:
                        terminal = Terminal(terminal_indices[0])
                        nonterminal = Nonterminal(rhs_indices[0])
                        candidates = [
                            Production(
                                lhs=Nonterminal(lhs_index),
                                rhs=(terminal, nonterminal),
                            ),
                            Production(
                                lhs=Nonterminal(lhs_index),
                                rhs=(nonterminal, terminal),
                            ),
                        ]
                    for candidate in candidates:
                        if candidate not in self.production_set:
                            return candidate
        return None

    def _sample_terminal_indices(self, count: int) -> list[int]:
        existing = [symbol.index for symbol in self.terminals]
        result: list[int] = []
        for _ in range(count):
            can_create = len(existing) < self.specification.max_terminals
            choice = _random_index(len(existing) + int(can_create))
            if choice == len(existing):
                new_index = len(existing)
                existing.append(new_index)
                result.append(new_index)
            else:
                result.append(existing[choice])
        return result

    def _terminal_index_options(self, count: int) -> list[tuple[int, ...]]:
        existing = tuple(symbol.index for symbol in self.terminals)
        options: list[tuple[int, ...]] = []

        def extend(prefix: tuple[int, ...], available: tuple[int, ...]) -> None:
            if len(prefix) == count:
                options.append(prefix)
                return
            for index in available:
                extend(prefix + (index,), available)
            if len(available) < self.specification.max_terminals:
                new_index = len(available)
                extend(prefix + (new_index,), available + (new_index,))

        extend((), existing)
        return options

    def _commit_candidate(self, production: Production) -> None:
        self._commit_symbols(production)
        self.productions.append(production)
        self.production_set.add(production)

    def _commit_nonterminal_candidate(self, production: Production) -> None:
        previous_count = len(self.nonterminals)
        previous_root = self.current_root
        self._commit_candidate(production)
        created_lhs = len(self.nonterminals) > previous_count
        if created_lhs:
            if previous_root is None or previous_root not in production.rhs:
                raise RuntimeError("new root rule does not reference the prior root")
            self.current_root = production.lhs
        if self.current_root is None:
            raise RuntimeError("committed rule lost the connected root")
        self.connected = self._reachable_nonterminals(self.current_root)
        self.hanging = set(self.nonterminals) - self.connected

    def _commit_symbols(self, production: Production) -> None:
        if production.lhs.index == len(self.nonterminals):
            self.nonterminals.append(production.lhs)
        elif production.lhs not in self.nonterminals:
            raise RuntimeError("production uses a non-contiguous new lhs")
        for symbol in production.rhs:
            if isinstance(symbol, Terminal):
                if symbol.index == len(self.terminals):
                    self.terminals.append(symbol)
                elif symbol not in self.terminals:
                    raise RuntimeError("production uses a non-contiguous new terminal")

    def _reachable_nonterminals(self, root: Nonterminal) -> set[Nonterminal]:
        reachable = {root}
        changed = True
        while changed:
            changed = False
            for production in self.productions:
                if production.lhs not in reachable:
                    continue
                for symbol in production.rhs:
                    if isinstance(symbol, Nonterminal) and symbol not in reachable:
                        reachable.add(symbol)
                        changed = True
        return reachable


def generate_unold_cfg(specification: UnoldCFGSpecification) -> CFG:
    """Construct one random consistent CFG from an exact feasible specification."""

    if not isinstance(specification, UnoldCFGSpecification):
        raise TypeError("specification must be an UnoldCFGSpecification")
    return _Construction(specification).build()
