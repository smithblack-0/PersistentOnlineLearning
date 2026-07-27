"""Independent correctness checks for generated CFGs."""

from collections import Counter

from .model import CFG, Nonterminal, RuleFamily, Terminal
from .specification import UnoldCFGSpecification


def validate_cfg(grammar: CFG) -> None:
    """Require every rule and symbol to be reachable and productive."""

    if len(set(grammar.productions)) != len(grammar.productions):
        raise ValueError("grammar contains duplicate productions")

    for production in grammar.productions:
        production.family

    reachable_nonterminals = {grammar.start}
    reachable_terminals: set[Terminal] = set()
    reachable_rules = set()
    changed = True
    while changed:
        changed = False
        for production in grammar.productions:
            if production.lhs not in reachable_nonterminals:
                continue
            if production not in reachable_rules:
                reachable_rules.add(production)
                changed = True
            for symbol in production.rhs:
                if isinstance(symbol, Nonterminal):
                    if symbol not in reachable_nonterminals:
                        reachable_nonterminals.add(symbol)
                        changed = True
                elif symbol not in reachable_terminals:
                    reachable_terminals.add(symbol)
                    changed = True

    if reachable_rules != set(grammar.productions):
        raise ValueError("grammar contains an unreachable production")
    if reachable_nonterminals != set(grammar.nonterminals):
        raise ValueError("grammar contains an unreachable nonterminal")
    if reachable_terminals != set(grammar.terminals):
        raise ValueError("grammar contains an unreachable terminal")

    productive_nonterminals: set[Nonterminal] = set()
    productive_rules = set()
    changed = True
    while changed:
        changed = False
        for production in grammar.productions:
            if all(
                isinstance(symbol, Terminal) or symbol in productive_nonterminals
                for symbol in production.rhs
            ):
                if production not in productive_rules:
                    productive_rules.add(production)
                    changed = True
                if production.lhs not in productive_nonterminals:
                    productive_nonterminals.add(production.lhs)
                    changed = True

    if productive_rules != set(grammar.productions):
        raise ValueError("grammar contains an unproductive production")
    if productive_nonterminals != set(grammar.nonterminals):
        raise ValueError("grammar contains an unproductive nonterminal")


def validate_unold_cfg(
    grammar: CFG,
    specification: UnoldCFGSpecification,
) -> None:
    """Verify consistency plus the exact requested Unold construction limits."""

    validate_cfg(grammar)
    counts = Counter(production.family for production in grammar.productions)
    expected = {
        RuleFamily.PARENTHESIS_WITHOUT_NONTERMINAL: (
            specification.parenthesis_without_nonterminal
        ),
        RuleFamily.PARENTHESIS_WITH_NONTERMINAL: (
            specification.parenthesis_with_nonterminal
        ),
        RuleFamily.ITERATION: specification.iteration_rules,
        RuleFamily.BRANCH: specification.branch_rules,
    }
    if dict(counts) != {family: count for family, count in expected.items() if count}:
        raise ValueError("grammar rule-family counts do not match the specification")
    if len(grammar.terminals) > specification.max_terminals:
        raise ValueError("grammar exceeds the terminal-symbol maximum")
    if len(grammar.nonterminals) > specification.max_nonterminals:
        raise ValueError("grammar exceeds the nonterminal-symbol maximum")
