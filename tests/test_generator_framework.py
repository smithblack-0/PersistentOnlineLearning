"""Tests for generator construction contracts and static CFG generation."""

import pytest
import torch

from persistent_online_learning.generator import (
    CFG,
    LexiconParameters,
    Nonterminal,
    SimpleEpsilonMachine,
    Terminal,
    TokenGenerator,
    UnoldCFGParameters,
    build_generator,
    generate_unold_cfg,
    register_generator,
)


def _specification() -> dict[str, object]:
    return {
        "type": "simple_epsilon",
        "vocab_size": 64,
        "state_count": 5,
        "hash_count": 8,
        "outcomes_per_state": 4,
        "epsilon": 0.05,
    }


def test_registry_dispatches_without_mutating_specification() -> None:
    specification = _specification()
    original = dict(specification)
    generator = build_generator(specification)
    assert isinstance(generator, SimpleEpsilonMachine)
    assert specification == original


def test_registry_reports_missing_and_unknown_type() -> None:
    with pytest.raises(ValueError, match="requires a type"):
        build_generator({"vocab_size": 64})
    with pytest.raises(ValueError, match="unknown generator type"):
        build_generator({"type": "missing"})


def test_registry_accepts_an_external_factory() -> None:
    class ConstantGenerator(TokenGenerator):
        def step(self) -> int:
            return 3

        def state_dict(self) -> dict[str, object]:
            return {}

        def load_state_dict(self, state: dict[str, object]) -> None:
            return None

    def build_constant() -> TokenGenerator:
        return ConstantGenerator()

    register_generator("test_constant", build_constant)
    generator = build_generator({"type": "test_constant"})
    assert generator.step() == 3


def test_default_generate_uses_repeated_transition() -> None:
    generator = build_generator(_specification())
    tokens = generator.generate(12)
    assert tokens.shape == (12,)


def _cfg_signature(grammar: CFG) -> tuple[object, ...]:
    terminal_vocabulary = tuple(
        (terminal.category, terminal.vocabulary) for terminal in grammar.terminals
    )
    rules = tuple(
        (
            node.name,
            tuple(
                tuple(
                    ("terminal", symbol.category)
                    if isinstance(symbol, Terminal)
                    else ("nonterminal", symbol.name)
                    for symbol in alternative
                )
                for alternative in node.alternatives
            ),
        )
        for node in grammar.nonterminals
    )
    return terminal_vocabulary, rules


def _generated_family_counts(grammar: CFG) -> tuple[int, int, int, int]:
    terminal_pairs = 0
    parenthesis = 0
    iteration = 0
    branches = 0
    for node in grammar.nonterminals:
        for alternative in node.alternatives:
            nonterminal_positions = tuple(
                isinstance(symbol, Nonterminal) for symbol in alternative
            )
            if len(alternative) == 2 and nonterminal_positions == (False, False):
                terminal_pairs += 1
            elif len(alternative) == 3 and nonterminal_positions == (
                False,
                True,
                False,
            ):
                parenthesis += 1
            elif len(alternative) == 2 and nonterminal_positions in (
                (False, True),
                (True, False),
            ):
                iteration += 1
            elif len(alternative) == 2 and nonterminal_positions == (True, True):
                branches += 1
            else:
                raise AssertionError(
                    f"unexpected generated alternative: {alternative!r}"
                )
    return terminal_pairs, parenthesis, iteration, branches


def test_cfg_centralizes_syntax_and_lexical_ownership() -> None:
    noun = Terminal(0)
    noun.set_vocabulary((0, 2))
    verb = Terminal(1)
    verb.set_vocabulary((1, 3))
    sentence = Nonterminal("sentence")
    phrase = Nonterminal("phrase")

    sentence.add_alternative(phrase, verb)
    phrase.add_alternative(noun, noun)
    phrase.add_alternative(noun, phrase)

    grammar = CFG(sentence, vocabulary_size=4)

    assert grammar.start is sentence
    assert grammar.nonterminals == (sentence, phrase)
    assert grammar.terminals == (noun, verb)
    assert grammar.rule_count == 3
    assert noun.vocabulary == (0, 2)
    with pytest.raises(RuntimeError, match="sealed CFG"):
        phrase.add_alternative(noun)
    with pytest.raises(RuntimeError, match="sealed CFG"):
        noun.set_vocabulary((0,))


def test_cfg_accepts_productive_recursion_and_rejects_unproductive_cycles() -> None:
    terminal = Terminal(0)
    terminal.set_vocabulary((0,))
    left = Nonterminal("left")
    right = Nonterminal("right")
    left.add_alternative(right)
    right.add_alternative(left)
    right.add_alternative(terminal)
    CFG(left, vocabulary_size=1)

    bad_terminal = Terminal(0)
    bad_terminal.set_vocabulary((0,))
    bad_root = Nonterminal("bad_root")
    bad_left = Nonterminal("bad_left")
    bad_right = Nonterminal("bad_right")
    bad_root.add_alternative(bad_terminal)
    bad_root.add_alternative(bad_left)
    bad_left.add_alternative(bad_right)
    bad_right.add_alternative(bad_left)
    with pytest.raises(ValueError, match="finite terminal derivation"):
        CFG(bad_root, vocabulary_size=1)


def test_terminal_centralizes_local_vocabulary_integrity() -> None:
    terminal = Terminal(0)
    with pytest.raises(ValueError, match="duplicate"):
        terminal.set_vocabulary((1, 1))
    terminal.set_vocabulary((0, 1))
    with pytest.raises(RuntimeError, match="already has"):
        terminal.set_vocabulary((2,))


def test_cfg_requires_complete_vocabulary_coverage() -> None:
    terminal = Terminal(0)
    terminal.set_vocabulary((0, 2))
    root = Nonterminal("root")
    root.add_alternative(terminal)
    with pytest.raises(ValueError, match="vocabulary index 1"):
        CFG(root, vocabulary_size=3)


def test_exact_unold_request_generates_syntax_and_lexicon() -> None:
    parameters = UnoldCFGParameters(
        terminal_pair_rules=5,
        parenthesis_rules=4,
        iteration_rules=3,
        branch_rules=2,
        max_nonterminals=8,
        lexicon=LexiconParameters(
            category_count=5,
            vocabulary_size=10,
            tokens_per_category=4,
        ),
    )
    torch.manual_seed(12)
    grammar = generate_unold_cfg(parameters)

    assert _generated_family_counts(grammar) == (5, 4, 3, 2)
    assert grammar.rule_count == 14
    assert len(grammar.terminals) == parameters.lexicon.category_count
    assert all(
        len(terminal.vocabulary) == parameters.lexicon.tokens_per_category
        for terminal in grammar.terminals
    )
    assert set().union(
        *(set(terminal.vocabulary) for terminal in grammar.terminals)
    ) == set(range(parameters.lexicon.vocabulary_size))


def test_unold_construction_is_deterministic_under_the_callers_torch_seed() -> None:
    parameters = UnoldCFGParameters(
        4,
        3,
        2,
        2,
        7,
        LexiconParameters(4, 12, 4),
    )
    torch.manual_seed(91)
    left = generate_unold_cfg(parameters)
    torch.manual_seed(91)
    right = generate_unold_cfg(parameters)
    assert _cfg_signature(left) == _cfg_signature(right)


@pytest.mark.parametrize(
    "parameters",
    [
        UnoldCFGParameters(1, 0, 0, 0, 1, LexiconParameters(1, 1, 1)),
        UnoldCFGParameters(4, 0, 0, 1, 3, LexiconParameters(2, 3, 2)),
        UnoldCFGParameters(2, 3, 0, 0, 3, LexiconParameters(2, 3, 2)),
        UnoldCFGParameters(2, 0, 5, 0, 4, LexiconParameters(2, 3, 2)),
        UnoldCFGParameters(3, 2, 2, 2, 5, LexiconParameters(3, 5, 2)),
    ],
)
def test_unold_constructor_survives_repeated_legal_random_choices(
    parameters: UnoldCFGParameters,
) -> None:
    expected = (
        parameters.terminal_pair_rules,
        parameters.parenthesis_rules,
        parameters.iteration_rules,
        parameters.branch_rules,
    )
    for seed in range(50):
        torch.manual_seed(seed)
        grammar = generate_unold_cfg(parameters)
        assert _generated_family_counts(grammar) == expected
        assert len(grammar.terminals) == parameters.lexicon.category_count


def test_lexicon_parameters_enforce_coverage_capacity() -> None:
    with pytest.raises(ValueError, match="enough slots"):
        LexiconParameters(2, 5, 2)
    with pytest.raises(ValueError, match="cannot exceed"):
        LexiconParameters(2, 3, 4)
    with pytest.raises(ValueError, match="terminal positions"):
        UnoldCFGParameters(1, 0, 0, 0, 1, LexiconParameters(3, 3, 1))


def test_representative_large_vocabulary_is_fully_used() -> None:
    parameters = UnoldCFGParameters(
        terminal_pair_rules=100,
        parenthesis_rules=0,
        iteration_rules=0,
        branch_rules=0,
        max_nonterminals=1,
        lexicon=LexiconParameters(
            category_count=200,
            vocabulary_size=10_000,
            tokens_per_category=200,
        ),
    )
    torch.manual_seed(7)
    grammar = generate_unold_cfg(parameters)

    assert len(grammar.terminals) == 200
    assert all(len(terminal.vocabulary) == 200 for terminal in grammar.terminals)
    assert set().union(
        *(set(terminal.vocabulary) for terminal in grammar.terminals)
    ) == set(range(10_000))


def test_large_static_cfg_validation_is_iterative() -> None:
    terminal = Terminal(0)
    terminal.set_vocabulary((0,))
    nodes = [Nonterminal(f"N{index}") for index in range(2_000)]
    nodes[-1].add_alternative(terminal)
    for index in range(len(nodes) - 1):
        nodes[index].add_alternative(nodes[index + 1])
    grammar = CFG(nodes[0], vocabulary_size=1)
    assert len(grammar.nonterminals) == len(nodes)
