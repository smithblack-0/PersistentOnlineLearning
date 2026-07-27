# Constructing random context-free grammars

The CFG package constructs ordinary grammar topology. It does not assign
production probabilities, condition rules on epsilon-machine state, map terminal
categories to vocabulary elements, or derive token sequences.

The first constructor follows the productive-first method of Unold, Kaczmarek,
and Culer, *Iterative method of generating artificial context-free grammars*:
<https://arxiv.org/abs/1911.05801>.

## Exact construction request

```python
import torch

from persistent_online_learning.generator import (
    UnoldCFGSpecification,
    generate_unold_cfg,
)

torch.manual_seed(0)
specification = UnoldCFGSpecification(
    parenthesis_without_nonterminal=8,
    parenthesis_with_nonterminal=5,
    iteration_rules=6,
    branch_rules=4,
    max_terminals=12,
    max_nonterminals=10,
)
grammar = generate_unold_cfg(specification)
```

The four rule counts are exact. `max_terminals` and `max_nonterminals` are
maxima, matching the paper; a generated grammar may use fewer symbols.
Terminals are abstract lexical-category IDs rather than final vocabulary IDs.

Construction uses the active PyTorch random stream. The caller may seed
immediately before construction when it needs a reproducible grammar. Creating a
new grammar means calling the constructor again rather than mutating an existing
grammar.

## Returned structure

`CFG.start` is the most recently created nonterminal and
`CFG.productions` preserves construction order. Productions belong to exactly one
of the supported families:

- `A -> a b`
- `A -> a B b`
- `A -> a B` or `A -> B a`
- `A -> B C`

`validate_cfg()` independently checks that every rule and symbol is reachable and
productive. `validate_unold_cfg()` additionally checks the requested rule counts
and symbol maxima. The constructor runs the latter before returning.

## Deliberate current boundary

The exact specification is the constructor input, not a finished project-wide
configuration schema. Later work may resolve total-rule targets, ranges, or
curriculum settings into this exact request. The lexical adapter and the
state-conditioned probability system are separate downstream units.
