# Fixed CFG generation

`generate_unold_cfg` creates one fixed context-free grammar. The returned `CFG`
contains the generated syntax and the lexical realization of its abstract
terminal categories. It contains no production probabilities or runtime state.

```python
import torch

from persistent_online_learning.generator import (
    LexiconParameters,
    UnoldCFGParameters,
    generate_unold_cfg,
)

parameters = UnoldCFGParameters(
    terminal_pair_rules=100,
    parenthesis_rules=40,
    iteration_rules=20,
    branch_rules=20,
    max_nonterminals=64,
    lexicon=LexiconParameters(
        category_count=200,
        vocabulary_size=10_000,
        tokens_per_category=200,
    ),
)

torch.manual_seed(7)
grammar = generate_unold_cfg(parameters)
```

The four rule counts describe the paper's syntax-rule families. The configured
`category_count` replaces the paper's terminal-symbol maximum with an exact
project requirement: every abstract terminal category must occur in the
resulting CFG. `max_nonterminals` remains a maximum.

Each `Terminal` owns exactly `tokens_per_category` distinct vocabulary indices.
Across all terminal categories, every index from `0` through
`vocabulary_size - 1` occurs at least once. Categories may overlap. The lexical
request is feasible exactly when a category cannot request more unique entries
than the vocabulary contains and the combined category slots can cover the
vocabulary:

```text
tokens_per_category <= vocabulary_size
category_count * tokens_per_category >= vocabulary_size
```

The constructor distributes the first use of every vocabulary index as evenly
as possible across the terminal categories. It then fills each category's
remaining slots by independently sampling indices not already present in that
category. This guarantees complete vocabulary use without forbidding overlap or
systematically giving some categories all of the initially unique vocabulary.

The syntax must also contain enough terminal positions to use every configured
category. Construction rejects requests that cannot satisfy this requirement or
the paper's rule-capacity and hanging-component constraints.

The active PyTorch random stream owns all random choices. Seed it before calling
`generate_unold_cfg` when the exact grammar and lexicon must be reproducible.
