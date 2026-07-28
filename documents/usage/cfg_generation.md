# Fixed CFG generation

`generate_unold_cfg` creates one fixed lexicalized context-free language. The
returned `LexicalizedCFG` contains two passive runtime artifacts:

- `language.grammar`: the sealed syntax graph;
- `language.lexicon`: the concrete token IDs assigned to each abstract terminal.

It contains no production probabilities or runtime generation state.

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
language = generate_unold_cfg(parameters)

start = language.grammar.start
first_terminal = language.grammar.terminals[0]
first_entry = next(
    entry for entry in language.lexicon.entries if entry.terminal is first_terminal
)
```

Each `Nonterminal` owns its ordered production alternatives. A production is a
tuple of `Terminal` and `Nonterminal` nodes, so recursion and shared subgraphs use
ordinary object references. Nodes accept local construction changes until the
constructor seals them. The finished `CFG` only stores the start node and the
selected terminal and nonterminal nodes; it does not perform reachability,
productivity, validation, or vocabulary work during string generation.

Each `Terminal` is an abstract lexical category identified by a readable name such
as `T0`. Its concrete realizations live in a separate `LexiconEntry`, not on the
terminal node. Every entry owns exactly `tokens_per_category` distinct token IDs.
Across all entries, every ID from `0` through `vocabulary_size - 1` occurs at least
once. Entries may overlap.

The lexical request is feasible when:

```text
tokens_per_category <= vocabulary_size
category_count * tokens_per_category >= vocabulary_size
```

The syntax must also contain enough terminal positions to use every configured
category. During construction, the Unold builder performs the paper-specific
feasibility, connectivity, reachability, productivity, and vocabulary-coverage
checks. Only after those checks pass does it seal the nodes and publish the passive
`LexicalizedCFG`.

The constructor distributes the first use of every vocabulary index as evenly as
possible across terminal categories, then fills each category's remaining slots by
independently sampling IDs not already present in that entry.

The active PyTorch random stream owns all random choices. Seed it before calling
`generate_unold_cfg` when the exact syntax and lexicon must be reproducible.
