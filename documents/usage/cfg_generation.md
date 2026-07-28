# Fixed CFG generation

`generate_unold_cfg` constructs one fixed context-free grammar whose terminal
nodes already contain their concrete vocabulary choices. The result is a static
language definition: it contains no probability tables, recurrent state, or
string-generation behavior.

```python
import torch

from persistent_online_learning.cfg import (
    CFGSpawnConfig,
    GrammarConfig,
    TerminalVocabularyConfig,
    generate_unold_cfg,
)

config = CFGSpawnConfig(
    grammar=GrammarConfig(
        terminal_pair_rules=100,
        parenthesis_rules=40,
        iteration_rules=20,
        branch_rules=20,
        max_nonterminals=64,
    ),
    terminal_vocabularies=TerminalVocabularyConfig(
        terminal_count=200,
        vocabulary_size=10_000,
        tokens_per_terminal=200,
    ),
)

torch.manual_seed(7)
grammar = generate_unold_cfg(config)

start = grammar.start
first_terminal = grammar.terminal_vocabularies[0]
concrete_choices = first_terminal.token_ids
```

The grammar is an object graph. Each `Nonterminal` owns ordered productions
containing `Nonterminal` or `TerminalVocabulary` nodes. Each
`TerminalVocabulary` directly owns a nonempty tuple of distinct concrete token
IDs.

Terminal vocabularies are not disjoint categories. Different terminal nodes may
overlap or contain identical token sets, and the same terminal node may be used
by many productions.

The terminal-vocabulary request requires:

```text
tokens_per_terminal <= vocabulary_size
terminal_count * tokens_per_terminal >= vocabulary_size
```

Construction guarantees every concrete ID in `0..vocabulary_size-1` appears in
at least one terminal and every terminal appears in at least one grammar
production.

`GrammarConfig` requests exact counts for the four rule families from Unold,
Kaczmarek, and Culer, *Iterative method of generating artificial context-free
grammars* (arXiv:1911.05801):

- terminal pair: `A -> a b`;
- parenthesis: `A -> a B b`;
- iteration: `A -> a B` or `A -> B a`;
- branch: `A -> B C`.

`max_nonterminals` is an upper bound rather than an exact count. Construction
chooses a feasible total within that bound before it builds topology.

Construction uses the active PyTorch random state as its external random
authority. It derives independent one-shot streams for concrete vocabulary
membership and syntax, so changing concrete vocabulary density does not silently
change syntax when the terminal count remains fixed.

The returned graph is sealed. Construction establishes reachability,
productivity, exact terminal use, vocabulary coverage, and production uniqueness
by the way each phase is built; it does not perform a second graph-wide audit at
the end. Independent contract tests re-check those guarantees across focused,
parameter-matrix, large-vocabulary, and deep-graph cases.
