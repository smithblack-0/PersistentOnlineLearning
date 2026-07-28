# Fixed CFG generation

`generate_unold_cfg` constructs one fixed context-free grammar whose terminal
nodes already contain their concrete vocabulary choices. The result contains no
probability tables or string-generation state.

```python
import torch

from persistent_online_learning.generator import (
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

The terminal-vocabulary request guarantees:

```text
tokens_per_terminal <= vocabulary_size
terminal_count * tokens_per_terminal >= vocabulary_size
```

Every concrete ID in `0..vocabulary_size-1` appears in at least one terminal, and
every terminal appears in at least one grammar production.

`GrammarConfig` uses the four rule families from Unold, Kaczmarek, and Culer,
*Iterative method of generating artificial context-free grammars*
(arXiv:1911.05801):

- terminal pair: `A -> a b`;
- parenthesis: `A -> a B b`;
- iteration: `A -> a B` or `A -> B a`;
- branch: `A -> B C`.

Construction uses the active PyTorch random state as its authority. It derives
independent one-shot streams for concrete vocabulary assignment and grammar
syntax, so changing concrete vocabulary density does not silently change syntax
when the terminal count remains fixed.

The returned nodes are sealed. Reachability, productivity, rule-capacity, and
vocabulary-coverage checks occur once during construction, not during later
string generation.
