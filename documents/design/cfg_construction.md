# Fixed CFG construction design

## Contract

Given one syntax request and one terminal-vocabulary request, construct a sealed
CFG whose terminal nodes already contain their concrete token choices.

The completed graph is directly traversable:

```text
Nonterminal
  -> production
     -> Nonterminal or TerminalVocabulary
        -> concrete token IDs
```

Probability tables, epsilon state, string generation, serialization, and optimized
runtime lookup are outside this subsystem.

## Package ownership

CFGs are static language definitions, not executable token generators. The source
tree therefore separates them from `persistent_online_learning.generator`:

```text
persistent_online_learning/
  cfg/
    grammar.py
    config.py
    construction/
      terminal_vocabularies.py
      planning.py
      rules.py
      topology.py
      terminal_assignment.py
      generate.py

  generator/
    ... executable token processes ...
```

`cfg` owns what the fixed language is. `cfg.construction` owns the one-shot process
that creates such a language. The construction package is not organized under an
`unold` namespace because Unold et al. are the source for part of the algorithm,
not the architectural owner of every construction step.

## Static graph

`cfg/grammar.py` owns the finished-language types:

- `Node`: shared name and open-to-sealed assembly lifecycle;
- `Nonterminal`: one LHS and its ordered production alternatives;
- `TerminalVocabulary`: one grammar terminal and its concrete token alternatives;
- `CFG`: passive publication container for the start and declared nodes.

`TerminalVocabulary` is intrinsic to the grammar. Its token IDs are not stored in
a separate lexicon. Different terminal vocabularies may overlap or contain
identical token sets; their identity is the grammar choice point, not an exclusive
vocabulary category.

Only `Nonterminal` needs assembly mutation because recursive edges are not all
known when nodes are created. Terminal vocabularies are complete at creation.

## Configuration ownership

`cfg/config.py` contains public requests rather than transient construction state.

`GrammarConfig` owns:

- exact `A -> a b` rule count;
- exact `A -> a B b` rule count;
- exact combined `A -> a B` / `A -> B a` rule count;
- exact `A -> B C` rule count;
- maximum, not exact, nonterminal count.

`TerminalVocabularyConfig` owns:

- number of terminal nodes;
- concrete vocabulary size;
- distinct concrete IDs per terminal;
- feasibility of global concrete-vocabulary coverage.

`CFGSpawnConfig` composes those independently meaningful requests and rejects
request-level incompatibilities such as insufficient terminal positions or symbol
capacity. It does not choose an actual construction plan.

## Construction phases and provenance

### 1. Terminal vocabularies

Owner: `construction/terminal_vocabularies.py`.

This phase is project-specific. Unold assumes an existing terminal alphabet; this
project first creates that alphabet as complete `TerminalVocabulary` nodes. Every
concrete vocabulary ID is assigned at least once before overlap fills remaining
per-terminal capacity.

Output: complete terminal nodes, with no syntax yet.

### 2. Construction planning

Owner: `construction/planning.py`.

Unold treats the number of nonterminals as a maximum and chooses symbol creation
while rules are inserted. This project deliberately enumerates feasible total
nonterminal counts and compatible initial-foundation sizes, then samples one
`ConstructionPlan` before topology begins.

The plan exists because each later new LHS must contain the previous root and
therefore consumes one RHS edge. Knowing the exact number of future node creations
makes that reserved-edge budget explicit instead of asking after each arbitrary
candidate whether future rules might still repair the graph.

Output: transient `ConstructionPlan`; it is never part of the published CFG.

### 3. Rule semantics

Owner: `construction/rules.py`.

The four rule shapes come directly from Unold et al. `RuleDraft` is a transient
phase-boundary representation: nonterminal topology is complete, while terminal
positions remain unlabeled. It exists so connectivity and finite terminal-label
capacity can be reasoned about separately without creating a second runtime graph.

### 4. Productive topology

Owner: `construction/topology.py`.

This is the main Unold-derived phase. It preserves the paper's productive-first
structure:

- terminal-only rules establish productive nonterminals first;
- every later RHS nonterminal is already productive;
- a newly created LHS contains the previous root;
- the newest created LHS becomes the new root;
- final productions must have enough remaining terminal labelings to stay unique.

Project-specific adaptations are documented in the module:

- the planner has already fixed the total nonterminal count;
- existing-LHS extensions use only the currently reachable component;
- productive but disconnected foundation roots are tracked as `hanging` and must
  be connected before unreserved RHS-edge capacity is exhausted.

Output: `SyntaxTopology`, containing final nonterminal nodes and unlabeled rule
drafts.

### 5. Terminal assignment

Owner: `construction/terminal_assignment.py`.

Unold requires unique rules. This project additionally requires every supplied
terminal node to be used. The phase assigns terminal nodes to the open positions
of the topology drafts while preserving both constraints, then materializes
ordinary productions directly on their owning nonterminals.

Output: the finished graph structure; the draft layer is discarded.

### 6. Publication

Owner: `construction/generate.py`.

The orchestrator derives independent syntax and concrete-vocabulary random streams
from the caller's active PyTorch RNG, sequences the phase owners, seals all nodes,
and returns `CFG`. It does not duplicate the calculations owned by those phases.

There is deliberately no final production-time graph audit.

## Verification authority

Construction code is responsible for establishing its invariants. Tests are
responsible for distrusting it.

`tests/cfg_assertions.py` therefore contains an independent graph oracle that is
never imported by production code. It re-checks:

- declared references and reachability from the start node;
- finite productivity of every nonterminal;
- exact use of every terminal node;
- complete concrete-vocabulary coverage;
- unique productions;
- exact requested rule-family counts;
- sealing and configured symbol bounds.

The oracle is run against representative requests and the accepted small-parameter
matrix. Separate tests cover overlapping/identical terminal vocabularies,
caller-seeded reproducibility, syntax independence from vocabulary density, large
10,000-token coverage, and iterative productivity analysis on a 2,000-node graph.

This separation is intentional: if a construction invariant regresses, tests must
fail rather than production spending FLOPs re-proving the invariant on every build.
