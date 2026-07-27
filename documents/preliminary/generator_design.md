# Synthetic language generator design

Status: current working design. Only the exact random CFG construction described
below is implemented.

## Purpose

The training distribution should repeatedly present the model with a newly
sampled language so ordinary weights learn an online language-acquisition
procedure while recurrent state learns the current language. The eventual
language generator combines recursive grammar with hidden evolving context.

The intended runtime process is:

1. maintain a LIFO derivation stack;
2. choose a production using probabilities conditioned on the current state of
   an epsilon machine;
3. leave the epsilon state unchanged during nonterminal expansion;
4. advance the epsilon state only after a terminal is emitted; and
5. regenerate grammar, probability tables, epsilon transitions, and lexical
   realization for a new training language.

The grammar topology and its later probability system are separate artifacts.

## Current implemented unit: random CFG topology

The first unit constructs an ordinary context-free grammar using the method in
Unold, Kaczmarek, and Culer, *Iterative method of generating artificial
context-free grammars*: <https://arxiv.org/abs/1911.05801>.

It preserves the paper's four rule families, productive-first ordering,
hanging-symbol connection requirement, unique-rule requirement, and final use
of the most recently created nonterminal as the start symbol.

The exact constructor input contains:

- parenthesis rules without a nonterminal;
- parenthesis rules with a nonterminal;
- iteration rules;
- branch rules;
- a maximum terminal count; and
- a maximum nonterminal count.

The constructor owns neither ranges nor curriculum policy. A later configuration
resolver may sample a feasible exact request without changing this contract.

### Random choices left open by the paper

The paper gives construction invariants rather than complete pseudocode. The
baseline implementation therefore makes the remaining choices explicitly:

- after terminal-only rules, choose uniformly among legal remaining rule
  families with quota left;
- sample existing symbol identities together with one possible new-symbol
  action when creation remains legal;
- reserve forced hanging-symbol placements before sampling free right-hand-side
  positions;
- sample existing left-hand sides from the connected component, while hanging
  symbols remain reserved as components to attach on a right-hand side; and
- use the active PyTorch random stream.

Restricting existing left-hand sides to the connected component is the minimal
constructive interpretation of the paper's hanging-symbol capacity guarantee.
It prevents an apparent attachment from merely joining two still-disconnected
components without reducing the number of roots that later rules must attach.
This choice is intentionally local to the constructor and may be compared with
a fuller disconnected-component policy if evidence makes that worthwhile.

## Downstream lexical adapter

CFG terminals are lexical categories, not final words or stems. A later lexical
adapter will map those categories onto a configured vocabulary. Its concrete
configuration should be added only when that unit is implemented, but it must be
able to support:

- many vocabulary elements per category;
- overlap between categories;
- coverage of the complete vocabulary; and
- independently regenerated mappings for new synthetic languages.

## Downstream state-conditioned PCFG

A later unit will assign a production distribution for each pair of epsilon state
and expandable nonterminal. Grammar expansion reads the distribution for the
current state, while only terminal emission advances the epsilon machine.
Probabilities and epsilon transitions must not be folded into the CFG
constructor.

## Curriculum consequence

More hidden states require more evidence from one language before the process is
identifiable. Curriculum complexity therefore includes both structural
complexity and the number of tokens supplied from each generated language. Long
training episodes may eventually approach dataset-scale lengths.

## Displaced initial direction

The earlier hashed, factorized controller with sparse local filtering remains a
possible comparison generator. It is no longer the leading synthetic-language
construction because the state-conditioned CFG gives a more direct proxy for
online grammar and hidden-context acquisition.

## Next focused units

1. Add total-rule and range resolution around the exact CFG request after its
   required configuration behavior is concrete.
2. Build the lexical-category adapter.
3. Assign ordinary production probabilities and implement LIFO derivation.
4. Add epsilon-state-conditioned production tables and terminal-triggered state
   transitions.
