# Status

The repository is in preliminary research and design. The exact random CFG construction unit exists; the complete synthetic-language generator and training experiment do not.

## Current objective

Determine whether a model can be architecturally and procedurally primed to continue learning online from an indefinitely long inference stream while using fixed-size recurrent state.

The model should be able to infer common rules from very long experience, preserve useful learned structure, revise it when later evidence contradicts it, and continue operating without a growing token cache.

## Current working direction

- Use maintained fully recurrent DeltaNet-family causal language models.
- Train on many independently resampled synthetic languages.
- Build each language around a random context-free grammar.
- Later assign production probabilities conditioned on epsilon-machine state.
- Advance the epsilon state only when a terminal category is emitted.
- Map terminal categories onto a normal vocabulary through a separate regenerated lexicon.

## Important distinction

Acquiring an unseen natural language after exposure to a corpus is an eventual high-value test. The immediate engineering work is constructing the synthetic-language training distribution in focused units.

## Implemented

- A composable reference epsilon-machine token process.
- Homogeneous FLA DeltaNet and Gated DeltaNet model exports.
- Exact random CFG construction using the productive-first and hanging-symbol method of Unold et al.
- Independent CFG reachability, productivity, rule-count, uniqueness, and symbol-limit validation.

## Not yet settled or implemented

- Total-rule, range, preset, and curriculum resolution around exact CFG requests.
- The lexical-category-to-vocabulary adapter.
- Production-probability assignment.
- The epsilon-machine transition and state-conditioned grammar table format.
- LIFO sequence derivation and sentence/reset lifecycle.
- The first integrated training curriculum and evaluation protocol.
- The practical sequence length needed for the first useful result.

## Documentation authority

- [`documents/preliminary/`](documents/preliminary/) contains exploratory notes and unresolved design work.
- [`documents/procedures/`](documents/procedures/) contains general research, engineering, writing, and lifecycle procedures.
- [`documents/llms/`](documents/llms/) contains concise LLM-specific routing and recurring failure warnings.
- [`documents/usage/`](documents/usage/) contains human-facing instructions for working systems.

## Next work

1. Review the exact CFG construction and its explicit random-choice policy.
2. Add complexity resolution incrementally around the exact constructor.
3. Build the lexical adapter as a separate unit.
4. Proceed to probabilities and derivation only after those boundaries survive review.
