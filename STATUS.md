# Status

The repository is in preliminary research and design. There is no accepted implementation yet, and the architecture and generator should still be treated as hypotheses rather than commitments.

## Current objective

Determine whether a model can be architecturally and procedurally primed to continue learning online from an indefinitely long inference stream while using fixed-size recurrent state.

The model should be able to infer common rules from very long experience, preserve useful learned structure, revise it when later evidence contradicts it, and continue operating without a growing token cache.

## Current working direction

- Build a completely recurrent custom model from existing recurrent sequence layers.
- Mix plain DeltaNet layers for persistent targeted associations with Gated DeltaNet layers for adaptive working memory.
- Train on many independently resampled procedural token generators so the model must learn how to identify a new process online.
- Begin with a generator that combines a hashed, factorized long-term controller with sparse recent-token filtering.

## Important distinction

Acquiring an unseen natural language after exposure to a corpus is an eventual high-value test. It is not the objective of the project. The objective is persistent online learning itself.

## Documentation authority

- [`documents/preliminary/`](documents/preliminary/) contains exploratory notes and unresolved design work.
- [`documents/procedures/`](documents/procedures/) contains general research, engineering, writing, and lifecycle procedures.
- [`documents/llms/`](documents/llms/) contains concise LLM-specific routing and recurring failure warnings.
- [`documents/usage/`](documents/usage/) is reserved for human-facing instructions for working systems.

## Not yet settled

- The ratio and placement of plain and gated DeltaNet layers.
- Whether another recurrent mixer such as Mamba is needed.
- The exact controller state and transition structure.
- The exact output collision and filtering mechanism.
- The first training curriculum and evaluation protocol.
- The practical sequence length needed for the first useful result.

## Next work

1. Turn the generator discussion into a precise minimal executable specification.
2. Inspect the current FLA DeltaNet interfaces and recurrent-state contracts.
3. Define the smallest experiment that distinguishes online process identification from ordinary local prediction.
4. Preserve alternatives as explicit comparisons rather than combining them prematurely.
