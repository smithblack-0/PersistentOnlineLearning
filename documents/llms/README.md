# LLM contributor notes

This directory contains short agent-specific routing and recurring failure warnings. It does not own the project's general research, engineering, writing, or checkpoint procedures.

## Load the relevant procedure

- Research, hypothesis, experiment, or interpretation work: [`../procedures/research_workflow.md`](../procedures/research_workflow.md)
- Substantial engineering, debugging, design, implementation, or review: [`../procedures/senior_engineering_workflow.md`](../procedures/senior_engineering_workflow.md)
- Substantial documentation: [`../procedures/technical_writing_workflow.md`](../procedures/technical_writing_workflow.md), consulting [`../procedures/technical_writing_standards.md`](../procedures/technical_writing_standards.md) as needed
- Checkpoint lifecycle or bucket synchronization: [`../procedures/checkpointing.md`](../procedures/checkpointing.md)

Read [`../../STATUS.md`](../../STATUS.md) for durable project position and only the task-relevant design, code, tests, and evidence. Status is not permission to choose an unfinished task when the user has not named one.

## Do not screw up these distinctions

- Persistent online learning is the objective. Unseen-language acquisition is an eventual test.
- A model's mathematical recurrence, optimized chunk-training form, and Hugging Face integration are separate concerns.
- Components are the model, generator, harness, tests, and documentation. Auditability, speed, progression, robustness, and refactorability are conditions on those components, not substitute components.
- Preliminary notes are exploratory. They do not become accepted designs because they are recent or detailed.
- Existing code establishes behavior, not automatic scientific authority.
- Reconstruct the proposed mechanism before replacing it with a nearby familiar one.
- Answer the current question before expanding into later experiments or production design.
- Procedures normally specify required outcomes, authority, and ownership. Do not reimplement framework functionality merely because a procedure describes one possible transaction pattern. Reuse prebuilt logic when it demonstrably satisfies the governing contract.

## Broad authority

When the user grants carte blanche or equivalent authority, ordinary scientific and architectural choices should be resolved provisionally by the contributor. Build a coherent iteration, test it, run the engineering, research, writing, and fresh regression passes, and report the choices and evidence afterward. Interrupt only for genuinely external, irreversible, or explicitly reserved decisions.

## Execution boundary

Do not run substantive model training in an agent's local container. Local validation may cover unit contracts, deterministic recurrence checks, serialization, checkpoint transactions, static compilation surfaces, and other small non-scientific probes. Prepare GPU training and scientific experiments for the user's Colab environment.

## Credentials

Never place Hugging Face tokens or other credentials in configs, dataclasses, checkpoints, manifests, telemetry, source files, or environment snapshots. Credentials enter runtime adapters only.

## Before committing

Run the complete relevant non-training test suite and then perform a source-order regression pass against the applicable procedures and standards. Fixes must be rechecked against the whole changed surface so correcting one defect does not regress another contract.
