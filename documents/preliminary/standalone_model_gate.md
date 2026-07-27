# Standalone recurrent model success gate

Status: accepted gate for the first model milestone.

The milestone passes when the repository exposes a standalone Hugging Face causal language model using either a homogeneous DeltaNet stack or a homogeneous Gated DeltaNet stack and all of the following are true.

## Complete model behavior

- The model is constructible from a Hugging Face configuration.
- Ordinary forward execution, causal-language-model loss, weight save/load, and Hugging Face `generate()` work.
- The model does not depend on the procedural generator, data pipeline, training harness, Lightning, Ray, telemetry, or experiment checkpoint policy.

## Recurrent correctness

- Full or chunked execution and tokenwise recurrent execution implement the same model within an appropriate numerical tolerance.
- Recurrent state is explicit through the Hugging Face cache interface and is not retained secretly between unrelated calls.
- Independent batch elements remain isolated.
- Padding does not update or contaminate valid sequence state.
- State dtype, device, reset, continuation, and cache reordering behavior are intentional and testable.

## Constant-shape inference

- `generate()` uses the recurrent inference form.
- The per-layer cache has shapes determined by model and batch configuration rather than generated sequence length.
- Generated tokens update the existing recurrent cache instead of accumulating token-level key/value history or replaying the full prefix.

## Architecture and reuse

- The implementation genuinely uses the maintained DeltaNet or Gated DeltaNet mechanisms rather than a nearby project-owned recurrence.
- Existing library configurations, model classes, kernels, cache objects, generation plumbing, normalization, MLPs, and checkpoint behavior are reused when they satisfy the gate.
- Project code remains parsable, configurable, compositional, and no more abstract than its actual responsibilities require.

## Accepted initial variants

The initial milestone accepts two separate model flavors:

- FLA `DeltaNetForCausalLM` with a homogeneous DeltaNet layer stack;
- FLA `GatedDeltaNetForCausalLM` with a homogeneous Gated DeltaNet layer stack.

A mixed DeltaNet/Gated DeltaNet stack is downstream work and is not required for this gate.
