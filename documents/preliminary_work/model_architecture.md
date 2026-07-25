# Model Architecture Notes

## Required properties

The model must be completely recurrent at inference:

- fixed-size state per layer;
- no growing attention KV cache;
- no requirement to retain the original token stream;
- ability to consume extremely long streams;
- targeted revision rather than only global decay;
- a practical parallel or chunkwise training form.

## Why the original inertial accumulator was displaced

The original idea accumulated contributions into persistent state. Its desired behavior was that repeated unsurprising evidence should have little effect while surprising or contradictory evidence should move memory strongly.

A cumulative accumulator risks becoming effectively immovable as evidence magnitude grows. Global decay restores plasticity by discarding old information merely because time passes, which defeats the intended permanent-memory role. Making the update depend nonlinearly on surprise relative to the previous state also introduces a difficult sequential training dependency.

DeltaNet supplies a cleaner existing mechanism: it corrects the value currently associated with an address rather than continually adding another copy of the same evidence.

## DeltaNet role

A DeltaNet layer maintains matrix-valued recurrent state and performs targeted delta-rule updates. Repeatedly setting the same association becomes approximately idempotent. Contradictory evidence can overwrite the addressed association without decaying unrelated directions.

The layer has:

- a recurrent decoding form with fixed-size state;
- optimized chunkwise or parallel training kernels;
- existing implementations in Flash Linear Attention (FLA).

## Plain and gated memory paths

The current architecture direction is a custom stack mixing two DeltaNet-family layers.

### Plain DeltaNet

Intended role:

- persistent targeted associations;
- no automatic time-based forgetting;
- retention until related evidence revises the addressed content.

### Gated DeltaNet

Intended role:

- adaptive working memory;
- deliberate learned forgetting or clearing;
- ordinary changing sequence dynamics where persistence is not always desirable.

The two states remain physically separate because they belong to different layers. Later layers may choose how to interpret or ignore retrieved information without requiring the persistent layer itself to erase it globally.

## Provisional stack

The exact arrangement is not settled. The current concept is a completely recurrent custom model built from existing FLA-native recurrent operators:

```text
embedding
-> gated DeltaNet blocks
-> periodic plain DeltaNet blocks
-> gated DeltaNet blocks
-> output head
```

Every block would retain its own recurrent state. Standard multi-head attention is not part of the current design.

Mamba or Mamba2 remains an available addition if the DeltaNet-family stack demonstrates a specific short- or medium-term modeling weakness. It is not currently required merely to make the architecture recurrent.

## Why a custom model is needed

FLA supplies the difficult sequence operators and optimized kernels, but a prebuilt configuration may not support an arbitrary mixture of plain and gated DeltaNet layers in the desired pattern.

The expected custom work is ordinary model architecture and state plumbing:

- construct the ordered layer stack;
- define configuration and serialization;
- carry per-layer recurrent states through generation;
- checkpoint and restore those states;
- verify chunk-training and recurrent-decoding equivalence;
- integrate the model with the training harness.

Custom CUDA or Triton kernels should not be required for the initial architecture.

## Main technical risks

### Fixed-state capacity

Constant memory is not unlimited memory. The model must learn useful compression, and similar addresses may interfere.

### Long-horizon optimization

Efficient kernels make the computation feasible but do not guarantee that useful gradients survive every effective horizon. This must be measured rather than presumed.

### Persistent-path misuse

The model may write transient details into plain DeltaNet state, causing interference, or avoid using the persistent path because gated layers solve shorter training examples more easily.

### State lifecycle

Reset, continuation, batching, padding, checkpointing, and resumed inference must have explicit contracts. Recurrent state errors can silently invalidate long-horizon experiments.

### Dependency stability

FLA, PyTorch, Triton, and CUDA versions should be pinned once an initial executable model exists. Recurrent and chunk modes must be regression-tested against each other.

## Relevant literature and implementations to inspect

- DeltaNet
- Gated DeltaNet and later variants
- Flash Linear Attention
- Mamba and Mamba2 as possible recurrent comparison mixers
- recurrent/chunk equivalence and state-passing patterns in existing FLA model classes
