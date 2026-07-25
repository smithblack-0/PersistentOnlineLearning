# Checkpoint lifecycle contract

## Purpose and ownership

The checkpoint subsystem owns the transition from speculative runtime state to durable recoverable state. It may delegate serialization, local atomic-save behavior, retention, or lifecycle hooks to PyTorch, Hugging Face, Lightning, or another framework when the delegated mechanism satisfies this contract.

The bucket adapter owns only remote synchronization. It does not decide which runtime state is authoritative, construct scientific payloads, or repair checkpoints.

The newest valid completed local checkpoint is authoritative during a running session. Hugging Face Storage Buckets provide durable backing storage through upload synchronization at the save boundary and download synchronization at the load boundary.

No model, optimizer, generator, telemetry, or experiment state becomes durable between checkpoint saves. Console output is informational and cannot reconstruct authoritative state.

## Required save behavior

A save implementation must provide these outcomes:

1. The committed payload contains every state required for scientific continuation, including model, optimizer, generator, trainer, experiment configuration, and buffered telemetry state when applicable.
2. A crash before local publication cannot expose a partially written checkpoint as valid.
3. Once published, a checkpoint is immutable or otherwise protected from in-place partial replacement.
4. Remote upload begins only after the local checkpoint has committed successfully.
5. An interrupted upload cannot supersede the previous completed remote checkpoint.
6. A convenience pointer such as `LATEST` cannot become authoritative before its target is complete and valid.
7. Retention or pruning cannot remove the previous recoverable checkpoint before the new checkpoint has crossed the required durability boundary.

A framework's pre-save, post-save, or checkpoint callbacks may own these behaviors directly. Custom save machinery is unnecessary when existing logic establishes the same guarantees.

## Required load behavior

A load implementation must provide these outcomes:

1. When remote backing is configured, download synchronization occurs at the load boundary before remote candidates are selected.
2. The loader chooses the newest valid completed checkpoint, not merely the path named by `LATEST`.
3. An incomplete or corrupt newest candidate falls back to the previous valid checkpoint.
4. A remote synchronization failure does not block use of an already valid local checkpoint.
5. Checkpoints are validated before their state becomes authoritative and are never repaired speculatively in place.
6. Configuration compatibility is judged by the scientific and continuation invariants that must match, not by incidental runtime settings such as local paths or compilation preferences.

## Reference transaction pattern

One acceptable implementation writes into a same-filesystem temporary directory, records a manifest and completion marker, atomically renames the directory into an immutable checkpoint name, updates a local pointer, and then synchronizes that completed directory to the bucket. Loading performs the inverse synchronization into temporary local storage, validates it, and atomically publishes it locally.

This pattern is illustrative rather than mandatory. A prebuilt checkpoint manager may use a different internal protocol when it demonstrably provides the required behavior above.

## Credentials

Credentials are external runtime dependencies. They must not appear in dataclass fields, JSON configuration, checkpoint payloads, telemetry, manifests, source files, or environment snapshots. The runtime supplies the Hugging Face token only to the bucket adapter or underlying client and never serializes that authenticated object.
