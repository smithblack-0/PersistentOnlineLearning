# Checkpoint procedure

## Authority model

The newest valid completed local checkpoint is authoritative during a running session. Hugging Face Storage Buckets provide durable backing storage through explicit save-boundary upload and load-boundary download synchronization.

No model, optimizer, generator, telemetry, or experiment state becomes durable between checkpoint saves. Console output is informational and cannot be used to reconstruct authoritative state.

## Save transaction

1. Write the complete checkpoint into a temporary directory on the same filesystem as the final checkpoint root.
2. Write model, optimizer, generator, trainer, experiment configuration, and buffered telemetry state.
3. Generate a manifest containing byte sizes and checksums.
4. Write `COMPLETED` last inside the temporary directory.
5. Flush the files and directory where the platform permits it.
6. Atomically rename the temporary directory to its immutable `checkpoint-XXXXXXXX` name.
7. Atomically update the local `LATEST` pointer.
8. Synchronize the completed local checkpoint to the configured bucket in upload mode.
9. Synchronize the remote `LATEST` pointer only after the checkpoint upload succeeds.
10. Prune excess local checkpoints only after the remote upload boundary.

An interrupted local save leaves only a temporary directory, which loaders ignore. An interrupted upload leaves the previous completed remote checkpoint authoritative.

## Load transaction

1. If bucket synchronization is configured, synchronize remote checkpoint metadata in download mode at the load boundary.
2. Consider candidates from newest to oldest.
3. Download a missing candidate into a temporary local directory.
4. Validate `COMPLETED`, manifest structure, required files, sizes, and checksums.
5. Atomically move the valid directory into the local checkpoint root.
6. Load the newest valid local checkpoint.
7. If remote synchronization fails, use an already valid local checkpoint when one exists.

The loader never trusts `LATEST` without validating the referenced checkpoint. A corrupt or incomplete newest checkpoint falls back to the previous valid checkpoint rather than being repaired in place.

## Credentials

Credentials are runtime dependencies. They must not appear in dataclass fields, JSON configuration, checkpoint payloads, telemetry, manifests, source files, or environment snapshots. The runtime reads the Hugging Face token only when constructing the bucket adapter and never serializes the adapter itself.
