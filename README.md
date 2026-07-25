# PersistentOnlineLearning

Research toward models that are architecturally and procedurally primed to keep learning online from an indefinitely long inference stream.

The project is not defined by a single transfer test or model architecture. The central question is whether a model can continuously infer, retain, revise, and use new rules without ordinary weight updates and without retaining an ever-growing token history.

## Current direction

The present working direction combines:

- a fully recurrent model assembled from existing recurrent sequence layers;
- persistent associative memory using plain DeltaNet layers;
- adaptive working memory using Gated DeltaNet layers;
- procedural pretraining on repeatedly resampled token-generating systems;
- a compact generator built around hashed, factorized controller state and sparse local transition filtering.

Acquiring an unseen natural language during frozen-weight inference is a high-value eventual test of the capability, not the definition of the project.

## Repository stage

This repository is currently in preliminary research and design. No implementation should be treated as settled. Early material lives under [`documents/preliminary_work/`](documents/preliminary_work/).

See [`STATUS.md`](STATUS.md) for the current human-readable project status.
