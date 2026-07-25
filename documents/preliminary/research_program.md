# Research Program

## Central question

Can a model be primed to learn online indefinitely?

Operationally, this means a model that can continue consuming an arbitrarily long inference stream using fixed-size recurrent state, infer new rules from experience, preserve useful learned structure, revise it when contradicted, and use it later without ordinary weight updates or an ever-growing token cache.

This does not imply unlimited factual storage. A fixed-size state must compress experience. The research question is whether training can teach the model to discover and retain the useful sufficient structure of a new process rather than merely preserve recent tokens.

## Current hypothesis

A model may learn a general online-learning procedure if pretraining repeatedly forces it to identify newly sampled token-generating systems.

Each training episode would expose the model to a fresh process. The ordinary weights would learn how to infer unfamiliar rules; the recurrent state would contain what has been learned about the current process.

The current program therefore couples two problems:

1. A recurrent architecture capable of retaining and revising useful information over extremely long streams.
2. A procedural training distribution rich enough to teach online system identification rather than short-context imitation.

## Why procedural generators

A procedural generator can provide unlimited independently sampled processes with known hidden rules. This makes it possible to measure whether the model:

- improves as it observes more of one process;
- preserves information beyond its local context;
- distinguishes persistent rules from noise;
- revises an inferred rule after genuine change;
- transfers its learned inference procedure to generators outside the training family.

The generator should create complexity from comparatively few hidden parameters. If the hidden process itself has millions of arbitrary independent values, it may be impossible to identify from one stream and therefore unsuitable as a training target.

## Eventual high-value test

A decisive later test is whether a model trained without natural language can consume a large corpus in an unseen language during frozen-weight inference and then demonstrate any reliable understanding of that language.

This is an evaluation of general online acquisition. It is not the definition of the project and should not dictate the earliest generator or evaluation design before a basic signal exists.

## Current research posture

The project should begin with the simplest experiment that can reveal online process identification. Complexity should be added only when an observed limitation requires it.

The architecture, generator, and evaluation should remain separable enough that a failure identifies which part failed rather than collapsing several speculative mechanisms into one uninterpretable result.
