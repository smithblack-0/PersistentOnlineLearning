# Research workflow

This workflow governs scientific exploration, literature investigation, hypothesis development, experiment design, evidence review, and interpretation for PersistentOnlineLearning.

It is not a project roadmap and does not assign work by itself. Enter with the user's named question and stop at the level of commitment the user requested.

## Core imperative

Convert an interesting idea into a discriminating scientific question without silently changing what the idea means.

The first obligation is accurate reconstruction. Before evaluating a proposal, be able to state:

- the intended mechanism;
- the capability it is supposed to produce;
- the prerequisite assumptions;
- the current uncertainty; and
- what the user is asking to decide now.

Do not substitute a nearby familiar research problem merely because it is easier to analyze.

## Stage 1 — classify the current research task

Identify whether the user is asking for:

- conceptual reconstruction;
- field or literature orientation;
- mechanism evaluation;
- comparison among directions;
- an initial experiment;
- a complete experimental program;
- implementation planning;
- interpretation of results; or
- a decision to continue, defer, or reject a direction.

Do not answer a later task merely because it may eventually follow. A discussion of whether a generator idea is promising is not permission to design the full natural-language transfer protocol.

## Stage 2 — define the claim under consideration

State the narrowest claim that preserves the proposal's value.

Separate:

- **objective:** the enduring capability the project seeks;
- **hypothesis:** the proposed causal explanation or mechanism;
- **test:** an observation that would increase or reduce confidence;
- **implementation candidate:** one way to instantiate the hypothesis; and
- **eventual demonstration:** a high-value result that may be far beyond the first test.

For this repository, unseen-language acquisition is an eventual test of persistent online learning, not the project objective itself.

## Stage 3 — inspect precedent and prerequisites

Search for work that establishes prerequisite components, close mechanisms, or known failure modes. Prefer primary sources and current implementations for technical claims.

For each precedent, ask:

- Which part of the proposed mechanism does it actually establish?
- Which conditions differ materially?
- Does it remove a blocker, supply an implementation, or merely show plausibility?
- Does it occupy the proposed contribution or only a neighboring one?

Do not report a list of papers as a conclusion. Synthesize what the field has made possible and what remains unresolved.

## Stage 4 — reduce the mechanism

Represent the proposal at the simplest level that exposes its causal structure. Use equations, state transitions, or compact pseudocode when they clarify ownership and dependence.

Clearly distinguish:

- required mechanism from optional parameterization;
- generator state from an observer's belief about that state;
- mathematical recurrence from an efficient training implementation;
- state capacity from temporal reach;
- output ambiguity from hidden-state complexity; and
- a baseline implementation from later refinements.

When a mechanism appears difficult, identify the exact dependency causing the difficulty rather than replacing it with an easier mechanism that no longer provides the intended behavior.

## Stage 5 — identify the decisive uncertainty

Find the uncertainty that currently blocks a decision.

Examples include:

- whether a state update has a practical parallel training form;
- whether a generator produces learnable long-range structure rather than noise;
- whether output filtering preserves hidden-state information;
- whether fixed recurrent state has enough capacity before interference dominates; or
- whether an observed transfer metric requires genuine structural learning.

Do not prematurely solve every future problem. Preserve unresolved questions that do not affect the current decision.

## Stage 6 — design the smallest discriminating experiment when requested

Use a simple experiment, initial experiment, proof of concept, or diagnostic experiment. Do not describe the first study as a “bounded experiment.”

A useful initial experiment should:

- isolate one important uncertainty;
- have a meaningful positive and negative interpretation;
- include controls that distinguish the proposed mechanism from easier explanations;
- avoid requiring later-stage production machinery;
- expose useful intermediate measurements; and
- fail informatively.

The experiment is not complete merely because it runs. Before accepting it, determine whether a positive result would support the intended claim and whether a negative result would identify a plausible failure source.

## Stage 7 — establish controls and observability

Controls should attack the strongest alternative explanations, not merely satisfy convention.

For persistent online learning, likely distinctions include:

- carried recurrent state versus reset state;
- long-history structure versus locally matched or shuffled streams;
- online process identification versus memorized generator conventions;
- persistent memory versus ordinary short-context prediction;
- true rule revision versus global forgetting; and
- semantic or structural transfer versus token-frequency adaptation.

Measure the mechanism's intermediate behavior where possible. A final scalar alone may not distinguish memory failure, generator failure, optimization failure, or evaluation failure.

## Stage 8 — interpret adversarially

Before claiming support, ask:

- What easier mechanism could produce this result?
- Did the evaluation leak stable encoding conventions?
- Did a local model solve the task without using persistent state?
- Did the model exploit token frequencies, repeated strings, or output-set imbalance?
- Did selection, filtering, or preprocessing manufacture the effect?
- Does the result survive new seeds, new process instances, and held-out generator families?

State what the evidence changes, not merely whether a metric improved.

## Research-direction review

When comparing projects, judge independently:

- field readiness;
- actual unresolved opening;
- prerequisite availability;
- fastest discriminating experiment;
- failure informativeness;
- expected compute and engineering burden;
- scientific upside;
- novelty and defensibility; and
- opportunity cost.

Give a real ranking. Do not treat every proposal as promising or turn every direction into a full project plan.

## Communication discipline

During exploratory discussion:

- reconstruct before correcting;
- distinguish a misunderstanding from a disagreement;
- answer the current question before running ahead;
- explain why a proposed mechanism helps at the scale the project requires;
- state when an alleged solution avoids rather than solves the key dependency;
- use concrete examples before field terminology when teaching a new model family; and
- stop once the requested decision has been supported.

When challenged, revisit the mechanism and evidence rather than defending the previous answer.

## Completion condition

Research work at the current level is complete when:

- the user's intended question has been preserved;
- the claim, mechanism, and prerequisites are explicit;
- precedent is synthesized rather than inventoried;
- the decisive uncertainty is identified;
- any proposed experiment tests that uncertainty with meaningful controls;
- likely alternative explanations are exposed; and
- conclusions do not exceed the evidence or the requested scope.
