# Business-technical writing workflow

Use this workflow for substantial PersistentOnlineLearning writing: research notes, decisions, designs, experiment contracts, plans, reports, audits, READMEs, user guidance, reference documentation, docstrings, and technical explanations.

This file defines the execution loop. The [technical-writing standards](technical_writing_standards.md) define how to judge the work. Begin here; consult only the standards relevant to the current stage or defect.

The workflow is iterative. A review defect sends the work back to the stage that owns the problem. Do not force every correction through a local prose edit.

## Stage 1 — identify the task and governing authority

**Purpose:** establish the requested writing result and which artifacts may govern or be changed.

Produce:

- the requested artifact or reader result;
- the governing research decision, experiment contract, plan, implementation, evidence, or review material;
- the artifact that owns the requested change; and
- any authority or scope change requiring human consultation.

Proceed when the task and authority are clear. Inspect more repository state when the owner is uncertain. Ask the smallest necessary question when the user has not supplied the current objective or a required authority decision.

## Stage 2 — inspect evidence and the documentation system

**Purpose:** establish the real technical and scientific model and relevant reader path before designing prose.

Produce:

- verified facts and direct sources;
- explicit distinctions among evidence, inference, intent, commitment, hypothesis, candidate scope, and open questions;
- relevant entry points, neighboring artifacts, public surfaces, and existing authoritative homes; and
- discovered gaps classified as writing, research, product, API, implementation, ownership, or unresolved-decision work.

Do not make the target document absorb every discovered gap.

Return here whenever review finds an unsupported claim, incorrect mechanism, missing source, or unresolved scientific or technical contradiction.

## Stage 3 — define the artifact contract

**Purpose:** decide the artifact's unique transmission job before drafting it.

Produce:

- standalone or participating boundary;
- artifact role and authority;
- audience and assumed starting model;
- reader purpose and required post-reading result;
- prerequisites and handoffs; and
- expected reading path.

Continue when the artifact has one coherent job. Return to Stage 1 when the request belongs to another authority or artifact, and to Stage 2 when the contract depends on unresolved evidence. Consult the user before changing accepted authority, scientific interpretation, experiment meaning, or material scope.

## Stage 4 — build the content model and assign information homes

**Purpose:** design the reader's model and cross-document allocation before writing finished prose.

Produce:

- the central claim, mechanism, decision, or task flow;
- the facts, limits, evidence, hypotheses, controls, dependencies, and open questions the reader needs;
- the order in which the reader must learn them;
- one primary home for each material point;
- links or brief orientation for information owned elsewhere; and
- explicit follow-up work for gaps outside the current artifact.

Prefer causal relationships and meaningful contrasts over inventories of implementation detail.

Return here when review finds missing concepts, duplicated material, orphan facts, a wrong information home, or a broken cross-document path.

## Stage 5 — draft the complete reader path

**Purpose:** turn the content model into one coherent working draft.

Write the complete source-order argument or task flow, including necessary headings, transitions, links, tables, diagrams, examples, equations, and references. Do not optimize isolated passages while the document model remains unstable.

Remain here for defects limited to sequence, explanation, layout, terminology, sentence structure, or local precision. Return to an earlier stage when a prose defect exposes a deeper contract, evidence, or information-ownership problem.

## Stage 6 — run independent standards passes

Run these independent passes from the [standards](technical_writing_standards.md):

1. contract;
2. scientific and technical credibility;
3. information architecture;
4. precision;
5. status and register; and
6. compression.

Record material defects before editing and identify the stage that owns each one.

| Defect | Return to |
| --- | --- |
| Unsupported claim, wrong mechanism, missing evidence | Stage 2 |
| Wrong document type, authority, audience, prerequisite, or handoff | Stage 3 |
| Missing concept, duplication, orphan, or wrong primary home | Stage 4 |
| Poor order, explanation, layout, terminology, or prose | Stage 5 |
| Proposed change to accepted scientific meaning or authority | Human consultation, then the owning stage |

Do not resolve a Stage 2–4 defect by polishing Stage 5 prose around it. After a correction, rerun every materially affected pass. Treat new criticism as evidence against the complete model, not as a replacement writing theory.

## Stage 7 — run fresh-reader and delivery review

Enter through the path a technically literate reader would plausibly use. Verify that the reader can build the intended model without repair, that authority and status remain legible, and that links, examples, equations, tables, and handoffs work.

Ask:

> Where would a technically literate reader build the wrong model, lose the thread, doubt the writer, or have to reread?

When the answer exposes a defect, classify it and follow the corresponding backward edge. Before delivery, update every artifact that owns an accepted correction and record remaining work outside the current artifact.

## Completion condition

Writing is complete when:

- the requested artifact fulfills one clear contract;
- the scientific and technical model and strongest claims are supported;
- every material point has an appropriate primary home;
- the reader path is coherent, efficient, and navigable;
- status and authority are accurate;
- the fresh-reader pass succeeds; and
- all accepted corrections are made in the artifacts that own them.

A polished draft is not complete when the underlying evidence, artifact role, information architecture, scientific status, or authority remains wrong.
