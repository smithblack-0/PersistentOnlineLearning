# Business-technical writing standards

These standards define how to judge substantial PersistentOnlineLearning writing: research notes, decisions, designs, experiment contracts, plans, reports, audits, READMEs, user guidance, reference documentation, docstrings, and technical explanations.

They are reference criteria, not an execution sequence and not a source of project authority. Use the [technical-writing workflow](technical_writing_workflow.md) to conduct the work; consult the relevant standards when defining, drafting, reviewing, or correcting an artifact.

## Governing objective

Business-technical writing is the controlled transfer of a decision-relevant scientific or technical model.

Its governing objective is **transmission efficiency**:

> Give the reader the most accurate and useful understanding possible for the least attention, inference, and rereading.

The relevant unit may be one artifact or a path through several artifacts. A participating document should be complete for its assigned job, not repeat the entire project until it can stand alone.

Do not equate rigorous writing with exhaustive detail. Select detail according to the work it performs for the reader.

## Artifact contract standard

Every substantial artifact must establish:

1. **Transmission boundary:** Is it standalone or one part of a larger documentation path?
2. **Artifact role:** What unique job does it perform? What may it assume, what must it establish, and where should it hand the reader next?
3. **Audience:** Who will use it, and how close are they to the subject?
4. **Reader purpose:** What decision, evaluation, implementation, or task brings the reader here?
5. **Starting model:** What can the reader safely be expected to know?
6. **Required result:** What should the reader understand, decide, or be able to do afterward?
7. **Authority:** Is the artifact exploratory, evidentiary, advisory, decisional, contractual, instructional, referential, or an audit record?
8. **Reading path:** Will the reader proceed in order, scan, compare options, follow steps, consult a section during work, or move among linked artifacts?

If these answers are unclear, prose is premature. Resolve the artifact role or supporting project model first.

## Standalone and participating artifacts

A standalone artifact owns the complete reader model required for its purpose. It may cite supporting evidence, but its central task does not depend on an undocumented path through other files.

A participating artifact owns one complete part of a larger reader path. It must not repeat neighboring documents merely to become locally self-contained. Participation does not excuse accidental incompleteness: prerequisites and handoffs must be explicit and navigable.

Before adding material, inspect:

- relevant entry points and neighboring documents;
- current accepted decisions and experiment contracts;
- public interfaces and behaviors the reader will encounter;
- existing authoritative homes for the same concepts;
- undocumented behavior and stale or competing explanations; and
- whether a discovered gap is a writing, research, API, implementation, ownership, or unresolved-decision problem.

Do not let the current file absorb every gap discovered during research.

## Five simultaneous quality standards

Review every substantial artifact against all five. Improving one does not justify silently damaging another.

### Useful

- Does the artifact answer the questions that bring the reader there?
- Can the reader make the intended decision or perform the intended task?
- Does it cover material consequences rather than merely describe nearby technology?
- Does it advance the larger reader path without rebuilding earlier material?

### Scientifically and technically credible

- Are mechanisms, constraints, causal claims, equations, and limitations correct?
- Can a knowledgeable reader see where the proposal works and where it stops?
- Are claims supported by evidence, precedent, derivation, or clearly identified hypothesis?
- Are easier alternative explanations addressed where they matter?
- Are unsupported aspirations expressed as intent, hypothesis, or candidate scope rather than current fact?

### Clear

- Does the reader receive the right mental model without repairing vague terms, hidden assumptions, or ambiguous references?
- Are definitions, examples, contrasts, and sequence supplied where they prevent material misunderstanding?
- Are objective, hypothesis, mechanism, implementation candidate, experiment, and eventual demonstration distinguishable?
- Are generator state, observer belief, model state, and training representation distinguished when relevant?

### Efficient

- Does every sentence add information, orientation, justified confidence, or a useful relationship?
- Is precision purchased only where its benefit exceeds its attention cost?
- Have repetition, premature detail, empty emphasis, and defensive qualification been removed?
- Does the artifact stop once it has fulfilled its assigned job?

### Coherent

- Does the artifact or documentation path build one usable model from beginning to end?
- Does each material fact have an appropriate primary home?
- Do headings, prose, lists, tables, diagrams, transitions, and links expose the actual hierarchy and relationships?
- Do research status and project authority remain consistent across artifacts?

## Scientific-substance standard

Include enough material for the intended reader to answer the relevant subset of these questions:

- What capability or question is under study?
- Why does it matter, and what becomes possible if it works?
- What is the proposed mechanism?
- Which prerequisites are already established and which remain speculative?
- What observation would increase or reduce confidence?
- What easier explanation could mimic the result?
- What does the current evidence establish?
- What can the mechanism not do, and why?
- What is true now, intended, accepted, hypothetical, candidate, or merely possible?
- What decision, risk, dependency, or next step follows?

These are selection questions, not a mandatory section template.

### Explain causality, not merely sequence

Where the mechanism matters:

1. establish the relevant lifecycle, state transition, or data flow;
2. identify what is shared and what differs;
3. explain why the difference produces the claimed behavior; and
4. derive the useful domain and hard boundary from the same mechanism.

An operational list that omits causality may be accurate but still fail to transmit the design.

## Precision standard

Precision is a cost-bearing resource.

Add precision when it:

- prevents a consequential misunderstanding;
- defines a capability, boundary, responsibility, lifecycle, control, or commitment;
- establishes mechanism or causality;
- supports a decision or implementation;
- distinguishes the proposal from a familiar alternative; or
- provides necessary evidence.

Defer or omit precision when it:

- answers a question the current discussion has not raised;
- introduces implementation detail before its governing concept;
- explains a standard consequence the reader can safely infer;
- narrows a deliberately broad and accurate statement without decision value;
- repeats a qualification already established by authority or structure; or
- competes with a more important point for attention.

Ask:

> What error, ambiguity, experiment interpretation, or decision would this added precision change, and is this where the reader needs it?

Define terms at the point of consequential ambiguity. Use one term consistently for one concept. Do not introduce vocabulary that saves the writer words while making the reader memorize unnecessary labels.

## Information-ownership standard

Assign every material point to the artifact or section whose purpose requires its full precision.

Elsewhere:

- omit it when the reader does not need it;
- name or summarize it briefly for orientation;
- link to its primary home when the reader may need the detail; or
- repeat it only when repetition saves more attention than it consumes.

An orphan is an accurate fact whose purpose at its location is unclear. It often means the fact belongs elsewhere, the document is missing the concept that gives it meaning, or the fact is unnecessary for this reader task.

During source-order review, ask why each paragraph appears at that exact point, what the reader can assume afterward, and whether another artifact is its better owner.

## Reader-path and layout standard

Organize around the reader's developing model, not discovery chronology, file order, implementation structure, or conversational history.

A common explanatory path is:

1. objective or unmet need;
2. specific contrast with existing approaches;
3. mechanism and causal explanation;
4. use and supported domain;
5. limitations and costs;
6. evidence or precedent; and
7. status, decision, or next step.

Change the order when the reader task requires it. A decision record begins with the decision. A troubleshooting guide begins with symptoms and recovery. A primer establishes one usable mental model before surveying terminology.

Use layout as syntax:

- paragraphs for reasoning and causal continuity;
- lists for parallel items, steps, criteria, or options;
- tables for exact repeated dimensions or mappings;
- equations for dependencies that prose would obscure;
- diagrams for topology, sequence, ownership, or state change when prose is materially slower; and
- descriptive links for movement among documentation layers.

Do not turn relational arguments into stacks of bullets merely to appear scannable.

## Authority and status standard

Use wording and artifact context to distinguish:

| Status | Meaning |
| --- | --- |
| Current fact | True of the present implementation or situation. |
| Evidence | An observed result with a stated basis. |
| Objective | Enduring capability the project seeks. |
| Accepted decision | A choice governing current work until reopened. |
| Hypothesis | A claim that evidence is intended to test. |
| Candidate | A possible mechanism, architecture, experiment, or future scope. |
| Preliminary note | Exploratory reasoning without governing authority. |

Do not weaken objectives into evasive possibilities merely because they are not release promises. Do not promote aspirations into commitments through confident grammar. Do not flatten a mixed authority state into “everything accepted” or “everything provisional.”

## Prose standard

Every sentence should perform at least one job:

- add a fact or claim;
- establish scope, status, or authority;
- explain mechanism, cause, consequence, or contrast;
- orient the reader within the argument;
- define a term or resolve an ambiguity;
- support a decision or next action; or
- derive a result not already stated.

Remove empty emphasis and repeated conclusions. Preserve transitions that explain why the next topic follows, show a change in abstraction or authority, or connect mechanism to value or limitation.

Use concrete, stable language. Keep actors visible when ownership matters. Keep terms consistent. Avoid vague pronouns, filler, stacked modifiers, and abstract nouns that hide the action.

Concision means reducing total reader effort, not mechanically shortening every sentence or removing explanations.

## Independent review passes

### Contract pass

- Does the artifact fulfill its reader purpose and authority?
- Is it complete for its assigned job without pretending to own the entire documentation system?
- Are prerequisites and handoffs explicit and usable?
- Has another document type contaminated it?

### Scientific and technical pass

- Are mechanisms, sequences, comparisons, controls, limits, and causal claims correct?
- Is the strongest wording supported?
- Are central facts specific enough for the intended decision or implementation?
- Are the main alternative explanations visible?

### Information-architecture pass

- Does the reader receive the governing model before dependent detail?
- Does each fact have one appropriate home?
- Are there repetitions, missing transitions, or orphans?
- Do cross-document moves preserve coverage and authority?

### Precision pass

- Which important claims remain vague?
- Which details are more precise than their purpose justifies?
- Did a clarification narrow an intentionally broad claim?
- Would a contrast, definition, example, equation, or reference transmit the point better?

### Status and register pass

- Are fact, evidence, objective, decision, hypothesis, candidate, and preliminary scope legible?
- Does modest persuasion stop after establishing relevance?
- Are limits candid without becoming generic defensive prose?

### Compression pass

- What can be removed without losing information, orientation, or justified confidence?
- Can layout or a link replace repetition?
- Does every remaining sentence improve the transmission?

### Fresh-reader pass

Enter through the path a technically literate reader would plausibly use. Ask:

> Where would I build the wrong model, lose the thread, doubt the writer, or have to reread?

Follow prerequisite and outward links far enough to verify the promised reader path.

## Common failure patterns

### The test replaces the objective

A striking eventual demonstration becomes the apparent purpose of the project.

**Correction:** restore the objective, then describe the demonstration as one test of the broader capability.

### The nearby literature replaces the proposal

A familiar mechanism is explained in detail while the user's actual mechanism disappears.

**Correction:** reconstruct the proposed dependency first, then state precisely what precedent supplies and what it does not.

### The document runs ahead

An exploratory question expands into a production architecture, full experiment program, or implementation plan before the current decision is calibrated.

**Correction:** answer the present question and preserve later work as unresolved rather than silently deciding it.

### Every document becomes standalone

Each artifact repeats orientation, definitions, mechanism, limits, and reference detail. Local completeness creates global duplication and inconsistency.

**Correction:** define each artifact's role, prerequisites, and handoffs. Keep it complete for that role.

### Rigor means maximum precision

Every claim accumulates qualifications and edge cases until the main point is one detail among many.

**Correction:** require each precision increase to name the consequential ambiguity or decision it serves.

### Concision means removing explanation

Transitions, examples, comparisons, and causal links disappear while isolated facts remain.

**Correction:** optimize total reader effort, not word count.

### Revision chases the latest criticism

The newest correction becomes the entire writing theory.

**Correction:** treat feedback as evidence against the whole model. Apply the correction, then recheck all objectives for regressions.

## Completion standard

Before delivery, answer from the finished artifact and reader path:

1. What exact part of the transmission does this artifact own?
2. What model should the intended reader now hold?
3. What decision or action can the reader now take?
4. Which scientific or technical details establish the central claims?
5. Which major limitation prevents overgeneralization?
6. Where is exact evidence, implementation, or reference detail located?
7. Are prerequisites, handoffs, and authorities real and navigable?
8. Is every material statement's status legible?
9. Which sentence contributes the least, and why does it remain?
10. Which detail received the most precision, and what consequence earns it?
11. Does the path contain any orphan, repetition, coverage gap, or unexplained transition?
12. Did the final revision improve the whole transmission rather than optimize one recent complaint?

The artifact is complete only when its scientific and technical content is correct, its local contract is fulfilled, its place in the larger reader path is coherent, and further compression would cost more understanding than it saves.
