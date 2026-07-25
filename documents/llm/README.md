# LLM operating context

## Purpose

This directory contains durable working-process instructions for LLMs that contribute to PersistentOnlineLearning. It does not summarize the project, identify the current assignment, or replace accepted decisions, research notes, experiment contracts, plans, code, tests, or human review.

The objective is transferability: a fresh session should be able to discover the relevant authority, understand how work is conducted, and begin a named task without relying on undocumented conversational habits.

## Start by classifying the task

Before loading project material, identify the kind of work requested:

- research exploration or literature investigation;
- hypothesis or experiment design;
- implementation, refactoring, or debugging;
- design or technical review;
- planning or milestone review;
- substantial technical writing;
- repository or pull-request operation; or
- a narrow informational question.

Read only the process modules relevant to that work:

- [Research workflow](research_workflow.md) for scientific reasoning, generator or architecture exploration, experiment design, evidence review, and interpretation;
- [Senior engineering workflow](senior_engineering_workflow.md) for substantial engineering, debugging, design, implementation, and code review;
- [Business-technical writing workflow](technical_writing_workflow.md) for the executable writing loop used by research notes, decisions, designs, plans, reports, audits, READMEs, user guidance, reference documentation, and substantial technical explanations; and
- [Business-technical writing standards](technical_writing_standards.md) as the detailed evaluation reference consulted by the writing workflow when a stage or defect requires it.

For writing work, enter through the workflow. Do not preload the complete standards file merely because writing is involved; consult the sections relevant to the current workflow stage or defect.

Many tasks require more than one process. An experiment implementation, for example, must survive both the research workflow and the engineering workflow. A governing design document must also survive the writing workflow.

## Discover project authority

Do not reconstruct intent from filenames, class names, the newest commit, or the root README.

Use the repository's authority system:

1. Explicit accepted decisions and experiment contracts govern the questions they resolve unless a specific clause is reopened.
2. Accepted designs and active plans govern implementation structure and work sequence within those higher contracts.
3. [`STATUS.md`](../../STATUS.md) records the latest durable project position and points to governing material.
4. Research notes and evidence support conclusions; they do not independently assign work or become accepted decisions.
5. Material under [`documents/preliminary_work/`](../preliminary_work/) is exploratory unless its status is explicitly changed.
6. Code and tests establish current implementation behavior. Existing behavior is evidence, not automatic scientific or architectural authority.
7. Review records preserve audit history; they do not become technical decision sources merely because they are newer.

When two artifacts disagree, identify their roles and authority before editing one. Correct the artifact that owns the contradiction rather than patching the nearest file.

## Discover current work without inventing intent

`STATUS.md` is a shared, human-readable description of the repository's latest durable position. It may identify recently completed work, work currently under review, the next planned capability, blockers, and links to governing material.

`STATUS.md` is not an autonomous work queue. Repository state cannot establish what the user wants to do in the present conversation.

When the user names a task, use that task and load only the relevant status, authority, plan, implementation, tests, and evidence. When the user asks to continue earlier work without identifying it:

1. read `STATUS.md`;
2. inspect live branch, pull-request, issue, and CI state where relevant; and
3. ask the user which work should be resumed or whether the recorded continuation remains current.

Do not silently choose the newest plan, experiment, pull request, or unfinished item.

## Maintain `STATUS.md` as shared documentation

Humans and LLMs maintain `STATUS.md` under the same writing standards. It must read like normal project documentation, not like a machine scratchpad or a message to a future model.

Update it when durable project position changes materially, such as:

- a central hypothesis is accepted, revised, rejected, or reopened;
- an experiment changes the evidence base materially;
- a governing design or experiment contract is accepted or replaced;
- the principal work under review changes;
- a durable blocker appears or is resolved; or
- the next planned capability changes.

Do not update it for every commit, test run, conversation, or speculative idea. Live GitHub details should be linked or queried rather than copied exhaustively.

Write status in project terms. Avoid conversational attribution, private session history, autonomous-resume instructions, and phrases such as “the human said,” “the LLM should remember,” or “continue where the previous model stopped.”

## Calibrate before expanding scope

When the user is thinking through an idea, answer the question currently under discussion before designing later stages. Do not assume a production target, success criterion, output modality, or implementation plan that the user has not established.

Distinguish clearly among:

- reconstructing the user's mechanism;
- evaluating it;
- identifying unresolved issues;
- proposing alternatives; and
- beginning design or implementation.

Do not turn conceptual discussion into an unsolicited project plan. Precision should be added where it changes the decision or prevents a consequential misunderstanding, not merely because more detail is available.

## Change control

Make ordinary in-scope improvements autonomously and report them. Consult the user before changing:

- the central scientific objective or interpretation;
- an accepted hypothesis, experiment contract, control, or success criterion;
- the intended meaning of persistent online learning;
- persistent-state authority, reset, recovery, or continuation semantics;
- a major architecture or generator commitment;
- CI or workflow files;
- repository permissions or release policy; or
- pull-request merge state.

Accepted decisions remain governing by default. Concrete contradictory evidence may justify reopening a specific clause; it does not make every surrounding decision provisional.

Do not merge a pull request unless the user explicitly requests the merge.

## Before substantial action

A fresh contributor should be able to state:

- the requested task and expected result;
- the governing authority and owning artifact;
- whether the work is exploratory, evaluative, decisional, or implementational;
- the relevant model, generator, experiment, or implementation boundary;
- the evidence needed before choosing a design;
- which changes are ordinary and which require consultation; and
- how completion will be judged.

If one of these cannot be determined from the request and repository, gather the missing evidence or ask the smallest necessary question before committing to a direction.
