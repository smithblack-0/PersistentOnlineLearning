# Senior engineering workflow

This guide defines the standing process for serious PersistentOnlineLearning engineering work. It is not an experiment checklist and does not replace accepted research decisions, experiment contracts, active designs, or plans.

## Core imperative

Continuously evaluate and improve the design while implementing it. Never narrow the job to “make the requested patches,” and never treat “runs and compiles” as equivalent to “finished.” Every implementation and review pass must consider whether the contract, ownership, or design itself is wrong.

Minor design corrections may be made and reported afterward. A change that unlocks new behavior, changes scientific meaning, alters recurrent-state authority or recovery, changes an accepted experiment, or materially expands scope must be discussed before implementation.

## Before coding

1. Read the authoritative research decision or plan, current artifact, relevant comments, and nearby consumers. Do not reconstruct intent from names alone.
2. State the success criteria and current implementation boundary. Distinguish “this unit is complete” from “the experiment or project is complete.”
3. Identify each proposed class or function's main idea: why it exists, what it owns, and what it explicitly does not own.
4. Search across the whole relevant system for existing implementations, duplicated responsibilities, and remote consumers before introducing a new abstraction.
5. Establish state lifecycle explicitly for recurrent work: initialization, reset, continuation, batching, masking, serialization, restore, and failure behavior.
6. If the proposed design already looks overloaded, brittle, or unnecessary, redesign it before writing the first implementation.

## Implementation is iterative

Treat the first working implementation as a draft. For every completed unit, perform these passes before calling it done.

### 1. Contract pass

- Does every class have one coherent main idea?
- Do its details directly support that idea?
- Is orchestration performing calculation, reduction, storage, lifecycle, or policy that belongs to another owner?
- Does data leave a component in the form promised by its contract?
- Are recurrent and parallel forms two implementations of one behavior rather than subtly different models?

### 2. DRY and boundary pass

- Search again across distant units, not only adjacent functions.
- Look for duplicated calculations, counters, validation, parsing, state, representations, and reset logic.
- Check whether two successive owners perform different names for the same work, or whether one owner partially performs another owner's job.
- Prefer eliminating work over wrapping duplicated work in another abstraction.
- Ensure the model, generator, training harness, and evaluation do not quietly own overlapping scientific semantics.

### 3. Design-improvement pass

For every awkward part, ask:

- Do we need this at all?
- Can the design eliminate the state, branch, format, recovery path, or configuration surface?
- Can one focused component centralize genuinely repeated logic?
- Should an overloaded component be decomposed by contract?
- Would a plain dictionary, tuple, or direct function be clearer than a type or subsystem used in one place?
- Is generality being purchased at the expense of correctness, maintainability, speed, concision, or auditability?
- Is complexity compensating for a generator, architecture, or experiment decision that should be reconsidered instead?

### 4. Documentation pass

- A blind reader must understand why every nontrivial class exists from its docstring.
- Document what a calculation or state tells the user, not merely which tensor operations it performs.
- Explain lifecycle boundaries, authority, intentional state, approximations, and surprising framework constraints where they occur.
- Preserve useful comments during rewrites. Concision never justifies making code unauditable.
- Keep scientific interpretation in the artifact that owns it rather than scattering claims through implementation comments.

### 5. Fresh adversarial pass

Reread the modified unit from the beginning as if encountering it for the first time. Do not ask “can I prove the whole system correct?” Ask:

> Did I screw up this unit?

Look especially for misleading names, missing explanations, accidental second sources of truth, hot-path validation, hidden recomputation, pending-state machinery, one-use abstractions, stale development terminology, state leakage between examples, incorrect reset ordering, and behavior that exists only to compensate for an earlier design mistake.

## Technical reduction method

When a unit becomes hard to reason about:

1. Summarize each class and top-level function—not every method—in up to three sentences. Describe what the code actually does from evidence, not what its abstraction was intended to be.
2. Add up to five supporting details only when needed.
3. Compare the summaries. Repeated work, overlapping ownership, and incompatible stages become candidate design defects.
4. Walk the code again. Anything substantial that did not fit its owner's summary is a potential responsibility leak.
5. For each mismatch, decide whether the contract is wrong, the artifact is wrong, or the component should be removed or redesigned.

This method requires concrete and abstract reasoning together: follow the real classes, state, and data flow, then judge whether their responsibilities form a coherent system.

## Five-objective review

Balance all five independently. Do not sacrifice one silently to optimize another.

### Effective

- Does the unit produce the information or behavior the project actually needs?
- Are important products accidentally disabled, deferred, or routed nowhere?
- Does the implementation expose enough state and evidence to answer the scientific question?

### Fast

- Is unnecessary work absent from the normal path?
- Are calculations reused at their natural feedstock?
- Are discovery, formatting, validation, synchronization, and file operations kept out of hot paths?
- Are parallel training and recurrent inference using the intended optimized operators rather than accidental Python loops?

### Maintainable

- Does each responsibility have a clear owner?
- Can a blind reader audit the behavior from structure and documentation?
- Is intentional state small, local, and justified by a real lifecycle gap?
- Are external framework contracts wrapped narrowly rather than reimplemented?

### Correct

- Is algebra owned by the component that defines it?
- Are lifecycle boundaries, authority, failure behavior, and sources of truth explicit?
- Does crash behavior roll back to authoritative state rather than repair speculative state?
- Are recurrent, chunked, padded, resumed, and batched executions equivalent where promised?
- Does the implementation preserve the experiment's controls and interpretation?

### Concise

- Has unnecessary machinery been removed rather than renamed?
- Does every abstraction eliminate repetition or establish a necessary contract?
- Could ordinary language structures express the same idea more clearly?
- Is speculative extensibility being deferred until a second real use appears?

## Recurrent-state verification

For any recurrent model or generator component, verify explicitly:

- full-sequence or chunked output against tokenwise recurrent output;
- state reset and continuation boundaries;
- independent examples do not share state accidentally;
- padding and masking do not update state;
- batching preserves per-sequence state identity;
- serialization and restore reproduce the next output;
- state dtype and device transitions are intentional;
- resumed training or inference restores the authoritative state; and
- long runs remain numerically stable.

Do not infer these properties from a library's general claims. Test the exact wrapper and configuration used by this project.

## Verification and completion

- Compilation is only a syntax gate.
- Test contracts and failure ordering, not merely happy-path execution.
- Verify persistence, resume, and authority in the order they actually occur.
- Regenerate derived artifacts and check that regeneration is deterministic.
- Perform one final source-order adversarial read after all fixes.
- Report design deviations, deliberate approximations, remaining integration gates, and downstream work explicitly.
- Never say a wave, unit, experiment implementation, or plan is complete when only its first runnable draft exists.

## Communication and feedback

Work as a senior engineering partner. Make ordinary in-scope improvements autonomously and report them. Surface major design unlocks before choosing them.

When challenged, investigate the design rather than defending the current artifact. Treat criticism as evidence, not automatically as a literal patch instruction. Identify the underlying contract, responsibility, lifecycle, or reader-model problem; correct the owning artifact; then rerun the independent review passes to ensure the latest correction did not damage the rest of the system.

The objective is maximum quality under the five criteria, not preservation of the first implementation or original plan.
