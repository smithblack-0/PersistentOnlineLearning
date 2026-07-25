# Generator Design Notes

## Design problem

The generator must expose learnable rules over medium and long horizons without requiring an unidentifiably large hidden parameter set. It must also emit tokens in a normal-sized vocabulary rather than a tiny symbolic alphabet.

A useful first generator should provide:

- compact hidden rules;
- long temporal reach;
- enough stochasticity that prediction is not trivial;
- enough structure that observing more of the stream improves prediction;
- a sharp enough output distribution that rules are visible;
- many independently resampled process instances.

## Current leading direction

The current candidate combines a hashed, factorized controller with a sparse recent-token filter.

### 1. Distributed token hashes

Each vocabulary token receives several independent random hash values:

```text
word -> [hash_1, hash_2, ..., hash_N]
```

Each hash value lies in a much smaller bin space. Collisions are deliberate. A single channel therefore treats many technically different tokens as members of the same functional class, while the full collection of hash values preserves distinctions between them.

The collision rate is a design parameter. It controls how many vocabulary items share one coarse functional property.

### 2. Factorized controller state

Each hash channel owns a small categorical hidden state. When a token is emitted, its hash value in each channel selects that channel's next transition.

Conceptually:

```text
for each channel:
    bin = token_hash[channel]
    state[channel] = transition[channel][state[channel]][bin]
```

The complete controller is the tuple of all channel states. Its representation grows linearly with the number of channels, while the number of joint configurations grows combinatorially.

The controller's responsibility is long- and medium-term structure that cannot be recovered from only the most recent words.

### 3. Controller candidate filtering

The controller maps its current state back into broad vocabulary classes. Hash collisions intentionally leave a candidate pool rather than identifying one exact token.

The controller should be understood as a filter:

```text
full vocabulary -> controller-compatible candidate subset
```

It does not need to resolve all local grammatical or collocational ambiguity.

### 4. Sparse recent-token filtering

A sparse bigram, trigram, or similarly local transition system applies a second filter based on recent output history:

```text
controller candidates -> locally valid candidates
```

This layer may serve as a first proxy for both shallow grammar and collocation. It should remove locally invalid words rather than replace the controller's long-horizon role.

The final token is sampled from the surviving set or from simple weights over that set.

## Why this direction is currently strongest

- It works directly with a normal vocabulary size.
- Its hidden rule count can remain comparatively small.
- Hash collisions create controlled redundancy and partial functional equivalence.
- Factorized state provides more history bandwidth than one small flat hidden state.
- The sparse local system supplies surface regularity without requiring a full handcrafted grammar.
- Every decision can be inspected and measured.

## Important unresolved details

### Output construction

The exact mapping from controller state to candidate words is not settled. It must avoid both extremes:

- smoothing the output across so many words that the controller's rules become invisible;
- becoming effectively one-to-one and eliminating the intended redundancy.

### Filter overlap

If the controller candidate set and local transition set are independently random, their intersection may frequently be empty. Their construction likely needs shared hash structure or another deliberate correlation.

### Controller necessity

The local filter may become surprisingly capable. The controller must continue to earn its complexity by creating dependencies that the local system cannot solve.

### Process diversity

Resampling transition probabilities alone may not create enough distinct rule systems. Resampling controller topology, selected transition tables, local connectivity, or slower latent regimes may be needed.

## Alternative generator families retained for comparison

- A very small randomly initialized or statistically reinitialized neural language model.
- A continuous logit-state or state-space generator with token-driven updates.
- Unifilar hidden Markov models or epsilon-machine-inspired generators.
- Hierarchical or chaotic deterministic systems with compact parameterizations.

These remain useful comparison families. They should not be merged into the first generator unless the simpler controller design exposes a specific deficiency.
