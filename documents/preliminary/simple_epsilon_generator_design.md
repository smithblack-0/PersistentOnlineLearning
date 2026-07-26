# Simple epsilon generator design

Status: provisional design for the first generator implementation.

## Purpose

The first generator flavor establishes the smallest complete recurrent token process and the component boundaries needed to revise it. It deliberately does not implement the later multi-channel hashed controller, local history filter, batching, telemetry, checkpoint policy, or storage integration.

The machine emits a token from its current hidden state, reduces that token to a hash code, and applies the corresponding state transition. The updated state controls the next emitted token.

## Construction flow

A caller may either inject already constructed components or ask the flavor factory to construct the default composition:

```text
HashReduction        StateCore        GenerativeDecoder
       \                 |                  /
        \                |                 /
             SimpleEpsilonMachine
```

Dictionary construction is a thin registry dispatch to the flavor factory. The registry does not parse flavor fields or own a general configuration language.

Constructing a machine creates one process. Replacing or rerolling a process means constructing another machine; hidden-rule construction is not exposed as a runtime lifecycle method.

## Responsibility map

| Component | Owns | Does not own | Dependencies | Reasons to change |
| --- | --- | --- | --- | --- |
| `TokenGenerator` and registry | Common generation and process-state restoration contract; mapping a `type` name to a flavor factory | Flavor mechanisms, parameter validation, process replacement, random-state authority, batching, persistence policy | Registered factory callables | The common generator contract or construction dispatch changes |
| `HashReduction` | A token-to-hash mapping, applying it, and loading/saving that mapping | Hidden state, transitions, output probabilities, process replacement, orchestration | Active PyTorch random stream during construction | The input-reduction mechanism changes |
| `StateCore` | Current categorical state, hash-indexed transition table, transition application, and process state | Token identities, hashing policy, output decoding, process replacement, orchestration | Active PyTorch random stream during construction | The hidden-state or transition mechanism changes |
| `GenerativeDecoder` | State-conditioned token supports and probabilities, epsilon mixing, and token sampling | Hashing, state transitions, process replacement, random-state authority, sequence orchestration | Active PyTorch random stream | The state-to-output mechanism changes |
| `SimpleEpsilonMachine` | Ordering calls between injected components, composing their process state, and the default component factory | Component algebra, construction randomness, process replacement, batch construction, telemetry, checkpoint/storage policy | Injected reducer, state core, and decoder | The machine flow or default composition changes |

## Primitive contracts

### `HashReduction`

- construction samples one complete mapping;
- `__call__(token) -> hash_code` applies it;
- `state_dict()` and `load_state_dict()` own mapping restoration.

### `StateCore`

- construction samples transitions and starts in state zero;
- `state` exposes the current state index;
- `transition(hash_code) -> next_state` updates and returns the state;
- `state_dict()` and `load_state_dict()` own transition and runtime restoration.

### `GenerativeDecoder`

- construction samples state-conditioned supports and weights;
- `distribution(state) -> probabilities` exposes the exact distribution;
- `sample(state) -> token` uses the active PyTorch random stream;
- `state_dict()` and `load_state_dict()` own decoder rules, not global RNG state.

Each state has a small disjoint support. With probability `epsilon`, sampling is uniform over the vocabulary; otherwise it follows the state-conditioned categorical distribution.

### `SimpleEpsilonMachine`

`step()` performs only:

1. sample a token from the decoder using the current state;
2. reduce the token to a hash code;
3. transition the state core with that hash code;
4. return the sampled token.

The class method `create()` is the composition root for the default primitives. Direct construction accepts any injected objects satisfying the narrow reducer, state-core, and decoder protocols.

## Process and random lifecycle

- Construction creates a complete valid process. There is no partially initialized public object.
- A new procedural process is represented by a newly constructed generator, not by mutating an existing generator's hidden rules.
- The harness owns process replacement and random-state authority. It may set PyTorch seeds before construction when reproducibility matters.
- `state_dict()` delegates process state to each component and does not duplicate global RNG state.
- Exact stochastic continuation requires the harness checkpoint to restore PyTorch RNG state after constructing and loading the generator.
- Callers should load into a newly configured machine and discard it if validation fails; the generator package does not own checkpoint transaction policy.

## Blind critique history

Earlier iterations removed concrete dependency coupling, full-vocabulary sampling on the normal path, redundant previous-token state, accidentally similar decoder supports, and possible use of uninitialized rule tensors.

The maintainability pass then rejected machine-owned sampling RNG. A further abstraction-level review rejected both injected RNG objects and seeded `reinitialize` methods: those interfaces exposed construction details through runtime orchestration.

The final correction removes `reinitialize` entirely. Component constructors establish hidden rules using the normal PyTorch random stream, and the harness constructs a new generator when it needs a new process.

## Deliberate limits

This is a scalar correctness implementation. A later flavor or optimized primitive may add vectorized generation after the component contracts and scientific behavior are accepted. The initial PR should not anticipate the full generator family.
