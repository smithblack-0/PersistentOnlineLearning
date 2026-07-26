# Using token generators

The generator package contains small stateful process mechanisms. It does not construct language-model batches, replace processes, manage experiment schedules, own random-state authority, or manage checkpoint storage.

## Construct the simple epsilon machine

Dictionary construction selects a registered flavor and forwards the remaining fields directly to its factory. Set the PyTorch seed in the harness before construction when a reproducible process is required:

```python
import torch

from persistent_online_learning.generator import build_generator

torch.manual_seed(0)
generator = build_generator(
    {
        "type": "simple_epsilon",
        "vocab_size": 512,
        "state_count": 16,
        "hash_count": 32,
        "outcomes_per_state": 8,
        "epsilon": 0.01,
    }
)

tokens = generator.generate(128)
```

The fields mean:

- `vocab_size`: possible token IDs;
- `state_count`: categorical hidden states;
- `hash_count`: shared token hash codes used to select transitions;
- `outcomes_per_state`: disjoint structured token outcomes available from each state; `state_count * outcomes_per_state` cannot exceed the vocabulary;
- `epsilon`: probability of sampling uniformly from the vocabulary.

## Inject components directly

Direct construction is useful when replacing one mechanism while preserving the machine flow:

```python
from persistent_online_learning.generator import (
    GenerativeDecoder,
    HashReduction,
    SimpleEpsilonMachine,
    StateCore,
)

machine = SimpleEpsilonMachine(
    reducer=HashReduction(vocab_size=512, hash_count=32),
    state_core=StateCore(state_count=16, hash_count=32),
    decoder=GenerativeDecoder(
        state_count=16,
        vocab_size=512,
        outcomes_per_state=8,
        epsilon=0.01,
    ),
)
```

`SimpleEpsilonMachine.step()` samples from the current state, reduces the emitted token to a hash code, and applies that hash-selected transition. Component instances remain available as `reducer`, `state_core`, and `decoder` for focused experiments.

## Replace a process

A new process is a new generator object. The harness chooses when replacement occurs and may seed PyTorch immediately before construction:

```python
torch.manual_seed(1)
generator = build_generator(specification)
```

The existing object is not rerolled in place. This keeps hidden-rule construction outside runtime orchestration and prevents stale component state from surviving a process replacement.

## Save and restore continuation

The generator returns ordinary nested process-state dictionaries. PyTorch RNG state belongs to the harness checkpoint because the same random stream may also be used by data loading, models, or other runtime systems:

```python
process_state = generator.state_dict()
random_state = torch.get_rng_state()

restored = build_generator(specification)
restored.load_state_dict(process_state)
torch.set_rng_state(random_state)
```

Restore RNG state after construction, because constructing the compatible replacement object samples temporary rules before `load_state_dict()` replaces them. Each component validates the process state it owns. Atomic publication and rollback remain checkpoint-subsystem responsibilities.

## Add another flavor

A new flavor should compose focused mechanisms and register one factory:

```python
from persistent_online_learning.generator import register_generator

register_generator("my_flavor", MyMachine.create)
```

The factory receives the dictionary fields after `type`. Avoid adding a general generator configuration hierarchy; field meaning and validation belong to the selected flavor and its components.
