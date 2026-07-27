# Using standalone recurrent models

The first supported model flavors are the Hugging Face-compatible DeltaNet and Gated DeltaNet causal language models provided by `flash-linear-attention`. PersistentOnlineLearning re-exports those upstream classes without wrapping their blocks, cache, recurrent kernels, or generation implementation.

## DeltaNet

```python
import torch

from persistent_online_learning.model import DeltaNetConfig, DeltaNetForCausalLM

config = DeltaNetConfig(
    vocab_size=32_000,
    hidden_size=768,
    num_hidden_layers=12,
    num_heads=12,
    expand_k=1.0,
    expand_v=1.0,
    attn_mode="chunk",
    use_short_conv=True,
    use_cache=True,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)
model = DeltaNetForCausalLM(config)
```

## Gated DeltaNet

```python
from persistent_online_learning.model import (
    GatedDeltaNetConfig,
    GatedDeltaNetForCausalLM,
)

config = GatedDeltaNetConfig(
    vocab_size=32_000,
    hidden_size=1_024,
    num_hidden_layers=12,
    head_dim=64,
    num_heads=12,
    num_v_heads=12,
    expand_v=2.0,
    attn_mode="chunk",
    use_short_conv=True,
    use_cache=True,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)
model = GatedDeltaNetForCausalLM(config)
```

The configuration objects are the authoritative model configuration. The project does not copy their fields into a second configuration language.

## Forward and loss

```python
input_ids = torch.randint(config.vocab_size, (2, 128), device="cuda")
model = model.to("cuda")
outputs = model(input_ids=input_ids, labels=input_ids)
outputs.loss.backward()
```

## Recurrent generation

```python
model.eval()
prompt = input_ids[:, :16]
generated = model.generate(
    input_ids=prompt,
    max_new_tokens=64,
    do_sample=False,
    use_cache=True,
)
```

FLA owns the recurrent cache and Hugging Face generation preparation. For homogeneous DeltaNet and Gated DeltaNet models, the cache stores fixed-size per-layer recurrent and short-convolution state rather than a token-length key/value history.

## Save and load

```python
model.save_pretrained("checkpoints/delta-model")
restored = type(model).from_pretrained("checkpoints/delta-model")
```

`save_pretrained()` stores model configuration and parameters. Runtime recurrent continuation remains in the returned Hugging Face cache and belongs to the inference or checkpointing caller.
