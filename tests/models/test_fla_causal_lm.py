"""Acceptance tests for the two upstream FLA causal-language-model flavors."""

from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
import torch
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel

from persistent_online_learning.model import (
    DeltaNetConfig,
    DeltaNetForCausalLM,
    GatedDeltaNetConfig,
    GatedDeltaNetForCausalLM,
)


def _delta_config() -> DeltaNetConfig:
    return DeltaNetConfig(
        attn_mode="chunk",
        hidden_size=64,
        expand_k=1.0,
        expand_v=1.0,
        use_gate=False,
        use_short_conv=True,
        allow_neg_eigval=False,
        conv_size=3,
        use_beta=True,
        use_output_norm=True,
        num_heads=4,
        qk_norm="l2",
        qk_activation="silu",
        hidden_ratio=2,
        intermediate_size=None,
        hidden_act="swish",
        num_hidden_layers=2,
        norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        initializer_range=0.02,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
        fuse_linear_cross_entropy=False,
        use_l2warp=False,
        vocab_size=64,
        attnres_block_size=None,
    )


def _gated_delta_config() -> GatedDeltaNetConfig:
    return GatedDeltaNetConfig(
        attn_mode="chunk",
        hidden_size=64,
        expand_v=1.0,
        use_gate=True,
        use_short_conv=True,
        allow_neg_eigval=False,
        conv_size=3,
        head_dim=16,
        num_heads=4,
        num_v_heads=4,
        hidden_ratio=2,
        intermediate_size=None,
        hidden_act="swish",
        num_hidden_layers=2,
        norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        initializer_range=0.02,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
        fuse_linear_cross_entropy=False,
        use_l2warp=False,
        vocab_size=64,
        attnres_block_size=None,
    )


Flavor = tuple[
    str,
    Callable[[], PretrainedConfig],
    type[PreTrainedModel],
]

_FLAVORS: tuple[Flavor, ...] = (
    ("delta_net", _delta_config, DeltaNetForCausalLM),
    ("gated_deltanet", _gated_delta_config, GatedDeltaNetForCausalLM),
)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("FLA recurrent kernels require CUDA for this acceptance test")


def _tensor_shapes(value: Any, path: str = "cache") -> tuple[tuple[str, tuple[int, ...]], ...]:
    if isinstance(value, torch.Tensor):
        return ((path, tuple(value.shape)),)
    if isinstance(value, dict):
        entries: list[tuple[str, tuple[int, ...]]] = []
        for key in sorted(value):
            entries.extend(_tensor_shapes(value[key], f"{path}.{key}"))
        return tuple(entries)
    if isinstance(value, (list, tuple)):
        entries = []
        for index, item in enumerate(value):
            entries.extend(_tensor_shapes(item, f"{path}[{index}]"))
        return tuple(entries)
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        return _tensor_shapes(tuple(value), path)
    return ()


@pytest.mark.parametrize(("model_type", "make_config", "model_class"), _FLAVORS)
def test_hugging_face_auto_model_constructs_the_upstream_flavor(
    model_type: str,
    make_config: Callable[[], PretrainedConfig],
    model_class: type[PreTrainedModel],
) -> None:
    config = make_config()
    assert config.model_type == model_type
    model = AutoModelForCausalLM.from_config(config)
    assert type(model) is model_class


@pytest.mark.cuda
@pytest.mark.parametrize(("_model_type", "make_config", "model_class"), _FLAVORS)
def test_forward_loss_and_gradients_reach_the_complete_model(
    _model_type: str,
    make_config: Callable[[], PretrainedConfig],
    model_class: type[PreTrainedModel],
) -> None:
    _require_cuda()
    torch.manual_seed(0)
    model = model_class(make_config()).cuda().train()
    input_ids = torch.randint(3, model.config.vocab_size, (2, 64), device="cuda")

    output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    assert output.loss is not None
    assert torch.isfinite(output.loss)
    output.loss.backward()

    assert model.get_input_embeddings().weight.grad is not None
    assert model.get_output_embeddings().weight.grad is not None
    recurrent_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if any(part in name for part in ("q_proj", "k_proj", "v_proj"))
    ]
    assert recurrent_parameters
    assert all(parameter.grad is not None for parameter in recurrent_parameters)


@pytest.mark.cuda
@pytest.mark.parametrize(("_model_type", "make_config", "model_class"), _FLAVORS)
def test_chunked_and_tokenwise_recurrent_logits_agree(
    _model_type: str,
    make_config: Callable[[], PretrainedConfig],
    model_class: type[PreTrainedModel],
) -> None:
    _require_cuda()
    torch.manual_seed(1)
    model = model_class(make_config()).cuda().eval()
    input_ids = torch.randint(3, model.config.vocab_size, (2, 96), device="cuda")

    with torch.no_grad():
        chunked_logits = model(input_ids=input_ids, use_cache=False).logits
        cache = None
        recurrent_logits = []
        for position in range(input_ids.shape[1]):
            output = model(
                input_ids=input_ids[:, position : position + 1],
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            recurrent_logits.append(output.logits)

    torch.testing.assert_close(
        torch.cat(recurrent_logits, dim=1),
        chunked_logits,
        atol=2e-3,
        rtol=2e-3,
    )


@pytest.mark.cuda
@pytest.mark.parametrize(("_model_type", "make_config", "model_class"), _FLAVORS)
def test_padding_preserves_valid_batch_results(
    _model_type: str,
    make_config: Callable[[], PretrainedConfig],
    model_class: type[PreTrainedModel],
) -> None:
    _require_cuda()
    torch.manual_seed(2)
    model = model_class(make_config()).cuda().eval()
    first = torch.randint(3, model.config.vocab_size, (1, 96), device="cuda")
    second = torch.randint(3, model.config.vocab_size, (1, 71), device="cuda")
    padded_second = torch.nn.functional.pad(second, (0, 25), value=model.config.pad_token_id)
    batch = torch.cat((first, padded_second), dim=0)
    attention_mask = torch.cat(
        (
            torch.ones((1, 96), dtype=torch.long, device="cuda"),
            torch.cat(
                (
                    torch.ones((1, 71), dtype=torch.long, device="cuda"),
                    torch.zeros((1, 25), dtype=torch.long, device="cuda"),
                ),
                dim=1,
            ),
        ),
        dim=0,
    )

    with torch.no_grad():
        batch_logits = model(
            input_ids=batch,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        first_logits = model(input_ids=first, use_cache=False).logits
        second_logits = model(input_ids=second, use_cache=False).logits

    torch.testing.assert_close(batch_logits[0], first_logits[0], atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(
        batch_logits[1, : second.shape[1]],
        second_logits[0],
        atol=2e-3,
        rtol=2e-3,
    )


@pytest.mark.cuda
@pytest.mark.parametrize(("_model_type", "make_config", "model_class"), _FLAVORS)
def test_recurrent_cache_shapes_do_not_grow_with_sequence_length(
    _model_type: str,
    make_config: Callable[[], PretrainedConfig],
    model_class: type[PreTrainedModel],
) -> None:
    _require_cuda()
    torch.manual_seed(3)
    model = model_class(make_config()).cuda().eval()
    prompt = torch.randint(3, model.config.vocab_size, (2, 96), device="cuda")

    with torch.no_grad():
        output = model(input_ids=prompt, use_cache=True)
        cache = output.past_key_values
        prompt_shapes = _tensor_shapes(cache)
        for _ in range(32):
            token = torch.randint(3, model.config.vocab_size, (2, 1), device="cuda")
            output = model(input_ids=token, past_key_values=cache, use_cache=True)
            cache = output.past_key_values

    assert prompt_shapes
    assert _tensor_shapes(cache) == prompt_shapes


@pytest.mark.cuda
@pytest.mark.parametrize(("_model_type", "make_config", "model_class"), _FLAVORS)
def test_save_load_and_generate_use_the_upstream_hugging_face_contract(
    _model_type: str,
    make_config: Callable[[], PretrainedConfig],
    model_class: type[PreTrainedModel],
) -> None:
    _require_cuda()
    torch.manual_seed(4)
    model = model_class(make_config()).cuda().eval()
    prompt = torch.randint(3, model.config.vocab_size, (2, 8), device="cuda")

    with TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        restored = model_class.from_pretrained(Path(directory)).cuda().eval()

        with torch.no_grad():
            expected = model(input_ids=prompt, use_cache=False).logits
            actual = restored(input_ids=prompt, use_cache=False).logits
            generated = restored.generate(
                input_ids=prompt,
                min_new_tokens=4,
                max_new_tokens=4,
                do_sample=False,
                use_cache=True,
            )

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert generated.shape == (2, 12)
