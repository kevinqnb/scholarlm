"""Rung-1 (model-free) unit tests for
``scholarlm.attribution.enable_decoder_gradient_checkpointing``.

Checkpointing is the fix for the rung-4 ``contrastive_gradient`` OOM on the
32-layer judges (build note ``2026-08-31-attribution-runner-01``, rung-4 second
qsub): ``ContrastiveGradientAttribution`` backpropagates from the final logits
through every decoder layer, so without checkpointing all ~7.6k context tokens'
activations are held for all 32 layers (~55–60 GB) and OOM an 80 GB GPU *even
after* the input-embedding freeze.

These use a 3-layer toy ``LlamaForCausalLM`` so the real HF
``gradient_checkpointing_enable`` path and the ``GradientCheckpointingLayer``
``training`` gate are exercised. The peak-memory drop on a real judge at a
>7.6k-token context is asserted in ``attribution_smoke.sh`` check 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch

transformers = pytest.importorskip("transformers")
from transformers import LlamaForCausalLM  # noqa: E402
from transformers.models.llama.configuration_llama import LlamaConfig  # noqa: E402

from scholarlm.attribution import (  # noqa: E402
    _reassert_decoder_layers_training,
    enable_decoder_gradient_checkpointing,
    freeze_model_except_input_embeddings,
)

_N_LAYERS = 3


class _Llm:
    def __init__(self, model):
        self._model = model


class _Judge:
    def __init__(self, model):
        self.llm = _Llm(model)


def _tiny_llama(*, attention_dropout: float = 0.0) -> LlamaForCausalLM:
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=_N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=attention_dropout,
    )
    model = LlamaForCausalLM(cfg)
    model.eval()  # nnsight loads judges in eval mode
    return model


def test_enable_sets_checkpointing_and_training_on_every_layer():
    m = _tiny_llama()
    for layer in m.model.layers:
        assert layer.gradient_checkpointing is False
        assert layer.training is False

    enable_decoder_gradient_checkpointing(_Judge(m))

    assert len(m.model.layers) == _N_LAYERS
    for layer in m.model.layers:
        assert layer.gradient_checkpointing is True
        assert layer.training is True                # engages the HF training gate
        assert layer.self_attn.training is False     # NOT recursive -> no dropout path


def test_enable_actually_engages_the_checkpoint_wrapper():
    """The ``training`` gate is the silent-no-op risk: prove the per-layer
    checkpoint function actually runs in a real forward/backward once the helper
    has flipped ``.training``."""
    m = _tiny_llama()
    freeze_model_except_input_embeddings(_Judge(m))
    enable_decoder_gradient_checkpointing(_Judge(m))

    calls = {"n": 0}
    for layer in m.model.layers:
        inner = layer._gradient_checkpointing_func

        def spy(*a, _inner=inner, **k):
            calls["n"] += 1
            return _inner(*a, **k)

        layer._gradient_checkpointing_func = spy

    ids = torch.tensor([[1, 2, 3, 4, 5]])
    emb = m.model.embed_tokens(ids)
    emb.retain_grad()
    m.model(inputs_embeds=emb, use_cache=False).last_hidden_state.pow(2).sum().backward()

    assert calls["n"] == _N_LAYERS, calls["n"]


def test_reassert_training_restores_the_gate_without_touching_children():
    """nnsight's trace lifecycle flips decoder-layer .training back to False after
    a forward (observed in attribution_smoke.sh check 0), which would silently
    un-checkpoint calls 2..N of a dataset run. attribute() calls
    _reassert_decoder_layers_training before every trace to restore it."""
    m = _tiny_llama()
    enable_decoder_gradient_checkpointing(_Judge(m))
    for layer in m.model.layers:          # simulate the post-trace reset
        layer.training = False

    _reassert_decoder_layers_training(_Judge(m))

    for layer in m.model.layers:
        assert layer.training is True
        assert layer.self_attn.training is False   # children untouched
        assert layer.gradient_checkpointing is True  # flag never got cleared


def test_enable_refuses_nonzero_dropout():
    m = _tiny_llama(attention_dropout=0.1)
    with pytest.raises(AssertionError, match="dropout"):
        enable_decoder_gradient_checkpointing(_Judge(m))
    # and it must not have half-applied
    for layer in m.model.layers:
        assert layer.training is False


def test_checkpointing_leaves_input_grad_bitwise_unchanged():
    """d(loss)/d(embed_out) must be *exactly* what it was without checkpointing —
    checkpointing only recomputes forward activations, it does not change the
    math. CPU float32 + deterministic math attention + dropout=0 -> bitwise
    equal (the GPU smoke's ~1e-2 run-to-run is non-deterministic SDPA backward,
    isolated out here)."""
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])

    def input_grad(*, checkpoint: bool) -> torch.Tensor:
        m = _tiny_llama()
        freeze_model_except_input_embeddings(_Judge(m))  # same regime as the runner
        if checkpoint:
            enable_decoder_gradient_checkpointing(_Judge(m))
        emb = m.model.embed_tokens(ids)
        emb.retain_grad()
        out = m.model(inputs_embeds=emb, use_cache=False)
        out.last_hidden_state.pow(2).sum().backward()
        return emb.grad.clone()

    g0 = input_grad(checkpoint=False)
    g1 = input_grad(checkpoint=True)
    assert torch.equal(g0, g1), (g0 - g1).abs().max()
