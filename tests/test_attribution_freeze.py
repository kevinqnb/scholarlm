"""Rung-1 (model-free) unit tests for
``scholarlm.attribution.freeze_model_except_input_embeddings``.

The freeze is the fix for the rung-4 llama-3.1-8b OOM (build note
``2026-08-31-attribution-runner-01``): ``attribute()`` runs ``target.backward()``
with no ``torch.no_grad()``, so without freezing, autograd stores a ``.grad``
buffer for every one of the judge's ~8B parameters. These tests use a tiny
hand-built ``nn.Module`` that exposes the same ``get_input_embeddings()`` /
``_model`` surface the real path uses. The frozen-vs-unfrozen byte-identical
check on a real judge lives in ``attribution_smoke.sh`` check 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch
import torch.nn as nn

from scholarlm.attribution import freeze_model_except_input_embeddings


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 4)
        self.lin = nn.Linear(4, 4)
        self.head = nn.Linear(4, 10)

    def get_input_embeddings(self):
        return self.embed


class _Llm:
    def __init__(self, model):
        self._model = model


class _Judge:
    def __init__(self, model):
        self.llm = _Llm(model)


def test_freeze_leaves_only_input_embeddings_trainable():
    m = _TinyModel()
    freeze_model_except_input_embeddings(_Judge(m))
    assert m.embed.weight.requires_grad is True
    frozen = {n: p.requires_grad for n, p in m.named_parameters() if n != "embed.weight"}
    assert frozen and not any(frozen.values()), frozen


def test_freeze_keeps_embed_output_in_autograd_graph():
    # The reason the embedding matrix must stay trainable: its output then
    # remains a non-leaf that participates in autograd, so grad flows back to
    # it through every (frozen) downstream block. Freezing it too would make
    # embed_out a requires_grad=False leaf and embed_out.grad None.
    m = _TinyModel()
    freeze_model_except_input_embeddings(_Judge(m))

    ids = torch.tensor([[1, 2, 3]])
    embed_out = m.embed(ids)
    assert embed_out.requires_grad is True

    loss = m.head(torch.relu(m.lin(embed_out))).sum()
    loss.backward()

    assert m.embed.weight.grad is not None            # attribution needs this
    assert m.lin.weight.grad is None                  # frozen -> no grad buffer
    assert m.head.weight.grad is None


def test_freeze_asserts_loud_when_no_input_embeddings():
    class _NoEmb(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(3, 3)

        def get_input_embeddings(self):
            return None

    with pytest.raises(AssertionError, match="get_input_embeddings"):
        freeze_model_except_input_embeddings(_Judge(_NoEmb()))


def test_freeze_is_idempotent():
    m = _TinyModel()
    j = _Judge(m)
    freeze_model_except_input_embeddings(j)
    freeze_model_except_input_embeddings(j)
    assert m.embed.weight.requires_grad is True
    assert not m.lin.weight.requires_grad
