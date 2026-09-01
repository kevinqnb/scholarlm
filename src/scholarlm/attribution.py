"""
Input-token attribution for JudgementLM's true/false judgements.

Two Input×Gradient methods share one interface (``AttributionMethod``) and
registry (``ATTRIBUTION_REGISTRY``):

- ``ContrastiveGradientAttribution`` — baseline: attributes the true/false
  logit contrast back to each context-token embedding.
- ``ProbeAttribution`` — attributes a trained head-probe's output (the
  no-Platt ``head_probe_noplatt.pkl`` variant only — see
  ``analysis.loaders.load_trained_probe``) back to each context-token
  embedding.

Both wrap an already-loaded ``JudgementLM`` instance (reusing its model,
tokenizer, and binary-token-id lookup) but run their own single-pass
``llm.trace()`` rather than calling ``JudgementLM.generate()`` — ``generate()``
detaches every tensor it saves, which is incompatible with the live autograd
graph attribution needs. Every ``INTERP_JUDGE_REGISTRY`` model (including
``qwen-2.5-7b-base``, the only judge model with a saved
``head_probe_noplatt.pkl``) uses ``max_new_tokens=1``, so ``generate()``'s
single generation step is itself the prefill forward pass — both methods only
ever need one forward pass, never multi-step generation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

try:
    import torch
    HAS_GPU_DEPS = True
except ImportError:
    print("Warning: PyTorch not available; attribution methods will not be usable.")
    HAS_GPU_DEPS = False

from scholarlm.judgementlm import JudgementLM, tokenize


def _logsumexp_true_false(logits_last: "torch.Tensor", binary_token_ids: dict[str, int]) -> tuple:
    """
    Case-marginalized log P(true)/log P(false) from a single position's logits,
    matching ``JudgementLM.generate()``'s ``p_true``/``p_false`` convention
    exactly (see build note 2026-08-18-token-attribution-01, Stage 1 item 4).
    """
    true_id_1 = binary_token_ids["true"]
    true_id_2 = binary_token_ids["True"]
    false_id_1 = binary_token_ids["false"]
    false_id_2 = binary_token_ids["False"]

    true_terms = [logits_last[true_id_1]]
    if true_id_1 != true_id_2:
        true_terms.append(logits_last[true_id_2])
    false_terms = [logits_last[false_id_1]]
    if false_id_1 != false_id_2:
        false_terms.append(logits_last[false_id_2])

    log_p_true = torch.logsumexp(torch.stack(true_terms), dim=0)
    log_p_false = torch.logsumexp(torch.stack(false_terms), dim=0)
    return log_p_true, log_p_false


def freeze_model_except_input_embeddings(judge: JudgementLM) -> None:
    """Set ``requires_grad=False`` on every judge-model parameter *except* the
    input-embedding matrix.

    Input×Gradient attribution only needs ``d(target)/d(input embeddings)``.
    ``attribute()`` runs ``target.backward()`` with no ``torch.no_grad()``; if
    every weight still has ``requires_grad=True`` (the HF default), autograd
    allocates a ``.grad`` buffer for all ~8B parameters — a second full copy of
    the model — which OOMs an 80 GB GPU on the larger judges (see build note
    ``2026-08-31-attribution-runner-01``, rung-4 first qsub: llama-3.1-8b OOM at
    a 1.6k-token context where qwen-2.5-7b peaked at 32.8 GB).

    This is mathematically inert for the score. The embedding matrix stays
    ``requires_grad=True`` so ``embed_tokens.output`` remains a non-leaf that
    participates in autograd (freezing it too would break the graph and make
    ``embed_out.grad`` ``None``). Every downstream op still enters the graph
    because it depends on ``embed_out``, so ``grad_{embed_out} target`` is
    unchanged — the transformer-block weight grads that we no longer store were
    never read (the scores are ``<grad_{embed_out}, embed_out>``). Verified
    byte-identical frozen-vs-unfrozen in ``attribution_smoke.sh`` check 0.
    """
    # Underlying HF nn.Module. `device_map="auto"` (nnterp's default) keeps the
    # whole model on one GPU when it fits — no accelerate offload swapping the
    # parameter tensors out from under these flags — which is the only regime
    # this attribution runs in. requires_grad_ is a flag flip, not a data copy,
    # so it is safe even on a meta/offloaded parameter.
    raw = judge.llm._model
    raw.requires_grad_(False)
    emb = raw.get_input_embeddings()
    assert emb is not None, (
        "judge model get_input_embeddings() is None — cannot attribute to "
        "input-token embeddings"
    )
    emb.weight.requires_grad_(True)


def enable_decoder_gradient_checkpointing(judge: JudgementLM) -> None:
    """Turn on activation (gradient) checkpointing for the judge's decoder stack.

    ``ContrastiveGradientAttribution`` runs the full forward with a live autograd
    graph and backpropagates from the final logits, so every decoder layer's
    activations (residual stream, attention projections, MLP intermediates) are
    retained for the whole context. At pond/supermat context lengths (up to
    ~7.6k tokens) that is ~55–60 GB on the 32-layer judges (llama-3.1-8b,
    mistral-7b) and OOMs an 80 GB GPU *even after* the input-embedding freeze
    (build note ``2026-08-31-attribution-runner-01``, rung-4 second qsub — the
    28-layer qwen-2.5-7b peaked at 76 GiB, a 3.6% margin; the 32-layer judges go
    over). Checkpointing recomputes each layer during the backward pass instead
    of storing it — ~35% more walltime for an ~5–8× cut to that term.

    Mathematically inert for the score: checkpointing only recomputes forward
    activations, it does not change ``d(target)/d(embed_out)``. Verified
    bitwise on CPU float32 in ``tests/test_attribution_checkpoint.py`` and by a
    peak-memory-drop assertion on a real judge at a >7.6k-token context in
    ``attribution_smoke.sh`` check 0.

    Two silent-failure modes this guards against explicitly (both are the
    CLAUDE.md "runs clean, number quietly wrong" trap):

    - HF gates the checkpoint call on ``module.training``
      (``transformers.modeling_layers.GradientCheckpointingLayer.__call__``:
      ``if self.gradient_checkpointing and self.training``). nnsight loads the
      judge in eval mode, so ``gradient_checkpointing_enable()`` alone is a
      no-op. We set ``.training = True`` on the decoder-layer objects only (not
      recursively) to engage the gate while leaving ``self_attn.training``
      ``False``.
    - ``.training = True`` would also activate any dropout. We assert every
      ``*dropout`` / ``*pdrop`` probability in the model config is ``0.0`` (or
      ``None``) first — a nonzero one would make the attribution scores
      stochastic in a way that still looks plausible.

    Applied by ``ContrastiveGradientAttribution`` only. ``ProbeAttribution``
    reads ``o_proj.input`` mid-stack; non-reentrant recompute would re-run that
    region in the backward pass, and probe's forward footprint already fits.
    """
    raw = judge.llm._model
    if not raw.supports_gradient_checkpointing:
        raise ValueError(
            f"{type(raw).__name__} does not support gradient checkpointing"
        )

    cfg = raw.config
    dropout_keys = [
        k for k in vars(cfg) if k.endswith("dropout") or k.endswith("pdrop")
    ]
    assert dropout_keys, (
        f"{type(cfg).__name__} exposes no *dropout / *pdrop keys — cannot verify "
        "that train() mode adds no stochasticity; refusing to enable checkpointing"
    )
    nonzero = {k: getattr(cfg, k) for k in dropout_keys if getattr(cfg, k)}
    assert not nonzero, (
        f"judge model config has nonzero dropout {nonzero} — setting decoder "
        "layers to train() mode for gradient checkpointing would make the "
        "attribution scores stochastic"
    )

    raw.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    layers = raw.model.layers
    n_layers = raw.config.num_hidden_layers
    assert len(layers) == n_layers, (
        f"raw.model.layers has {len(layers)} entries, config num_hidden_layers "
        f"is {n_layers}"
    )
    for layer in layers:
        assert getattr(layer, "gradient_checkpointing", False), (
            f"{type(layer).__name__}.gradient_checkpointing is False after "
            "gradient_checkpointing_enable() — the checkpoint wrapper will not run"
        )
        layer.training = True


class AttributionMethod(ABC):
    """
    Base interface for input-token attribution methods against a loaded
    ``JudgementLM``. Every implementation returns one scalar Input×Gradient
    score per context token: ``s(x_t) = <grad_{x_t} target, x_t>``.
    """

    @abstractmethod
    def attribute(self, instructions: str, context: str, query: str) -> dict[str, Any]:
        """
        Args:
            instructions: The instructions string.
            context: The context string.
            query: The query string.

        Returns:
            dict with at least:
                'scores' (np.ndarray): float32, shape (n_context_tokens,),
                    one Input×Gradient score per context token, in
                    ``context_token_indices`` order.
                'context_token_indices' (list[int]): token positions in the
                    full prompt that ``scores`` corresponds to.
        """
        ...


class ContrastiveGradientAttribution(AttributionMethod):
    """
    Baseline attribution: ``s(x_t) = <grad_{x_t}(f_T - f_F), x_t>``, where
    ``f_T``/``f_F`` are the case-marginalized log P(true)/log P(false) at the
    first (and, given ``max_new_tokens=1``, only) generation step.
    """

    def __init__(self, judge: JudgementLM):
        self.judge = judge
        freeze_model_except_input_embeddings(judge)
        enable_decoder_gradient_checkpointing(judge)

    def attribute(self, instructions: str, context: str, query: str) -> dict[str, Any]:
        judge = self.judge
        (
            tokenized_prompt,
            instruction_token_indices,
            context_token_indices,
            query_token_indices,
        ) = tokenize(
            instructions, context, query, judge.tokenizer,
            use_chat_template=judge.use_chat_template, answer_cue=judge.answer_cue,
        )

        with judge.llm.trace(tokenized_prompt, logits_to_keep=1) as tracer:
            embed_out = judge.llm.model.embed_tokens.output
            embed_vals = embed_out.detach().save()

            logits_last = judge.llm.logits[0, -1, :]
            log_p_true, log_p_false = _logsumexp_true_false(logits_last, judge._binary_token_ids)
            target = log_p_true - log_p_false

            with target.backward():
                grad = embed_out.grad.save()
            saved_target = target.save()

        scores_all = (grad[0].float() * embed_vals[0].float()).sum(dim=-1)
        scores = scores_all[context_token_indices].cpu().numpy().astype(np.float32)

        return {
            "scores": scores,
            "context_token_indices": context_token_indices,
            "target": float(saved_target.item()),
        }


class ProbeAttribution(AttributionMethod):
    """
    Attributes a trained head probe's output back to each context-token
    embedding: ``s(x_t) = <grad_{x_t} g(X), x_t>``, where
    ``g(X) = sigmoid(<theta, scaler(concat(attn_output[l, h, :] for (l, h) in
    top_k_heads))>)`` is reconstructed as explicit torch ops from the fitted
    sklearn artifact (``head_probe_noplatt.pkl`` — a bare
    ``Pipeline(StandardScaler, LogisticRegression)``, no Platt/
    CalibratedClassifierCV wrapper).
    """

    def __init__(self, judge: JudgementLM, probe_data: dict):
        max_new_tokens = judge.sampling_params["max_new_tokens"]
        assert max_new_tokens == 1, (
            "ProbeAttribution assumes generate()'s single tracer.iter step is "
            "itself the prefill forward pass (head_probe_noplatt.pkl was "
            "trained on exactly that position) — see build note "
            "2026-08-18-token-attribution-01, Stage 1 item 2. Got "
            f"max_new_tokens={max_new_tokens}."
        )
        pipeline = probe_data["probe"]
        assert type(pipeline).__name__ == "Pipeline", (
            "ProbeAttribution requires a bare sklearn Pipeline (StandardScaler "
            f"+ LogisticRegression), not a Platt-calibrated wrapper. Got {type(pipeline)}."
        )
        scaler = pipeline.named_steps["scaler"]
        clf = pipeline.named_steps["clf"]

        self.judge = judge
        self.top_k_heads = [(int(l), int(h)) for l, h in probe_data["top_k_heads"]]
        self.n_heads = probe_data["n_heads"]
        self.head_dim = probe_data["head_dim"]

        n_features = len(self.top_k_heads) * self.head_dim
        assert scaler.mean_.shape == (n_features,), (
            f"Probe scaler.mean_ shape {scaler.mean_.shape} does not match "
            f"len(top_k_heads) * head_dim = {n_features}."
        )

        self.scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32)
        self.scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32)
        self.clf_coef = torch.tensor(clf.coef_[0], dtype=torch.float32)
        self.clf_intercept = torch.tensor(clf.intercept_[0], dtype=torch.float32)

        freeze_model_except_input_embeddings(judge)

    def attribute(self, instructions: str, context: str, query: str) -> dict[str, Any]:
        judge = self.judge
        (
            tokenized_prompt,
            instruction_token_indices,
            context_token_indices,
            query_token_indices,
        ) = tokenize(
            instructions, context, query, judge.tokenizer,
            use_chat_template=judge.use_chat_template, answer_cue=judge.answer_cue,
        )

        # nnsight requires Envoy reads (layer.self_attn.o_proj.input) in the
        # model's actual forward execution order — top_k_heads is sorted by
        # probe F1, not by layer, so fetch each distinct layer once in
        # ascending order, then reassemble per-head vectors in top_k_heads'
        # original order (which the fitted scaler/coef depend on).
        unique_layers = sorted({l for l, _ in self.top_k_heads})
        with judge.llm.trace(tokenized_prompt, logits_to_keep=1) as tracer:
            embed_out = judge.llm.model.embed_tokens.output
            embed_vals = embed_out.detach().save()

            layer_heads = {}
            for l in unique_layers:
                layer = judge.llm.model.layers[l]
                attn_in = layer.self_attn.o_proj.input
                if attn_in.ndim == 3:  # (batch, seq_len, hidden_size)
                    vec = attn_in[0, -1, :]
                else:  # (seq_len, hidden_size)
                    vec = attn_in[-1, :]
                layer_heads[l] = vec.view(self.n_heads, self.head_dim)

            head_vecs = [layer_heads[l][h, :] for l, h in self.top_k_heads]
            feat = torch.cat(head_vecs, dim=0).float()

            scaled = (feat - self.scaler_mean.to(feat.device)) / self.scaler_scale.to(feat.device)
            logit = (scaled * self.clf_coef.to(feat.device)).sum() + self.clf_intercept.to(feat.device)
            prob = torch.sigmoid(logit)

            with prob.backward():
                grad = embed_out.grad.save()
            saved_prob = prob.save()

        scores_all = (grad[0].float() * embed_vals[0].float()).sum(dim=-1)
        scores = scores_all[context_token_indices].cpu().numpy().astype(np.float32)

        return {
            "scores": scores,
            "context_token_indices": context_token_indices,
            "probe_output": float(saved_prob.item()),
        }


ATTRIBUTION_REGISTRY: dict[str, type[AttributionMethod]] = {
    "contrastive_gradient": ContrastiveGradientAttribution,
    "probe": ProbeAttribution,
}
