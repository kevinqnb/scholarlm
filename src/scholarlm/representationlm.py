"""Key-term representation collection from a non-instruction-tuned base LLM.

``RepresentationLM`` passes each document through a base model as raw
next-token prediction (no chat template, BOS added, one forward pass per
document) and reads hidden states at a configurable list of layers, at the
token positions that fall inside a whole-word occurrence of a supplied key
term. One forward pass yields every requested layer.

Design decisions (see ``notes/scholarlm/builds/2026-08-31-representation-lm-01.md``
and its 2026-09-01 multi-layer extension):

- **Layers.** ``layers`` is a list of integers, default
  ``[0, 8, 16, 24, 32]`` (``DEFAULT_LAYERS``), where:
    - ``0`` — the token-embedding output (residual stream *before* block 0),
      read as ``self.llm.token_embeddings``.
    - ``1 <= L <= n_layers-1`` — the residual stream *after* ``L`` transformer
      blocks, read as ``self.llm.model.layers[L-1].output[0]`` (the idiom
      ``JacobianLensLM`` uses).
    - ``L == n_layers`` (32 for Llama-3.1-8B) — the **post-final-norm** hidden
      state (the vector the unembedding sees), read as
      ``self.llm.ln_final.output``. This is exactly what the single-layer
      version collected, so the layer-``n_layers`` dataset is a strict
      superset of the pre-2026-09-01 artifact.
  ``resolve_layers()`` validates the list against ``n_layers`` and returns it
  sorted + de-duplicated; reads are emitted in ascending order inside the
  trace because nnsight forbids out-of-order envoy reads (smoke run 1 of the
  original build died on exactly this).
  **Norm-space caveat:** layer ``n_layers`` is post-final-norm; every other
  layer is a raw *pre-norm* residual. Row-L2 norms are not comparable across
  that boundary (pre ≈ 68, post ≈ 145 for the pond run) — do not compare
  magnitudes across layers without accounting for it.
- **Read point.** ``nnterp.StandardizedTransformer`` exposes
  ``self.llm.ln_final.output`` and ``self.llm.token_embeddings`` as real
  per-position tensors inside a trace (nnterp validates exactly this at
  model-load time). The post-final-norm residual is read directly — do not
  re-apply ``model.model.norm``.
- **BOS.** The Llama-3.1-8B fast tokenizer has no ``add_bos_token`` attribute
  to flip; ``add_special_tokens=True`` is what prepends ``<|begin_of_text|>``.
  The ``JacobianLensLM`` guard is kept for tokenizers that do expose the flag.
- **Matching.** Case-insensitive, whole-word (regex ``\b`` word boundaries),
  simple plural (``term`` and ``term+"s"``). For a term split into multiple
  subword tokens, only the **last** subword token's representation is kept.
  ``offset_mapping`` intervals for space-prefixed BPE tokens include the
  leading space, so token selection is "the token whose char interval
  contains the match's last character", not offset-set equality.
- **Over-length docs.** Tail truncation only (no chunking): keep the first
  ``config.max_position_embeddings`` tokens, drop matches past the cutoff,
  warn loud, count truncated docs. For llama-3.1-8b-base (128K context) vs a
  ~10-15K-token pond paper this never fires, but it stays fail-loud.
- **OOM prevention (carried from ``JacobianLensLM.compute_scores``).**
  ``torch.no_grad()`` wraps the whole trace (nnsight defers the forward to
  ``__exit__``); ``logits_to_keep=1`` stops HF unembedding every position;
  each layer's ``[n_occurrences, d]`` slice is taken *inside* the trace and
  only that slice is ``.save()``d — never the full ``[seq, d]`` residual, for
  any layer; per-document ``empty_cache`` / ``gc.collect``.
"""
from __future__ import annotations

import gc
import re
import warnings

import numpy as np

try:
    import torch
    from nnterp import StandardizedTransformer
    HAS_GPU_DEPS = True
except ImportError:
    print("Warning: PyTorch and/or nnterp not available; RepresentationLM will not run.")
    HAS_GPU_DEPS = False


# ---------------------------------------------------------------------------
# Pure text/tokenizer helpers — no model, unit-testable on their own
# ---------------------------------------------------------------------------


def find_key_term_occurrences(
    text: str, key_terms: list[str]
) -> list[tuple[str, int, int]]:
    """Every whole-word, case-insensitive occurrence of each key term.

    Matches ``term`` and its simple plural ``term + "s"`` only. Whole-word via
    regex ``\\b`` boundaries, so ``pondweed`` / ``respond`` do NOT match
    ``pond`` and ``ponds`` DOES.

    Args:
        text: Document text.
        key_terms: Base terms, e.g. ``["pond", "lake", "wetland"]``.

    Returns:
        ``[(base_term, match_start, match_end), ...]`` sorted by span. The
        label is always the base term, even for a plural surface form.
    """
    if not key_terms:
        raise ValueError("key_terms is empty")
    occurrences: list[tuple[str, int, int]] = []
    for term in key_terms:
        if not term or term != term.strip():
            raise ValueError(f"key term {term!r} is empty or has surrounding whitespace")
        pattern = re.compile(r"\b" + re.escape(term) + r"s?\b", re.IGNORECASE)
        for m in pattern.finditer(text):
            occurrences.append((term, m.start(), m.end()))
    occurrences.sort(key=lambda t: (t[1], t[2]))
    return occurrences


def select_last_subword_index(
    offset_mapping: list[tuple[int, int]], match_start: int, match_end: int
) -> int:
    """Index of the single token whose char interval contains the match's last char.

    For a multi-subword term this is the **last** subword. Special tokens
    (BOS) carry a ``(0, 0)`` offset and are never selected.

    Raises:
        AssertionError: If zero or more than one token contains the last char
            (e.g. an offset_mapping that doesn't line up with ``text``).
    """
    last_char = match_end - 1
    hits = [
        i
        for i, (s, e) in enumerate(offset_mapping)
        if s <= last_char < e
    ]
    assert len(hits) == 1, (
        f"expected exactly one token covering char {last_char} of match "
        f"[{match_start}, {match_end}); found {len(hits)}: {hits}"
    )
    return hits[0]


# ---------------------------------------------------------------------------
# Layer selection — pure, unit-testable on its own
# ---------------------------------------------------------------------------


DEFAULT_LAYERS: list[int] = [0, 8, 16, 24, 32]
"""Default layer list. ``0`` = token embeddings, ``32`` = post-final-norm.

Assumes a 32-block model (Llama-3.1-8B). ``resolve_layers`` raises if a value
exceeds the loaded model's ``n_layers``.
"""


def resolve_layers(layers: list[int], n_layers: int) -> list[int]:
    """Validate a layer list against a model depth; return it sorted ascending.

    Args:
        layers: Requested layer indices. ``0`` is the token-embedding output,
            ``1..n_layers-1`` are post-block residuals, ``n_layers`` is the
            post-final-norm state.
        n_layers: Number of transformer blocks in the model.

    Returns:
        The distinct requested layers, sorted ascending — the order reads must
        be emitted in inside an nnsight trace.

    Raises:
        ValueError: empty list, a non-int entry, an out-of-range entry
            (``< 0`` or ``> n_layers``), or a duplicate.
    """
    if not layers:
        raise ValueError("layers is empty")
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    seen: set[int] = set()
    for L in layers:
        if isinstance(L, bool) or not isinstance(L, (int, np.integer)):
            raise ValueError(f"layer {L!r} is not an int")
        L = int(L)
        if not (0 <= L <= n_layers):
            raise ValueError(
                f"layer {L} out of range for a {n_layers}-block model "
                f"(valid: 0 (embeddings) .. {n_layers} (post-final-norm))"
            )
        if L in seen:
            raise ValueError(f"duplicate layer {L} in {layers}")
        seen.add(L)
    return sorted(seen)


# ---------------------------------------------------------------------------
# RepresentationLM
# ---------------------------------------------------------------------------


class RepresentationLM:
    """Collect per-layer representations for key-term tokens (one pass/doc).

    Args:
        model_name: HF repo id / NNsight model name of a base (non-instruct) model.
        layers: Layer indices to collect (see the module docstring). ``None``
            uses ``DEFAULT_LAYERS`` = ``[0, 8, 16, 24, 32]``. Validated against
            the loaded model's ``n_layers`` and stored sorted+deduped as
            ``self.layers``.
        nnsight_kwargs: Extra kwargs for ``StandardizedTransformer``
            (e.g. ``{"torch_dtype": torch.bfloat16}``).
        hf_cache_dir: HuggingFace cache directory. Currently unused by this
            class (the model is loaded through nnterp/nnsight, which read
            ``HF_HOME`` / ``HF_HUB_CACHE`` from the environment) — accepted for
            signature parity with ``JacobianLensLM`` and future use.
        verbose: Print device setup.
    """

    def __init__(
        self,
        model_name: str,
        layers: list[int] | None = None,
        nnsight_kwargs: dict | None = None,
        hf_cache_dir: str | None = None,
        verbose: bool = False,
    ):
        if not HAS_GPU_DEPS:
            raise RuntimeError(
                "RepresentationLM requires torch + nnterp (install the 'gpu' extra)."
            )
        self.model_name = model_name
        self.hf_cache_dir = hf_cache_dir
        self.verbose = verbose

        self._setup_devices()

        self.llm = StandardizedTransformer(
            model_name, enable_attention_probs=False, **(nnsight_kwargs or {})
        )
        self.tokenizer = self.llm.tokenizer
        # No-op for the Llama-3.1-8B fast tokenizer (attr is None); real effect
        # only for tokenizers that expose the flag. BOS is otherwise added by
        # add_special_tokens=True in _tokenize().
        if getattr(self.tokenizer, "add_bos_token", None) is not None:
            self.tokenizer.add_bos_token = True

        self.n_layers = len(self.llm.model.layers)
        self.hidden_size = int(self.llm.config.hidden_size)
        self.max_position_embeddings = int(self.llm.config.max_position_embeddings)

        self.layers = resolve_layers(
            DEFAULT_LAYERS if layers is None else layers, self.n_layers
        )
        if self.verbose:
            print(f"Collecting layers: {self.layers} (n_layers={self.n_layers})")

        # Populated by collect().
        self.n_truncated: int = 0
        self.truncated_docs: list[dict] = []

    def _setup_devices(self):
        """LLM on cuda:0, tensors on cuda:1 if >=2 GPUs, else share; CPU fallback.

        Copied verbatim from ``JacobianLensLM._setup_devices``.
        """
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus >= 2:
                self.llm_device = torch.device("cuda:0")
                self.tensor_device = torch.device("cuda:1")
                if self.verbose:
                    print(f"Using {num_gpus} GPUs: LLM on cuda:0, tensors on cuda:1")
            else:
                self.llm_device = torch.device("cuda:0")
                self.tensor_device = torch.device("cuda:0")
                if self.verbose:
                    print("Using single GPU: cuda:0")
        else:
            self.llm_device = torch.device("cpu")
            self.tensor_device = torch.device("cpu")
            if self.verbose:
                print("No GPU available, using CPU")

    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Raw-text tokenization with BOS and offsets. No chat template."""
        enc = self.tokenizer(
            text, return_offsets_mapping=True, add_special_tokens=True
        )
        input_ids = list(enc["input_ids"])
        offsets = [tuple(o) for o in enc["offset_mapping"]]
        assert len(input_ids) == len(offsets)
        return input_ids, offsets

    def _read_point(self, layer: int):
        """Trace-time tensor for a layer (see the module docstring for the map).

        Must be called in ascending ``layer`` order inside a single trace —
        nnsight forbids reading an envoy output after execution has passed it.
        """
        if layer == 0:
            return self.llm.token_embeddings
        if layer == self.n_layers:
            return self.llm.ln_final.output
        return self.llm.model.layers[layer - 1].output[0]

    def _collect_reps(
        self, input_ids: list[int], token_indices: list[int]
    ) -> dict[int, np.ndarray]:
        """Hidden states at ``token_indices`` for every layer in ``self.layers``.

        Returns ``{layer: (k, d) float32}``. One prefill-only trace; each
        layer's ``[k, d]`` slice is taken inside the trace so the full
        ``[seq, d]`` residual is never retained or moved, for any layer. Read
        points are emitted in ascending-layer order (``self.layers`` is sorted).
        """
        assert token_indices, "no token indices to collect"
        assert max(token_indices) < len(input_ids)

        saved: dict[int, "torch.Tensor"] = {}
        with torch.no_grad(), self.llm.trace(input_ids, logits_to_keep=1):
            for layer in self.layers:  # ascending — resolve_layers sorted it
                h = self._read_point(layer)
                # Best-effort guard: logits_to_keep must not truncate the
                # per-position axis. nnsight resolves shapes eagerly during
                # trace in this version (same as JacobianLensLM's `.ndim` use).
                # If this line ever raises for proxy-semantics reasons rather
                # than a real mismatch it is safe to delete — the
                # `seq[token_indices, :]` gather raises IndexError on a
                # collapsed sequence dim and the post-trace `reps.shape`
                # assert catches anything else.
                assert h.shape[-2] == len(input_ids), (
                    f"layer {layer}: read-point seq dim {h.shape[-2]} != "
                    f"n_tokens {len(input_ids)} — read point / logits_to_keep "
                    "is wrong."
                )
                seq = h[0] if h.ndim == 3 else h
                saved[layer] = seq[token_indices, :].detach().to(torch.float32).save()

        reps: dict[int, np.ndarray] = {}
        for layer, picked in saved.items():
            arr = np.asarray(picked.detach().cpu().numpy(), dtype=np.float32)
            assert arr.shape == (len(token_indices), self.hidden_size), (
                layer, arr.shape,
            )
            reps[layer] = arr

        del saved
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        assert set(reps) == set(self.layers)
        return reps

    # ------------------------------------------------------------------

    def verify_read_point(self, text: str, key_terms: list[str], n_check: int = 5) -> None:
        """Smoke-only gate on the per-layer read points + determinism.

        The layer→module map rests on nnterp's renaming, not a measurement.
        On one document, at ``n_check`` key-term positions, this checks:

        1. **Determinism.** Two identical forward passes must be bitwise equal
           at every requested layer (bf16 kernels do not guarantee this).
        2. **Adjacent layers distinct.** No two consecutive requested layers
           may be ``np.allclose`` — catches an off-by-one or a read point that
           silently returns the same module twice.
        3. **Final layer is post-final-norm, not the pre-norm residual**
           (only when ``n_layers`` is requested):
             a. ``ln_final.output`` differs from ``layers[-1].output``.
             b. **Exact RMSNorm signature** — ``rms(ln_final.output / weight)``
                ≈ 1 per row (the defining identity ``y = w * x / rms(x)``).
             c. **Off-by-one pin** — ``ln_final.output`` equals the model's
                RMSNorm recomputed in numpy from ``layers[-1].output``
                (median relative error < 5e-2; bf16 kernel vs float32 numpy,
                so not exact — but an off-by-one is wildly off).

        The always-read pre-final-norm residual (``layers[n_layers-1].output``)
        backs 3a/3c even when layer ``n_layers-1`` is not itself requested.
        All reads are emitted in forward order (embeddings → blocks ascending →
        ln_final) and no envoy is re-invoked. Raises on any failure.
        """
        input_ids, offsets = self._tokenize(text)
        occ = find_key_term_occurrences(text, key_terms)
        if not occ:
            raise RuntimeError("verify_read_point: no key-term occurrences in the sample document")
        idxs: list[int] = []
        for _term, ms, me in occ:
            idxs.append(select_last_subword_index(offsets, ms, me))
            if len(idxs) >= n_check:
                break

        want_lnf = self.n_layers in self.layers
        pre_final_block = self.n_layers - 1  # 0-indexed block, always read

        def _run() -> dict:
            saved: dict = {}
            with torch.no_grad(), self.llm.trace(input_ids, logits_to_keep=1):
                if 0 in self.layers:
                    e = self.llm.token_embeddings
                    e = e[0] if e.ndim == 3 else e
                    saved[0] = e[idxs, :].detach().to(torch.float32).save()
                block_reads = sorted(
                    {L - 1 for L in self.layers if 1 <= L <= self.n_layers - 1}
                    | {pre_final_block}
                )
                for bi in block_reads:  # ascending
                    h = self.llm.model.layers[bi].output[0]
                    h = h[0] if h.ndim == 3 else h
                    saved[("block", bi)] = h[idxs, :].detach().to(torch.float32).save()
                if want_lnf:
                    lo = self.llm.ln_final.output
                    lo = lo[0] if lo.ndim == 3 else lo
                    saved[self.n_layers] = lo[idxs, :].detach().to(torch.float32).save()
            out = {
                k: np.asarray(v.detach().cpu().numpy(), dtype=np.float32)
                for k, v in saved.items()
            }
            del saved
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return out

        r1 = _run()
        r2 = _run()

        def _layer_arr(run: dict, L: int) -> np.ndarray:
            if L == 0:
                return run[0]
            if L == self.n_layers:
                return run[self.n_layers]
            return run[("block", L - 1)]

        per1 = {L: _layer_arr(r1, L) for L in self.layers}
        per2 = {L: _layer_arr(r2, L) for L in self.layers}
        pre_final1 = r1[("block", pre_final_block)]

        # Diagnostics first, so a failed gate leaves an interpretable log.
        lines = ["verify_read_point diagnostics:"]
        for L in self.layers:
            lines.append(
                f"  layer {L:>2d}: mean row L2 "
                f"{np.linalg.norm(per1[L], axis=1).mean():.4g}, "
                f"max|pass1-pass2| {np.abs(per1[L] - per2[L]).max():.4g}"
            )
        lines.append(
            f"  pre-final-norm residual (layers[{pre_final_block}].output) "
            f"mean row L2 {np.linalg.norm(pre_final1, axis=1).mean():.4g}"
        )
        print("\n".join(lines))

        # (1) determinism, every requested layer
        for L in self.layers:
            det_diff = float(np.abs(per1[L] - per2[L]).max())
            assert np.array_equal(per1[L], per2[L]), (
                f"verify_read_point: layer {L} differs across two identical "
                f"forward passes (max abs diff {det_diff:.4g}) — collection is "
                "non-deterministic; investigate before interpreting the run."
            )

        # (2) adjacent requested layers distinct
        for a, b in zip(self.layers, self.layers[1:]):
            assert not np.allclose(per1[a], per1[b], atol=1e-3), (
                f"verify_read_point: layer {a} and layer {b} reads are "
                "~identical — read points collapsed (off-by-one or duplicate)."
            )

        # (3) final read point is post-final-norm
        if want_lnf:
            post1 = per1[self.n_layers]
            assert not np.allclose(post1, pre_final1, atol=1e-3), (
                "verify_read_point: ln_final.output == layers[-1].output — the "
                "final read point is the pre-norm residual, NOT post-final-norm."
            )
            norm_mod = getattr(self.llm.ln_final, "_module", None)
            if norm_mod is None or not hasattr(norm_mod, "weight"):
                raise RuntimeError(
                    "verify_read_point: cannot read the final-norm weight "
                    "(self.llm.ln_final._module.weight) for the RMSNorm check."
                )
            w = norm_mod.weight.detach().float().cpu().numpy()
            assert w.shape == (self.hidden_size,), w.shape
            eps = float(
                getattr(norm_mod, "variance_epsilon", None)
                or getattr(norm_mod, "eps", None)
                or 1e-5
            )
            row_rms = np.sqrt(np.mean((post1 / w) ** 2, axis=1))
            assert np.all((row_rms > 0.9) & (row_rms < 1.1)), (
                f"verify_read_point: rms(ln_final.output / final_norm_weight) per "
                f"row = {np.array2string(row_rms, precision=4)} — not ≈ 1, so "
                "ln_final.output is not the model's RMSNorm output."
            )
            rms = np.sqrt(np.mean(pre_final1 ** 2, axis=1, keepdims=True) + eps)
            recomputed = w * pre_final1 / rms
            rel = np.abs(recomputed - post1) / (np.abs(post1) + 1e-6)
            assert float(np.median(rel)) < 5e-2, (
                f"verify_read_point: ln_final.output != RMSNorm(layers[-1].output) "
                f"recomputed in numpy (median rel err {np.median(rel):.3g}) — the "
                "final read point is not the norm of the true last residual "
                "(off-by-one in the layer→module map?)."
            )
            print(
                f"verify_read_point: OK — {len(self.layers)} read points, all "
                f"bitwise-deterministic; adjacent layers distinct; final layer "
                f"passes RMSNorm identity (row rms ∈ "
                f"[{row_rms.min():.4f}, {row_rms.max():.4f}]) and matches numpy "
                f"RMSNorm of layers[{pre_final_block}].output "
                f"(median rel err {np.median(rel):.2g})."
            )
        else:
            print(
                f"verify_read_point: OK — {len(self.layers)} read points, all "
                "bitwise-deterministic; adjacent layers distinct. "
                f"(layer {self.n_layers} not requested — final-norm identity "
                "check skipped.)"
            )

    # ------------------------------------------------------------------

    def collect(
        self,
        documents: dict[str, str] | list[str],
        key_terms: list[str],
    ) -> dict[str, np.ndarray]:
        """Collect key-term representations across a document collection.

        Args:
            documents: ``{doc_id: text}`` or a list of texts (ids become
                ``"0"``, ``"1"``, …).
            key_terms: Base terms to match (case-insensitive, whole-word,
                simple plural).

        Returns:
            Dict (row count ``n`` for every parallel array):
              - ``representations``: ``{layer: float32 [n, hidden_size]}`` — one
                array per entry of ``self.layers``, same row order as the
                provenance arrays below.
              - ``layers``: ``int64 [k]`` — ``self.layers``, sorted ascending.
              - ``labels``: ``str [n]`` — the base key term for each row
              - ``doc_ids``: ``str [n]``
              - ``char_starts`` / ``char_ends``: ``int64 [n]`` — the regex
                match span in the source document
              - ``token_indices``: ``int64 [n]`` — collected token position
                (post-truncation, into the BOS-prefixed sequence)
        """
        if not key_terms:
            raise ValueError("key_terms is empty")
        if isinstance(documents, list):
            documents = {str(i): t for i, t in enumerate(documents)}
        if not documents:
            raise ValueError("documents is empty")

        self.n_truncated = 0
        self.truncated_docs = []

        rep_blocks: dict[int, list[np.ndarray]] = {L: [] for L in self.layers}
        labels: list[str] = []
        doc_ids: list[str] = []
        char_starts: list[int] = []
        char_ends: list[int] = []
        token_indices: list[int] = []

        for doc_id, text in documents.items():
            input_ids, offsets = self._tokenize(text)
            # Full offsets are kept for token selection; only input_ids passed
            # to the forward pass are truncated. A match whose last-subword
            # token index lands in the dropped tail is skipped below.
            n_full = len(input_ids)
            cutoff = n_full
            if n_full > self.max_position_embeddings:
                cutoff = self.max_position_embeddings
                dropped = n_full - cutoff
                warnings.warn(
                    f"[RepresentationLM] doc {doc_id!r}: {n_full} tokens "
                    f"> max_position_embeddings {self.max_position_embeddings}; "
                    f"tail-truncating, dropping {dropped} tokens.",
                    stacklevel=2,
                )
                input_ids = input_ids[:cutoff]
                self.n_truncated += 1
                self.truncated_docs.append(
                    {"doc_id": doc_id, "n_tokens": n_full, "dropped": dropped}
                )

            occurrences = find_key_term_occurrences(text, key_terms)

            doc_tok_idx: list[int] = []
            doc_meta: list[tuple[str, int, int]] = []
            for term, ms, me in occurrences:
                surface = text[ms:me].lower()
                assert surface in {term, term + "s"}, (
                    f"doc {doc_id!r}: match text {surface!r} not {term!r}/{term + 's'!r}"
                )
                try:
                    tok_i = select_last_subword_index(offsets, ms, me)
                except AssertionError as e:
                    raise AssertionError(
                        f"doc {doc_id!r}, {term!r} occurrence at [{ms}, {me}] "
                        f"({text[ms:me]!r}): {e}"
                    ) from e
                if tok_i >= cutoff:
                    # Occurrence fell in the truncated tail.
                    continue
                doc_tok_idx.append(tok_i)
                doc_meta.append((term, ms, me))

            if not doc_tok_idx:
                continue

            reps = self._collect_reps(input_ids, doc_tok_idx)
            for L in self.layers:
                assert reps[L].shape == (len(doc_tok_idx), self.hidden_size)
                rep_blocks[L].append(reps[L])
            for (term, ms, me), ti in zip(doc_meta, doc_tok_idx):
                labels.append(term)
                doc_ids.append(str(doc_id))
                char_starts.append(int(ms))
                char_ends.append(int(me))
                token_indices.append(int(ti))

        if not labels:
            raise RuntimeError(
                "No key-term occurrences found in any document — nothing to collect."
            )

        representations = {
            L: np.concatenate(rep_blocks[L], axis=0).astype(np.float32)
            for L in self.layers
        }
        n = len(labels)

        out = {
            "representations": representations,
            "layers": np.asarray(self.layers, dtype=np.int64),
            "labels": np.asarray(labels, dtype=object).astype("U"),
            "doc_ids": np.asarray(doc_ids, dtype=object).astype("U"),
            "char_starts": np.asarray(char_starts, dtype=np.int64),
            "char_ends": np.asarray(char_ends, dtype=np.int64),
            "token_indices": np.asarray(token_indices, dtype=np.int64),
        }

        # Boundary asserts.
        for L in self.layers:
            assert representations[L].shape == (n, self.hidden_size), (
                L, representations[L].shape,
            )
            assert np.isfinite(representations[L]).all(), f"non-finite reps at layer {L}"
        assert len(out["layers"]) == len(self.layers)
        for k, v in out.items():
            if k in ("representations", "layers"):
                continue
            assert len(v) == n, f"{k}: len {len(v)} != n {n}"
        triples = list(zip(out["doc_ids"].tolist(), out["char_starts"].tolist(), out["char_ends"].tolist()))
        assert len(set(triples)) == n, (
            f"{n - len(set(triples))} rows share a (doc_id, char_start, char_end) "
            "— overlapping key terms?"
        )
        return out
