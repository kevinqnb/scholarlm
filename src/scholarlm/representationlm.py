"""Key-term representation collection from a non-instruction-tuned base LLM.

``RepresentationLM`` passes each document through a base model as raw
next-token prediction (no chat template, BOS added, one forward pass per
document) and reads the **last-layer, post-final-norm** hidden state (the
vector the unembedding sees) at the token positions that fall inside a
whole-word occurrence of a supplied key term.

Design decisions (see ``notes/scholarlm/builds/2026-08-31-representation-lm-01.md``):

- **Read point.** ``nnterp.StandardizedTransformer`` exposes
  ``self.llm.ln_final.output`` as a real per-position tensor inside a trace
  (nnterp validates exactly this at model-load time). That is the
  post-final-norm residual — read it directly, do not re-apply
  ``model.model.norm``.
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
  the ``[n_occurrences, d]`` slice is taken *inside* the trace and only that
  slice is ``.save()``d; per-document ``empty_cache`` / ``gc.collect``.
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
# RepresentationLM
# ---------------------------------------------------------------------------


class RepresentationLM:
    """Collect last-layer post-final-norm representations for key-term tokens.

    Args:
        model_name: HF repo id / NNsight model name of a base (non-instruct) model.
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

    def _collect_reps(self, input_ids: list[int], token_indices: list[int]) -> np.ndarray:
        """Post-final-norm hidden states at ``token_indices``. Shape ``(k, d)``.

        Runs one prefill-only trace. The ``[k, d]`` slice is taken inside the
        trace so the full ``[seq, d]`` residual is never retained or moved.
        """
        assert token_indices, "no token indices to collect"
        assert max(token_indices) < len(input_ids)

        with torch.no_grad(), self.llm.trace(input_ids, logits_to_keep=1):
            ln_out = self.llm.ln_final.output
            # Best-effort guard: logits_to_keep must not truncate the norm
            # output. nnsight resolves shapes eagerly during trace in this
            # version (same as JacobianLensLM's `.ndim` use). If this line ever
            # raises for proxy-semantics reasons rather than a real mismatch,
            # it is safe to delete — the real guards still fire: the
            # `seq[token_indices, :]` gather below raises IndexError on a
            # collapsed sequence dim, and the post-trace `reps.shape` assert
            # catches anything else.
            assert ln_out.shape[-2] == len(input_ids), (
                f"ln_final.output seq dim {ln_out.shape[-2]} != n_tokens "
                f"{len(input_ids)} — read point / logits_to_keep is wrong."
            )
            seq = ln_out[0] if ln_out.ndim == 3 else ln_out
            picked = seq[token_indices, :].detach().to(torch.float32).save()

        reps = np.asarray(picked.detach().cpu().numpy(), dtype=np.float32)

        del picked
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        assert reps.shape == (len(token_indices), self.hidden_size), reps.shape
        return reps

    # ------------------------------------------------------------------

    def verify_read_point(self, text: str, key_terms: list[str], n_check: int = 5) -> None:
        """Smoke-only gate on the ``ln_final.output`` read point + determinism.

        The "``ln_final.output`` is the post-final-norm state" decision rests on
        nnterp's module renaming, not a measurement. On one document this
        checks, at ``n_check`` key-term positions:

        1. **Not the pre-norm residual.** ``ln_final.output`` must differ
           substantially from ``layers[-1].output`` (the raw last-block
           residual) — a wrong read point that returned the residual would
           fail here.
        2. **Exact RMSNorm signature.** Dividing ``ln_final.output`` elementwise
           by the final-norm weight vector must yield a vector whose per-row
           RMS is ≈ 1 (that is the defining identity of RMSNorm:
           ``y = w * x / rms(x)``). This is exact, not heuristic, and fails
           hard if the collected vector is the pre-norm residual or any other
           layer's output.
        3. **Determinism.** Two identical forward passes must be bitwise equal
           (bf16 kernels do not guarantee this).

        Every module output is read in forward order (layer 31 → ln_final) and
        no envoy is re-invoked, to stay clear of nnsight's execution-ordering
        rules. Raises on any failure — this is a gate, not a diagnostic.
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

        def _run() -> tuple[np.ndarray, np.ndarray]:
            with torch.no_grad(), self.llm.trace(input_ids, logits_to_keep=1):
                pre = self.llm.model.layers[self.n_layers - 1].output[0]
                pre_seq = pre[0] if pre.ndim == 3 else pre
                pre_s = pre_seq[idxs, :].detach().to(torch.float32).save()

                ln_out = self.llm.ln_final.output
                ln_seq = ln_out[0] if ln_out.ndim == 3 else ln_out
                post_s = ln_seq[idxs, :].detach().to(torch.float32).save()
            out = (
                np.asarray(post_s.detach().cpu().numpy(), dtype=np.float32),
                np.asarray(pre_s.detach().cpu().numpy(), dtype=np.float32),
            )
            del post_s, pre_s
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return out

        post1, pre1 = _run()
        post2, _ = _run()

        norm_mod = getattr(self.llm.ln_final, "_module", None)
        if norm_mod is None or not hasattr(norm_mod, "weight"):
            raise RuntimeError(
                "verify_read_point: cannot read the final-norm weight "
                "(self.llm.ln_final._module.weight) for the RMSNorm check."
            )
        w = norm_mod.weight.detach().float().cpu().numpy()
        assert w.shape == (self.hidden_size,), w.shape
        row_rms = np.sqrt(np.mean((post1 / w) ** 2, axis=1))
        pre_norms = np.linalg.norm(pre1, axis=1)
        post_norms = np.linalg.norm(post1, axis=1)

        # Diagnostics first, so a failed gate leaves an interpretable log.
        print(
            "verify_read_point diagnostics:\n"
            f"  weight: shape {w.shape}, mean {w.mean():.4g}, "
            f"min {w.min():.4g}, max {w.max():.4g}\n"
            f"  pre-norm  (layers[{self.n_layers - 1}].output) row L2: "
            f"{np.array2string(pre_norms, precision=2)}\n"
            f"  post-norm (ln_final.output)        row L2: "
            f"{np.array2string(post_norms, precision=2)}\n"
            f"  rms(post / weight) per row: {np.array2string(row_rms, precision=4)}\n"
            f"  max|post pass1 - pass2|: {np.abs(post1 - post2).max():.4g}"
        )

        # (1) not the pre-norm residual
        assert not np.allclose(post1, pre1, atol=1e-3), (
            "verify_read_point: ln_final.output == layers[-1].output — the "
            "collected vector is the pre-norm residual, NOT post-final-norm."
        )

        # (2) exact RMSNorm signature: rms(ln_final.output / weight) ≈ 1 per row
        assert np.all((row_rms > 0.9) & (row_rms < 1.1)), (
            f"verify_read_point: rms(ln_final.output / final_norm_weight) per row "
            f"= {np.array2string(row_rms, precision=4)} — not ≈ 1, so ln_final.output "
            "is not the model's RMSNorm output. Read point is wrong."
        )

        # (3) determinism
        det_diff = float(np.abs(post1 - post2).max())
        assert np.array_equal(post1, post2), (
            f"verify_read_point: two identical forward passes differ (max abs diff "
            f"{det_diff:.4g}) — collection is non-deterministic; investigate before "
            "interpreting the full run."
        )

        print(
            f"verify_read_point: OK — RMSNorm identity holds "
            f"(row rms of output/weight ∈ [{row_rms.min():.4f}, {row_rms.max():.4f}]); "
            f"row L2 norm pre {pre_norms.mean():.3g} -> post {post_norms.mean():.3g}; "
            f"deterministic across two passes."
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
            Dict of equal-length parallel arrays (row count ``n``):
              - ``representations``: ``float32 [n, hidden_size]``
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

        rep_blocks: list[np.ndarray] = []
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
            rep_blocks.append(reps)
            for (term, ms, me), ti in zip(doc_meta, doc_tok_idx):
                labels.append(term)
                doc_ids.append(str(doc_id))
                char_starts.append(int(ms))
                char_ends.append(int(me))
                token_indices.append(int(ti))

        if not rep_blocks:
            raise RuntimeError(
                "No key-term occurrences found in any document — nothing to collect."
            )

        representations = np.concatenate(rep_blocks, axis=0).astype(np.float32)
        n = representations.shape[0]

        out = {
            "representations": representations,
            "labels": np.asarray(labels, dtype=object).astype("U"),
            "doc_ids": np.asarray(doc_ids, dtype=object).astype("U"),
            "char_starts": np.asarray(char_starts, dtype=np.int64),
            "char_ends": np.asarray(char_ends, dtype=np.int64),
            "token_indices": np.asarray(token_indices, dtype=np.int64),
        }

        # Boundary asserts.
        assert representations.shape == (n, self.hidden_size)
        for k, v in out.items():
            assert len(v) == n, f"{k}: len {len(v)} != n {n}"
        triples = list(zip(out["doc_ids"].tolist(), out["char_starts"].tolist(), out["char_ends"].tolist()))
        assert len(set(triples)) == n, (
            f"{n - len(set(triples))} rows share a (doc_id, char_start, char_end) "
            "— overlapping key terms?"
        )
        return out
