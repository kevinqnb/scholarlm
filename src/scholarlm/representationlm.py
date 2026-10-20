"""Key-term representation collection from a causal or bidirectional language model.

``RepresentationLM`` finds whole-word occurrences of key terms in OCR'd papers and
returns the hidden state at the term's tokens, for a chosen list of layers. Text is
fed as raw text (no chat template; BOS/CLS added), one unpadded sequence per forward
pass, and one pass yields every requested layer.

Pipeline per document: clean the OCR text -> find key-term matches -> (optionally)
cut a paragraph window around each match -> tokenize -> forward pass -> pool the
term's token states.

- **Architectures.** ``"causal"`` loads through nnterp/nnsight. ``"bidirectional"``
  loads through plain HF ``AutoModel`` with ``output_hidden_states=True`` and is
  restricted to the model types in ``BIDIRECTIONAL_MODEL_TYPES`` (ModernBERT; nnterp
  does not support it). LLM2Vec is not supported: its bidirectional attention comes
  from the ``llm2vec`` package, so a stock loader would silently give causal states.
- **Layers.** ``layers`` is a list of ints in ``0..n_layers``, same meaning for both
  architectures:

  - ``0``: embedding output, before block 0.
  - ``1..n_layers-1``: residual stream after that many blocks (pre-norm).
  - ``n_layers``: post-final-norm state (what the unembedding / ``last_hidden_state``
    holds).

  Row norms are not comparable across the pre-/post-norm boundary. The causal path
  also reads ``token_embeddings`` (raw) for layer 0, whereas ModernBERT's layer 0 is
  the embedding output after its LayerNorm.
- **Matching.** Case-insensitive, whole-word, term or term + ``"s"``. A token belongs
  to a match when its character interval overlaps the match span (BPE intervals
  include the leading space; special tokens have a ``(0, 0)`` offset and never
  match).
- **Pooling.** ``"mean"`` averages the states of all tokens in the match (float64
  accumulation, float32 result); ``"last"`` keeps only the last token. For a
  single-token match the two are bitwise equal.
- **Cleaning.** ``CleaningConfig`` removes tables, references and ``<page>`` tags,
  in that order. Offsets in the output (``char_starts`` / ``char_ends``) index the
  *cleaned* text, not the raw OCR.
- **Context windows.** ``context_radius=r`` runs each match through paragraphs
  ``x-r .. x+r`` only (``x`` = the match's paragraph; paragraphs are blank-line
  separated blocks of the cleaned text). Matches in one paragraph share a pass.
  ``None`` runs the whole cleaned document.
- **Over-length input.** A sequence longer than the model's
  ``max_position_embeddings`` raises (``overflow_policy="error"``). ``"tail_drop"``
  (causal only) keeps the first ``max_position_embeddings`` tokens, drops matches
  past the cutoff, warns, and counts the truncation; it is not allowed for
  bidirectional models because it keeps only matches near the start of each document.
- **Memory.** Everything runs under ``torch.no_grad()``. The causal trace uses
  ``logits_to_keep=1`` and saves only the gathered ``[n_tokens, d]`` slice of each
  layer, never the full ``[seq, d]`` residual.
"""
from __future__ import annotations

import bisect
import gc
import hashlib
import re
import warnings
from dataclasses import dataclass

import numpy as np

from .utils.references import _drop_references

try:
    import torch
    HAS_TORCH = True
except ImportError:
    print("Warning: PyTorch not available; RepresentationLM will not run.")
    HAS_TORCH = False

try:
    from nnterp import StandardizedTransformer
    HAS_NNTERP = True
except ImportError:
    HAS_NNTERP = False

# True when the causal (nnterp) backend can run; the bidirectional backend needs only torch.
HAS_GPU_DEPS = HAS_TORCH and HAS_NNTERP


# ---------------------------------------------------------------------------
# Matching, token selection, pooling — pure, no model needed
# ---------------------------------------------------------------------------


def find_key_term_occurrences(
    text: str, key_terms: list[str]
) -> list[tuple[str, int, int]]:
    """Find every whole-word, case-insensitive occurrence of each key term.

    Matches ``term`` and ``term + "s"`` only, on ``\\b`` word boundaries: ``ponds``
    matches ``pond``; ``pondweed`` and ``respond`` do not.

    Args:
        text: Document text.
        key_terms: Base terms, e.g. ``["pond", "lake", "wetland"]``. Non-empty, and
            no term may be empty or carry leading/trailing whitespace.

    Returns:
        ``[(base_term, match_start, match_end), ...]`` sorted by ``(start, end)``;
        the label is always the base term, even for a plural surface form.

    Raises:
        ValueError: If ``key_terms`` is empty or contains an empty/whitespace-padded term.
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
    """Index of the token whose character interval contains the match's last character.

    Args:
        offset_mapping: Per-token ``(start, end)`` character offsets into the text
            that was tokenized (special tokens are ``(0, 0)`` and never selected).
        match_start: Match start offset (used only in the error message).
        match_end: Match end offset, exclusive.

    Returns:
        The token index (the last subword of the match).

    Raises:
        AssertionError: If zero or several tokens contain the last character
            (offsets do not line up with the text).
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


def select_term_token_indices(
    offset_mapping: list[tuple[int, int]], match_start: int, match_end: int
) -> list[int]:
    """Indices of every token whose character interval overlaps the match.

    Args:
        offset_mapping: Per-token ``(start, end)`` character offsets into the text
            that was tokenized.
        match_start: Match start offset.
        match_end: Match end offset, exclusive.

    Returns:
        Ascending, contiguous, non-empty token indices; the last one equals
        ``select_last_subword_index``. Special tokens ``(0, 0)`` never overlap.

    Raises:
        AssertionError: If no token overlaps, the overlapping run is not contiguous,
            or its last token differs from ``select_last_subword_index``.
    """
    hits = [
        i
        for i, (s, e) in enumerate(offset_mapping)
        if s < match_end and e > match_start
    ]
    assert hits, f"no token overlaps match [{match_start}, {match_end})"
    assert hits == list(range(hits[0], hits[-1] + 1)), (
        f"tokens overlapping match [{match_start}, {match_end}) are not "
        f"contiguous: {hits}"
    )
    assert hits[-1] == select_last_subword_index(offset_mapping, match_start, match_end)
    return hits


def pool_rows(rows: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Mean over consecutive groups of rows.

    Args:
        rows: ``[sum(lengths), d]`` gathered token states, groups stored back to back.
        lengths: ``int [k]`` group sizes, each ``>= 1``.

    Returns:
        ``float32 [k, d]``; row ``j`` is the mean of group ``j`` (accumulated in
        float64). A length-1 group is returned bitwise unchanged.

    Raises:
        AssertionError: If ``lengths`` is not a 1-D array of values ``>= 1`` summing
            to ``rows.shape[0]``, or ``rows`` is not 2-D.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    assert lengths.ndim == 1 and (lengths >= 1).all(), lengths
    assert rows.ndim == 2 and rows.shape[0] == int(lengths.sum()), (rows.shape, lengths.sum())
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    sums = np.add.reduceat(rows.astype(np.float64), starts, axis=0)
    return (sums / lengths[:, None]).astype(np.float32)


# ---------------------------------------------------------------------------
# Text cleaning and paragraph windows — pure, no model needed
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CleaningConfig:
    """Which OCR markup ``clean_ocr_text`` removes (all on by default).

    Attributes:
        remove_references: Cut the references/bibliography section.
        remove_tables: Remove ``<table>...</table>`` blocks.
        remove_page_tags: Remove ``<page number="N">`` / ``</page>`` tags.
        min_kept_fraction_after_references: If set (``0 < x <= 1``), a reference cut
            that keeps less than this fraction of the (table-stripped) text raises
            instead of dropping most of the paper. Requires ``remove_references``.
            ``None`` disables the check.
    """

    remove_references: bool = True
    remove_tables: bool = True
    remove_page_tags: bool = True
    min_kept_fraction_after_references: float | None = None

    def __post_init__(self):
        f = self.min_kept_fraction_after_references
        if f is not None and not (0 < f <= 1):
            raise ValueError(f"min_kept_fraction_after_references must be in (0, 1], got {f!r}")
        if f is not None and not self.remove_references:
            raise ValueError("min_kept_fraction_after_references set but remove_references is False")


_TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
_PAGE_TAG_RE = re.compile(r"</?page\b[^>]*>", re.IGNORECASE)
# One or more blank lines; a line of only spaces/tabs counts as blank.
_PARA_SEP_RE = re.compile(r"\n(?:[ \t]*\n)+")


def clean_ocr_text(text: str, cleaning: CleaningConfig) -> tuple[str, dict]:
    """Remove tables, references and page tags from OCR text.

    Order is tables, references, page tags. Tables go first so a ``<th>References
    </th>`` column header is not taken for the bibliography heading; page tags go
    last because the references cutter needs well-formed ``<page>`` tags. Each
    removed table or tag becomes a blank line (so neighbouring words never merge),
    then blank-line runs collapse to one and the text is stripped. With every
    removal switched off the text is returned untouched. Cleaning cleaned text is a
    no-op.

    Args:
        text: Raw OCR text.
        cleaning: Which removals to apply.

    Returns:
        ``(cleaned_text, info)``. ``info`` keys: ``raw_chars``, ``cleaned_chars``,
        ``n_tables_removed``, ``n_page_tags_removed``, ``reference_heading_found``
        (``None`` if references are not removed; ``False`` means no heading was
        located, so any references are still in the text), and
        ``kept_fraction_after_references`` (``None`` if references are not removed,
        else kept chars / chars before the cut).

    Raises:
        AssertionError: If a ``<table`` or ``<page`` tag survives its own removal
            (e.g. an unclosed ``<table>``).
        ValueError: If the reference cut keeps less than
            ``cleaning.min_kept_fraction_after_references`` of the text.
    """
    info = {
        "raw_chars": len(text),
        "reference_heading_found": None,
        "kept_fraction_after_references": None,
        "n_tables_removed": 0,
        "n_page_tags_removed": 0,
    }
    if not (cleaning.remove_references or cleaning.remove_tables or cleaning.remove_page_tags):
        info["cleaned_chars"] = len(text)
        return text, info

    if cleaning.remove_tables:
        text, info["n_tables_removed"] = _TABLE_RE.subn("\n\n", text)
        assert "<table" not in text.lower() and "</table" not in text.lower(), (
            "a <table> tag survived table removal (unclosed table?)"
        )
    if cleaning.remove_references:
        before = len(text)
        text, ref_info = _drop_references(text)
        info["reference_heading_found"] = bool(ref_info["reference_heading_found"])
        info["kept_fraction_after_references"] = len(text) / before if before else 1.0
        floor = cleaning.min_kept_fraction_after_references
        if floor is not None and info["kept_fraction_after_references"] < floor:
            raise ValueError(
                f"reference cut ({ref_info['reference_cut_shape']!r} at char "
                f"{ref_info['reference_cut']}) keeps only "
                f"{info['kept_fraction_after_references']:.3f} of the text, below "
                f"min_kept_fraction_after_references={floor}"
            )
    if cleaning.remove_page_tags:
        text, info["n_page_tags_removed"] = _PAGE_TAG_RE.subn("\n\n", text)
        assert "<page" not in text.lower() and "</page" not in text.lower(), (
            "a <page> tag survived page-tag removal"
        )
    text = _PARA_SEP_RE.sub("\n\n", text).strip()
    info["cleaned_chars"] = len(text)
    return text, info


def split_paragraphs(text: str) -> list[tuple[int, int]]:
    """Split text into blank-line-separated, non-blank paragraphs.

    Args:
        text: Text to split.

    Returns:
        ``[(start, end), ...]`` character spans of the paragraphs, in order.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _PARA_SEP_RE.finditer(text):
        if m.start() > pos:
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def paragraph_of_span(
    paragraphs: list[tuple[int, int]], start: int, end: int
) -> int:
    """Find the paragraph that contains a character span.

    Args:
        paragraphs: Spans from ``split_paragraphs``.
        start: Span start offset.
        end: Span end offset, exclusive.

    Returns:
        Index into ``paragraphs`` of the paragraph containing ``[start, end)``.

    Raises:
        AssertionError: If the span is not inside a single paragraph.
    """
    i = bisect.bisect_right([s for s, _ in paragraphs], start) - 1
    assert i >= 0 and start >= paragraphs[i][0] and end <= paragraphs[i][1], (
        f"span [{start}, {end}) is not inside a single paragraph"
    )
    return i


def window_span(
    paragraphs: list[tuple[int, int]], index: int, radius: int
) -> tuple[int, int]:
    """Character span of paragraphs ``index-radius .. index+radius``.

    Args:
        paragraphs: Spans from ``split_paragraphs``.
        index: Center paragraph index.
        radius: Paragraphs to include on each side (``>= 0``).

    Returns:
        ``(start, end)`` from the first included paragraph's start to the last one's
        end, clipped at the first/last paragraph of the document.

    Raises:
        AssertionError: If ``radius < 0`` or ``index`` is out of range.
    """
    assert radius >= 0 and 0 <= index < len(paragraphs), (index, radius, len(paragraphs))
    lo = max(index - radius, 0)
    hi = min(index + radius, len(paragraphs) - 1)
    return paragraphs[lo][0], paragraphs[hi][1]


# ---------------------------------------------------------------------------
# Layer selection — pure, no model needed
# ---------------------------------------------------------------------------


DEFAULT_LAYERS: list[int] = [0, 8, 16, 24, 32]
"""Default layers for a 32-block model: embeddings, three mid-depth residuals, and
the post-final-norm state. ``resolve_layers`` raises on a shallower model, so
others must pass ``layers`` explicitly."""


def resolve_layers(layers: list[int], n_layers: int) -> list[int]:
    """Validate a layer list against the model depth.

    Args:
        layers: Requested layer indices in ``0..n_layers`` (see the module docstring
            for their meaning).
        n_layers: Number of transformer blocks in the model.

    Returns:
        The layers sorted ascending, the order reads must be emitted in inside an
        nnsight trace.

    Raises:
        ValueError: If ``layers`` is empty, ``n_layers < 1``, an entry is not an int
            (bool excluded), an entry is outside ``0..n_layers``, or an entry repeats.
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


ARCHITECTURES = ("causal", "bidirectional")
POOLINGS = ("mean", "last")
OVERFLOW_POLICIES = ("error", "tail_drop")
BIDIRECTIONAL_MODEL_TYPES = ("modernbert",)
"""HF ``config.model_type`` values the bidirectional backend supports. Anything else
raises, in particular LLM2Vec checkpoints (see the module docstring)."""


class RepresentationLM:
    """Collect per-layer representations of key-term tokens.

    Args:
        model_name: HF repo id or local path. Causal: a base (non-instruct) model.
            Bidirectional: an encoder whose ``model_type`` is in
            ``BIDIRECTIONAL_MODEL_TYPES``.
        architecture: ``"causal"`` or ``"bidirectional"``. Required, never inferred.
        layers: Layer indices to collect. ``None`` uses ``DEFAULT_LAYERS``, which
            only fits a 32-block model. Stored validated, sorted and de-duplicated
            as ``self.layers``.
        nnsight_kwargs: Extra kwargs for ``StandardizedTransformer`` (causal only).
        hf_kwargs: Extra kwargs for ``AutoModel.from_pretrained`` (bidirectional only).
        hf_cache_dir: HF cache directory; used by the bidirectional backend only
            (nnterp reads ``HF_HOME`` / ``HF_HUB_CACHE`` itself).
        device: Torch device string, e.g. ``"cuda:0"`` or ``"cpu"``. Required for
            bidirectional; must be ``None`` for causal, which picks its own device.
        pooling: ``"mean"`` or ``"last"``; see the module docstring.
        cleaning: ``CleaningConfig``, or ``None`` for no cleaning.
        context_radius: ``None`` for whole documents, else paragraphs on each side
            of a match's paragraph (``>= 0``).
        overflow_policy: ``"error"`` or ``"tail_drop"`` (causal only).
        verbose: Print device and layer setup.

    Attributes:
        layers: Validated layer list.
        n_layers, hidden_size, max_position_embeddings: Read from the model config.
        n_truncated, truncated_docs: Windows truncated by the last ``collect`` under
            ``tail_drop`` and a record of each.
        doc_info: Per-document cleaning diagnostics from the last ``collect``.

    Raises:
        ValueError: On an invalid argument combination, an unsupported
            bidirectional ``model_type``, or invalid ``layers``.
        RuntimeError: If torch (or, for causal, nnterp) is not installed.
    """

    def __init__(
        self,
        model_name: str,
        architecture: str,
        layers: list[int] | None = None,
        nnsight_kwargs: dict | None = None,
        hf_kwargs: dict | None = None,
        hf_cache_dir: str | None = None,
        device: str | None = None,
        pooling: str = "mean",
        cleaning: CleaningConfig | None = CleaningConfig(),
        context_radius: int | None = None,
        overflow_policy: str = "error",
        verbose: bool = False,
    ):
        if architecture not in ARCHITECTURES:
            raise ValueError(f"architecture {architecture!r} not in {ARCHITECTURES}")
        if pooling not in POOLINGS:
            raise ValueError(f"pooling {pooling!r} not in {POOLINGS}")
        if overflow_policy not in OVERFLOW_POLICIES:
            raise ValueError(f"overflow_policy {overflow_policy!r} not in {OVERFLOW_POLICIES}")
        if context_radius is not None and (
            isinstance(context_radius, bool)
            or not isinstance(context_radius, int)
            or context_radius < 0
        ):
            raise ValueError(f"context_radius must be None or an int >= 0, got {context_radius!r}")
        if not HAS_TORCH:
            raise RuntimeError("RepresentationLM requires torch (install the 'gpu' extra).")
        if architecture == "causal":
            if not HAS_NNTERP:
                raise RuntimeError("causal RepresentationLM requires nnterp (install the 'gpu' extra).")
            if device is not None or hf_kwargs is not None:
                raise ValueError("`device` / `hf_kwargs` apply to the bidirectional architecture only")
        else:
            if device is None:
                raise ValueError("bidirectional RepresentationLM needs an explicit `device` (no CPU/GPU inference)")
            if nnsight_kwargs is not None:
                raise ValueError("`nnsight_kwargs` applies to the causal architecture only")
            if overflow_policy == "tail_drop":
                raise ValueError(
                    "tail_drop is causal-only: for a bidirectional model it keeps only "
                    "matches near the start of each document (selection-biased rows)"
                )
        self.model_name = model_name
        self.architecture = architecture
        self.hf_cache_dir = hf_cache_dir
        self.pooling = pooling
        self.cleaning = cleaning
        self.context_radius = context_radius
        self.overflow_policy = overflow_policy
        self.verbose = verbose

        if architecture == "causal":
            self._setup_devices()
            self.llm = StandardizedTransformer(
                model_name, enable_attention_probs=False, **(nnsight_kwargs or {})
            )
            self.tokenizer = self.llm.tokenizer
            # Only tokenizers that expose the flag need it set; others get BOS from
            # add_special_tokens=True in _tokenize().
            if getattr(self.tokenizer, "add_bos_token", None) is not None:
                self.tokenizer.add_bos_token = True
            self.n_layers = len(self.llm.model.layers)
            self.hidden_size = int(self.llm.config.hidden_size)
            self.max_position_embeddings = int(self.llm.config.max_position_embeddings)
        else:
            from transformers import AutoConfig, AutoModel, AutoTokenizer

            config = AutoConfig.from_pretrained(model_name, cache_dir=hf_cache_dir)
            if config.model_type not in BIDIRECTIONAL_MODEL_TYPES:
                raise ValueError(
                    f"model_type {config.model_type!r} is not a validated bidirectional "
                    f"backend {BIDIRECTIONAL_MODEL_TYPES}. LLM2Vec is not supported: its "
                    "bidirectional attention comes from the llm2vec package, and a stock "
                    "loader would silently return causal representations."
                )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=hf_cache_dir)
            assert self.tokenizer.is_fast, "offset_mapping needs a fast tokenizer"
            self.llm_device = torch.device(device)
            self.tensor_device = self.llm_device
            self.model = (
                AutoModel.from_pretrained(model_name, cache_dir=hf_cache_dir, **(hf_kwargs or {}))
                .to(self.llm_device)
                .eval()
            )
            self.n_layers = int(config.num_hidden_layers)
            self.hidden_size = int(config.hidden_size)
            self.max_position_embeddings = int(config.max_position_embeddings)
            assert len(self.model.layers) == self.n_layers

        self.layers = resolve_layers(
            DEFAULT_LAYERS if layers is None else layers, self.n_layers
        )
        if self.verbose:
            print(
                f"Collecting layers: {self.layers} (n_layers={self.n_layers}, "
                f"architecture={architecture}, pooling={pooling}, "
                f"context_radius={context_radius}, overflow={overflow_policy})"
            )

        # Populated by collect().
        self.n_truncated: int = 0
        self.truncated_docs: list[dict] = []
        self.doc_info: dict[str, dict] = {}

    def _setup_devices(self):
        """Choose devices for the causal backend (sets ``llm_device``, ``tensor_device``).

        With two or more GPUs the model sits on ``cuda:0`` and tensors on ``cuda:1``;
        with one GPU both share ``cuda:0``; with none, both fall back to the CPU.
        Takes no inputs and returns nothing.
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
    # Tokenization and forward passes
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Tokenize raw text with special tokens and character offsets.

        Args:
            text: Text to tokenize (no chat template is applied).

        Returns:
            ``(input_ids, offsets)``, equal-length lists; ``offsets[i]`` is token
            ``i``'s ``(start, end)`` in ``text`` (``(0, 0)`` for special tokens).
        """
        enc = self.tokenizer(
            text, return_offsets_mapping=True, add_special_tokens=True
        )
        input_ids = list(enc["input_ids"])
        offsets = [tuple(o) for o in enc["offset_mapping"]]
        assert len(input_ids) == len(offsets)
        return input_ids, offsets

    def _read_point(self, layer: int):
        """Causal backend: the trace-time tensor holding a layer's state.

        Must be called in ascending ``layer`` order inside one trace; nnsight forbids
        reading an envoy output after execution has passed it.

        Args:
            layer: A layer index in ``0..n_layers``.

        Returns:
            The nnsight proxy: token embeddings for ``0``, ``ln_final.output`` for
            ``n_layers``, otherwise the output of block ``layer - 1``.
        """
        if layer == 0:
            return self.llm.token_embeddings
        if layer == self.n_layers:
            return self.llm.ln_final.output
        return self.llm.model.layers[layer - 1].output[0]

    def _gather_causal(
        self, input_ids: list[int], flat: list[int]
    ) -> dict[int, np.ndarray]:
        """Read the states at given token positions with one nnsight trace.

        Args:
            input_ids: One unpadded token sequence.
            flat: Token positions to read, each ``< len(input_ids)``.

        Returns:
            ``{layer: float32 [len(flat), hidden_size]}`` for every layer in
            ``self.layers``.
        """
        saved: dict[int, "torch.Tensor"] = {}
        with torch.no_grad(), self.llm.trace(input_ids, logits_to_keep=1):
            for layer in self.layers:  # ascending order is required (see _read_point)
                h = self._read_point(layer)
                # logits_to_keep must not shrink the per-position axis. If this ever
                # fails for proxy-semantics reasons it can be dropped: the `flat`
                # gather below raises IndexError on a collapsed sequence axis and the
                # shape check in _collect_reps catches anything else.
                assert h.shape[-2] == len(input_ids), (
                    f"layer {layer}: read-point seq dim {h.shape[-2]} != "
                    f"n_tokens {len(input_ids)} — read point / logits_to_keep "
                    "is wrong."
                )
                seq = h[0] if h.ndim == 3 else h
                saved[layer] = seq[flat, :].detach().to(torch.float32).save()

        rows = {
            layer: np.asarray(picked.detach().cpu().numpy(), dtype=np.float32)
            for layer, picked in saved.items()
        }
        del saved
        return rows

    def _forward_bidirectional(self, input_ids: list[int]):
        """Run one unpadded forward pass of the bidirectional model.

        Args:
            input_ids: One token sequence.

        Returns:
            The HF model output, with ``hidden_states`` (``n_layers + 1`` tensors:
            embeddings, then each block's output) and ``last_hidden_state``.
        """
        ids = torch.tensor([input_ids], device=self.llm_device)
        with torch.no_grad():
            out = self.model(input_ids=ids, output_hidden_states=True)
        assert len(out.hidden_states) == self.n_layers + 1, (
            len(out.hidden_states), self.n_layers,
        )
        return out

    def _layer_state(self, out, layer: int):
        """Bidirectional backend: select one layer's states from a forward output.

        Args:
            out: Output of ``_forward_bidirectional``.
            layer: A layer index in ``0..n_layers``.

        Returns:
            Tensor ``[seq, hidden_size]``: ``last_hidden_state`` (post-final-norm)
            for ``n_layers``, else ``hidden_states[layer]``.
        """
        h = out.last_hidden_state if layer == self.n_layers else out.hidden_states[layer]
        return h[0] if h.ndim == 3 else h

    def _gather_bidirectional(
        self, input_ids: list[int], flat: list[int]
    ) -> dict[int, np.ndarray]:
        """Read the states at given token positions with one HF forward pass.

        Args:
            input_ids: One unpadded token sequence.
            flat: Token positions to read, each ``< len(input_ids)``.

        Returns:
            ``{layer: float32 [len(flat), hidden_size]}`` for every layer in
            ``self.layers``.
        """
        out = self._forward_bidirectional(input_ids)
        rows: dict[int, np.ndarray] = {}
        for layer in self.layers:
            seq = self._layer_state(out, layer)
            assert seq.shape[0] == len(input_ids), (layer, seq.shape, len(input_ids))
            rows[layer] = np.asarray(
                seq[flat, :].detach().to(torch.float32).cpu().numpy(), dtype=np.float32
            )
        del out
        return rows

    def _collect_reps(
        self, input_ids: list[int], token_groups: list[list[int]]
    ) -> dict[int, np.ndarray]:
        """Pooled states of each token group, for every layer in ``self.layers``.

        Args:
            input_ids: One unpadded token sequence (special tokens included).
            token_groups: One non-empty list of token positions per occurrence; each
                group's states are averaged (a single-token group is unchanged).

        Returns:
            ``{layer: float32 [len(token_groups), hidden_size]}``.
        """
        assert token_groups and all(token_groups), "empty token group"
        flat = [i for g in token_groups for i in g]
        assert max(flat) < len(input_ids)
        lengths = np.asarray([len(g) for g in token_groups], dtype=np.int64)

        gather = self._gather_causal if self.architecture == "causal" else self._gather_bidirectional
        gathered = gather(input_ids, flat)

        reps: dict[int, np.ndarray] = {}
        for layer, rows in gathered.items():
            assert rows.shape == (len(flat), self.hidden_size), (layer, rows.shape)
            reps[layer] = pool_rows(rows, lengths)
            assert reps[layer].shape == (len(token_groups), self.hidden_size)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        assert set(reps) == set(self.layers)
        return reps

    # ------------------------------------------------------------------
    # Text preparation
    # ------------------------------------------------------------------

    def prepare_documents(
        self, documents: dict[str, str]
    ) -> tuple[dict[str, str], dict[str, dict]]:
        """Clean every document per ``self.cleaning``.

        ``collect`` calls this itself; call it directly to see the exact text that
        ``collect`` will search (e.g. to predict row counts).

        Args:
            documents: ``{doc_id: raw OCR text}``.

        Returns:
            ``(cleaned, info)`` keyed by ``str(doc_id)``. ``info[doc_id]`` holds the
            ``clean_ocr_text`` diagnostics plus ``cleaned_sha256``, the hash of the
            cleaned text (offsets in ``collect`` output index that text).

        Raises:
            AssertionError, ValueError: From ``clean_ocr_text``, prefixed with the
                offending document id.
        """
        cleaned: dict[str, str] = {}
        info: dict[str, dict] = {}
        for doc_id, text in documents.items():
            if self.cleaning is None:
                t, i = text, {"raw_chars": len(text), "cleaned_chars": len(text),
                              "reference_heading_found": None,
                              "kept_fraction_after_references": None,
                              "n_tables_removed": 0, "n_page_tags_removed": 0}
            else:
                try:
                    t, i = clean_ocr_text(text, self.cleaning)
                except (AssertionError, ValueError) as e:
                    raise type(e)(f"doc {doc_id!r}: {e}") from e
            i["cleaned_sha256"] = hashlib.sha256(t.encode("utf-8")).hexdigest()
            cleaned[str(doc_id)] = t
            info[str(doc_id)] = i
        assert len(cleaned) == len(documents), "document ids collide after str()"
        return cleaned, info

    # ------------------------------------------------------------------
    # Smoke-test gates
    # ------------------------------------------------------------------

    def verify_read_point(self, text: str, key_terms: list[str], n_check: int = 5) -> None:
        """Smoke-test gate: check the layer read points and causal/bidirectional behaviour.

        Cleans ``text`` as ``collect`` does and, if ``context_radius`` is set, cuts it
        to the window around the first match, so the check runs on input ``collect``
        could also see. Then runs the architecture's read-point checks and
        ``_check_future_dependence``. Prints diagnostics.

        Args:
            text: Raw OCR text containing at least one key term.
            key_terms: Terms to match.
            n_check: Maximum number of matches used by the read-point checks.

        Raises:
            ValueError: If the sample is longer than ``max_position_embeddings``.
            RuntimeError: If the sample has no match, or no usable text after the
                first match for the future-dependence check.
            AssertionError: If any check fails.
        """
        cleaned, _ = self.prepare_documents({"verify": text})
        text = cleaned["verify"]
        if self.context_radius is not None:
            occ = find_key_term_occurrences(text, key_terms)
            if not occ:
                raise RuntimeError("verify_read_point: no key-term occurrences in the sample document")
            paras = split_paragraphs(text)
            a, b = window_span(paras, paragraph_of_span(paras, occ[0][1], occ[0][2]), self.context_radius)
            text = text[a:b]
        n_tok = len(self._tokenize(text)[0])
        if n_tok > self.max_position_embeddings:
            raise ValueError(
                f"verify_read_point: sample is {n_tok} tokens > max_position_embeddings "
                f"{self.max_position_embeddings}; set a smaller context_radius."
            )
        if self.architecture == "causal":
            self._verify_causal_read_points(text, key_terms, n_check)
        else:
            self._verify_bidirectional_read_points(text, key_terms, n_check)
        self._check_future_dependence(text, key_terms)

    def _first_occurrence_last_token(
        self, text: str, key_terms: list[str]
    ) -> tuple[list[int], int]:
        """Tokenize ``text`` and locate its first key-term match.

        Args:
            text: Text containing at least one key term.
            key_terms: Terms to match.

        Returns:
            ``(input_ids, last_token_index)`` where the index is the last subword of
            the first match.

        Raises:
            RuntimeError: If there is no match.
        """
        input_ids, offsets = self._tokenize(text)
        occ = find_key_term_occurrences(text, key_terms)
        if not occ:
            raise RuntimeError("verify_read_point: no key-term occurrences in the sample document")
        _t, ms, me = occ[0]
        return input_ids, select_last_subword_index(offsets, ms, me)

    def _check_future_dependence(self, text: str, key_terms: list[str]) -> None:
        """Check that later tokens affect the first match only if the model is bidirectional.

        Reverses the tokens after the first match (same length, same prefix) and
        compares the match's last-token state at every layer. Causal: states must be
        bitwise unchanged. Bidirectional: every layer ``>= 1`` must change; layer 0
        is context-free and must not. Prints the per-layer max absolute difference.

        Args:
            text: Cleaned text containing a key term followed by at least two
                distinct tokens (a bidirectional model's trailing special token is
                left in place).
            key_terms: Terms to match.

        Raises:
            RuntimeError: If there is no usable text after the first match.
            AssertionError: If a layer behaves contrary to the architecture.
        """
        ids, last = self._first_occurrence_last_token(text, key_terms)
        end = len(ids) if self.architecture == "causal" else len(ids) - 1  # keep trailing [SEP]
        tail = ids[last + 1:end]
        if len(tail) < 2 or tail == tail[::-1]:
            raise RuntimeError(
                "verify_read_point: the sample document has no usable text after "
                "its first key term for the future-dependence check"
            )
        perturbed = ids[:last + 1] + tail[::-1] + ids[end:]
        assert len(perturbed) == len(ids) and perturbed != ids
        a = self._collect_reps(ids, [[last]])
        b = self._collect_reps(perturbed, [[last]])
        diffs = {L: float(np.abs(a[L] - b[L]).max()) for L in self.layers}
        print("future-dependence max|diff| by layer: " + ", ".join(f"{L}:{d:.4g}" for L, d in diffs.items()))
        if self.architecture == "causal":
            for L in self.layers:
                assert np.array_equal(a[L], b[L]), (
                    f"verify_read_point: causal layer {L} state at the term changed "
                    f"(max abs diff {diffs[L]:.4g}) when only LATER tokens changed."
                )
        else:
            for L in self.layers:
                if L == 0:
                    assert diffs[0] == 0.0, f"layer 0 should be context-free, diff {diffs[0]:.4g}"
                else:
                    assert diffs[L] > 0.0, (
                        f"verify_read_point: bidirectional layer {L} state at the term did "
                        "NOT change when later tokens changed -- the model is attending "
                        "causally."
                    )
        print("verify_read_point: future-dependence check OK")

    def _verify_bidirectional_read_points(self, text: str, key_terms: list[str], n_check: int) -> None:
        """Check the bidirectional backend's read points.

        Two identical forward passes must agree bitwise at every layer; adjacent
        requested layers must differ; and, if ``n_layers`` is requested,
        ``last_hidden_state`` must differ from ``hidden_states[n_layers]`` and equal
        ``final_norm`` applied to it exactly. Prints a summary.

        Args:
            text: Cleaned text containing at least one key term.
            key_terms: Terms to match.
            n_check: Maximum number of matches whose last-token states are compared.

        Raises:
            RuntimeError: If there is no match.
            AssertionError: If any check fails.
        """
        input_ids, offsets = self._tokenize(text)
        occ = find_key_term_occurrences(text, key_terms)
        if not occ:
            raise RuntimeError("verify_read_point: no key-term occurrences in the sample document")
        idxs = [select_last_subword_index(offsets, ms, me) for _t, ms, me in occ[:n_check]]

        def _run() -> dict[int, np.ndarray]:
            """One forward pass; returns ``{layer: float32 [len(idxs), d]}`` at ``idxs``."""
            out = self._forward_bidirectional(input_ids)
            res = {
                L: np.asarray(self._layer_state(out, L)[idxs, :].detach().to(torch.float32).cpu().numpy())
                for L in self.layers
            }
            if self.n_layers in self.layers:
                pre = out.hidden_states[self.n_layers]
                post = out.last_hidden_state
                assert not torch.equal(pre, post), (
                    "verify_read_point: last_hidden_state == hidden_states[n_layers] "
                    "-- the final norm is missing from the final read point."
                )
                assert torch.equal(self.model.final_norm(pre), post), (
                    "verify_read_point: last_hidden_state != final_norm(hidden_states[n_layers])"
                )
            return res

        r1, r2 = _run(), _run()
        for L in self.layers:
            assert np.array_equal(r1[L], r2[L]), (
                f"verify_read_point: layer {L} differs across two identical forward "
                f"passes (max abs diff {np.abs(r1[L] - r2[L]).max():.4g})"
            )
        for a, b in zip(self.layers, self.layers[1:]):
            assert not np.allclose(r1[a], r1[b], atol=1e-3), (
                f"verify_read_point: layer {a} and layer {b} reads are ~identical"
            )
        print(
            f"verify_read_point: OK -- {len(self.layers)} read points, bitwise-deterministic; "
            "adjacent layers distinct; final layer == final_norm(pre-norm residual)."
        )

    def _verify_causal_read_points(self, text: str, key_terms: list[str], n_check: int = 5) -> None:
        """Check the causal backend's layer-to-module map and determinism.

        The map rests on nnterp's module renaming, so it is measured here at up to
        ``n_check`` match positions:

        1. Two identical forward passes agree bitwise at every requested layer.
        2. Adjacent requested layers differ (catches an off-by-one or a read point
           returning the same module twice).
        3. If ``n_layers`` is requested, it is the post-final-norm state, not the
           pre-norm residual: it differs from block ``n_layers - 1``'s output, its
           rows divided by the norm weight have RMS about 1, and it matches the
           RMSNorm of that block's output recomputed in numpy (median relative error
           below 5e-2; bf16 kernel vs float32 numpy, so not exact).

        Block ``n_layers - 1`` is always read for check 3. All reads are emitted in
        forward order and no envoy is read twice. Prints diagnostics before asserting,
        so a failure leaves an interpretable log.

        Args:
            text: Cleaned text containing at least one key term.
            key_terms: Terms to match.
            n_check: Maximum number of matches whose last-token states are compared.

        Raises:
            RuntimeError: If there is no match, or the final-norm weight is unreadable.
            AssertionError: If any check fails.
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
            """One trace; returns arrays at ``idxs`` keyed ``0``, ``("block", i)``, ``n_layers``."""
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
            """Array for layer ``L`` (``float32 [len(idxs), d]``) from a ``_run`` result."""
            if L == 0:
                return run[0]
            if L == self.n_layers:
                return run[self.n_layers]
            return run[("block", L - 1)]

        per1 = {L: _layer_arr(r1, L) for L in self.layers}
        per2 = {L: _layer_arr(r2, L) for L in self.layers}
        pre_final1 = r1[("block", pre_final_block)]

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
    # Collection
    # ------------------------------------------------------------------

    def collect(
        self,
        documents: dict[str, str] | list[str],
        key_terms: list[str],
    ) -> dict[str, np.ndarray]:
        """Collect key-term representations across a document collection.

        Each document is cleaned, its matches are grouped into forward passes (one
        pass per document, or per paragraph window if ``context_radius`` is set), and
        each match's token states are pooled per layer. Rows are ordered by document,
        then window, then match position. Also sets ``self.doc_info`` (cleaning
        diagnostics) and, under ``tail_drop``, ``self.n_truncated`` /
        ``self.truncated_docs``.

        Args:
            documents: ``{doc_id: raw OCR text}``, or a list of texts (ids become
                ``"0"``, ``"1"``, ...).
            key_terms: Base terms to match (case-insensitive, whole-word, optional
                plural ``s``).

        Returns:
            Dict with ``n`` rows in every per-row array:

            - ``representations``: ``{layer: float32 [n, hidden_size]}``, one per
              entry of ``self.layers``.
            - ``layers``: ``int64 [k]``, ``self.layers``.
            - ``labels``: ``str [n]``, the base key term.
            - ``doc_ids``: ``str [n]``.
            - ``char_starts`` / ``char_ends``: ``int64 [n]``, match span in the
              *cleaned* document text.
            - ``paragraph_indices``: ``int64 [n]``, paragraph of the cleaned text
              holding the match.
            - ``window_starts`` / ``window_ends``: ``int64 [n]``, cleaned-text span
              the forward pass saw (the whole document if ``context_radius`` is None).
            - ``token_indices``: ``int64 [n]``, last pooled token's position in the
              pass's special-token-prefixed sequence (relative to the window).
            - ``token_first_indices``: ``int64 [n]``, first pooled token's position.
            - ``n_pooled_tokens``: ``int64 [n]``, tokens averaged (1 for
              ``pooling="last"``).

        Raises:
            ValueError: If ``key_terms`` or ``documents`` is empty, or a window
                exceeds ``max_position_embeddings`` under ``overflow_policy="error"``.
            RuntimeError: If no document contains a key term.
            AssertionError: If an output array has the wrong shape or a non-finite
                value, a token span cannot be located, or two rows share a
                ``(doc_id, char_start, char_end)``.
        """
        if not key_terms:
            raise ValueError("key_terms is empty")
        if isinstance(documents, list):
            documents = {str(i): t for i, t in enumerate(documents)}
        if not documents:
            raise ValueError("documents is empty")

        documents, self.doc_info = self.prepare_documents(documents)
        self.n_truncated = 0
        self.truncated_docs = []

        rep_blocks: dict[int, list[np.ndarray]] = {L: [] for L in self.layers}
        cols: dict[str, list] = {
            k: [] for k in (
                "labels", "doc_ids", "char_starts", "char_ends", "paragraph_indices",
                "window_starts", "window_ends", "token_indices", "token_first_indices",
                "n_pooled_tokens",
            )
        }

        for doc_id, text in documents.items():
            occurrences = find_key_term_occurrences(text, key_terms)
            if not occurrences:
                continue
            paragraphs = split_paragraphs(text)
            assert paragraphs, f"doc {doc_id!r} has occurrences but no paragraphs"

            # A unit is one forward pass: ((window_start, window_end), occurrences),
            # each occurrence being (term, match_start, match_end, paragraph_index).
            occ_para = [
                (term, ms, me, paragraph_of_span(paragraphs, ms, me))
                for term, ms, me in occurrences
            ]
            if self.context_radius is None:
                units = [((0, len(text)), occ_para)]
            else:
                by_para: dict[int, list] = {}
                for o in occ_para:
                    by_para.setdefault(o[3], []).append(o)
                units = [
                    (window_span(paragraphs, pi, self.context_radius), occs)
                    for pi, occs in sorted(by_para.items())
                ]

            for (w_start, w_end), unit_occs in units:
                window_text = text[w_start:w_end]
                input_ids, offsets = self._tokenize(window_text)
                n_full = len(input_ids)
                cutoff = n_full
                if n_full > self.max_position_embeddings:
                    where = (
                        f"doc {doc_id!r} window [{w_start}, {w_end}) "
                        f"(paragraph {unit_occs[0][3]})"
                    )
                    if self.overflow_policy == "error":
                        raise ValueError(
                            f"{where}: {n_full} tokens > max_position_embeddings "
                            f"{self.max_position_embeddings}; set a smaller context_radius, "
                            "or (causal only) overflow_policy='tail_drop'."
                        )
                    cutoff = self.max_position_embeddings
                    warnings.warn(
                        f"[RepresentationLM] {where}: {n_full} tokens > "
                        f"{cutoff}; tail-truncating, dropping {n_full - cutoff} tokens.",
                        stacklevel=2,
                    )
                    input_ids = input_ids[:cutoff]
                    self.n_truncated += 1
                    self.truncated_docs.append(
                        {"doc_id": doc_id, "window_start": w_start, "window_end": w_end,
                         "n_tokens": n_full, "dropped": n_full - cutoff}
                    )

                groups: list[list[int]] = []
                kept: list[tuple[str, int, int, int]] = []
                for term, ms, me, pi in unit_occs:
                    surface = text[ms:me].lower()
                    assert surface in {term, term + "s"}, (
                        f"doc {doc_id!r}: match text {surface!r} not {term!r}/{term + 's'!r}"
                    )
                    try:
                        term_tokens = select_term_token_indices(offsets, ms - w_start, me - w_start)
                    except AssertionError as e:
                        raise AssertionError(
                            f"doc {doc_id!r}, {term!r} occurrence at [{ms}, {me}] "
                            f"({text[ms:me]!r}): {e}"
                        ) from e
                    if term_tokens[-1] >= cutoff:
                        continue  # past the truncation cutoff (tail_drop only)
                    groups.append(term_tokens if self.pooling == "mean" else term_tokens[-1:])
                    kept.append((term, ms, me, pi))

                if not groups:
                    continue

                reps = self._collect_reps(input_ids, groups)
                for L in self.layers:
                    assert reps[L].shape == (len(groups), self.hidden_size)
                    rep_blocks[L].append(reps[L])
                for (term, ms, me, pi), g in zip(kept, groups):
                    cols["labels"].append(term)
                    cols["doc_ids"].append(str(doc_id))
                    cols["char_starts"].append(int(ms))
                    cols["char_ends"].append(int(me))
                    cols["paragraph_indices"].append(int(pi))
                    cols["window_starts"].append(int(w_start))
                    cols["window_ends"].append(int(w_end))
                    cols["token_indices"].append(int(g[-1]))
                    cols["token_first_indices"].append(int(g[0]))
                    cols["n_pooled_tokens"].append(len(g))

        if not cols["labels"]:
            raise RuntimeError(
                "No key-term occurrences found in any document — nothing to collect."
            )

        representations = {
            L: np.concatenate(rep_blocks[L], axis=0).astype(np.float32)
            for L in self.layers
        }
        n = len(cols["labels"])

        out = {
            "representations": representations,
            "layers": np.asarray(self.layers, dtype=np.int64),
            "labels": np.asarray(cols["labels"], dtype=object).astype("U"),
            "doc_ids": np.asarray(cols["doc_ids"], dtype=object).astype("U"),
        }
        for k in ("char_starts", "char_ends", "paragraph_indices", "window_starts",
                  "window_ends", "token_indices", "token_first_indices", "n_pooled_tokens"):
            out[k] = np.asarray(cols[k], dtype=np.int64)

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
        assert (out["window_starts"] <= out["char_starts"]).all()
        assert (out["char_ends"] <= out["window_ends"]).all()
        assert (out["token_first_indices"] <= out["token_indices"]).all()
        if self.pooling == "last":
            assert (out["n_pooled_tokens"] == 1).all()
        triples = list(zip(out["doc_ids"].tolist(), out["char_starts"].tolist(), out["char_ends"].tolist()))
        assert len(set(triples)) == n, (
            f"{n - len(set(triples))} rows share a (doc_id, char_start, char_end) "
            "— overlapping key terms?"
        )
        return out
