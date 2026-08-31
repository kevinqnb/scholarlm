"""Key-term representation collection (NNsight / RepresentationLM).

Passes every non-excluded document of a dataset through a base
(non-instruction-tuned) model as raw next-token prediction and collects the
last-layer, post-final-norm hidden state at the token positions inside a
whole-word occurrence of a supplied key term. Writes an ``n x d`` array plus
the parallel label / provenance arrays.

This is exploratory groundwork for the ``naacl-27`` direction — see
``notes/scholarlm/builds/2026-08-31-representation-lm-01.md``. No probing /
clustering / separation analysis here; this runner stops at the artifact.

Standard output path (a separate tree, like jacobian_lens/):
    data/experiments/{dataset}/representation_lm/{model}/{date}/

Saves:
  - ``representations.npz`` — ``representations`` (float32 [n, d]), ``labels``,
    ``doc_ids``, ``char_starts``, ``char_ends``, ``token_indices`` (all length
    n), plus scalars ``key_terms``, ``model_name``, ``hidden_size``,
    ``n_documents``, ``n_truncated``, ``seed``.
  - ``run_metadata.json``

Usage
-----
    python experiments/run_representation_lm.py \\
        --dataset pond \\
        --model llama-3.1-8b-base \\
        --key-terms pond lake wetland \\
        [--limit N] [--date YYYY_mm_dd]

NOTE: ``scripts/submit.sh <id>`` cannot run this — the contract adapter only
covers entry_point extraction|ablation, and ``llama-3.1-8b-base`` is not in
the extraction MODEL_REGISTRY. The full run is a hand-authored single-GPU
qsub job (see the build note).

Available models: the keys of REPRESENTATION_LM_REGISTRY.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_EXPERIMENTS_DIR))

from dotenv import load_dotenv
load_dotenv()

import numpy as np

from scholarlm import RepresentationLM
from scholarlm.representationlm import find_key_term_occurrences

import paths
from model_registry import REPRESENTATION_LM_REGISTRY
from run_extraction import load_dataset_config, load_papers
from utils import load_config, set_seeds, write_run_metadata


def _load_documents(dataset_config, limit: int | None) -> dict[str, str]:
    """OCR text for every non-excluded paper, via run_extraction.load_papers.

    load_papers applies paper_filter -> paper_exclude -> paper_subset (unlike
    judge_common.load_documents_for_dataset, which skips paper_exclude).
    """
    ocr_dir = str(Path(dataset_config.data_dir) / "ocr_output_raw")
    text, text_info = load_papers(dataset_config, ocr_dir)
    assert len(text) == len(text_info)

    documents: dict[str, str] = {}
    for t, info in zip(text, text_info):
        doc_id = info["document_id"]
        assert doc_id not in documents, f"duplicate document_id {doc_id!r}"
        documents[doc_id] = t

    if dataset_config.paper_exclude is not None:
        bad = set(documents) & set(dataset_config.paper_exclude)
        assert not bad, f"excluded papers leaked into the document set: {sorted(bad)}"

    if limit is not None:
        documents = {k: documents[k] for k in list(documents)[:limit]}

    assert documents, "no documents loaded"
    return documents


def run_representation_lm(
    dataset: str,
    model_key: str,
    key_terms: list[str],
    output_dir: Path,
    seed: int,
    limit: int | None = None,
    verify_read_point: bool = False,
) -> None:
    if model_key not in REPRESENTATION_LM_REGISTRY:
        raise KeyError(
            f"Unknown model {model_key!r}. Available: {sorted(REPRESENTATION_LM_REGISTRY)}"
        )
    model_cfg = REPRESENTATION_LM_REGISTRY[model_key]

    dataset_config = load_dataset_config(dataset)
    documents = _load_documents(dataset_config, limit)
    print(f"Documents        : {len(documents)}")
    print(f"Key terms        : {key_terms}")

    # Independent known-answer prediction: total row count and per-term counts
    # from a fresh regex pass. (The module uses the same regex helper, so this
    # is a consistency check, not full independence — the regex behaviour
    # itself is covered by tests/test_representationlm.py.)
    predicted_per_term: dict[str, int] = {t: 0 for t in key_terms}
    for text in documents.values():
        for term, _, _ in find_key_term_occurrences(text, key_terms):
            predicted_per_term[term] += 1
    predicted_total = sum(predicted_per_term.values())
    print(f"Predicted rows   : {predicted_total}  {predicted_per_term}")

    llm = RepresentationLM(
        model_name=model_cfg["model_id"],
        nnsight_kwargs=model_cfg["nnsight_kwargs"],
        hf_cache_dir=os.environ.get("HF_CACHE"),
        verbose=True,
    )

    if verify_read_point:
        print("\n--- verify_read_point (post-final-norm + determinism gate) ---")
        llm.verify_read_point(next(iter(documents.values())), key_terms)
        print("--- verify_read_point passed ---\n")

    start_time = time.time()
    out = llm.collect(documents, key_terms)

    n = out["representations"].shape[0]
    collected_per_term = {
        t: int((out["labels"] == t).sum()) for t in key_terms
    }
    print(f"Collected rows   : {n}  {collected_per_term}")

    # Every occurrence is collected unless it fell in a truncated tail.
    dropped_to_truncation = predicted_total - n
    assert dropped_to_truncation >= 0
    if llm.n_truncated == 0:
        assert n == predicted_total, (
            f"row count {n} != predicted {predicted_total} with no truncation"
        )
        assert collected_per_term == predicted_per_term, (
            f"per-term {collected_per_term} != predicted {predicted_per_term}"
        )
    else:
        print(
            f"NOTE: {llm.n_truncated} doc(s) tail-truncated; "
            f"{dropped_to_truncation} occurrence(s) past the cutoff dropped."
        )

    assert out["representations"].shape == (n, llm.hidden_size)
    assert np.isfinite(out["representations"]).all(), "non-finite representations"

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "representations.npz"
    np.savez_compressed(
        npz_path,
        representations=out["representations"],
        labels=out["labels"],
        doc_ids=out["doc_ids"],
        char_starts=out["char_starts"],
        char_ends=out["char_ends"],
        token_indices=out["token_indices"],
        key_terms=np.asarray(key_terms, dtype=object).astype("U"),
        model_name=np.asarray(model_cfg["model_id"]),
        hidden_size=np.asarray(llm.hidden_size),
        n_documents=np.asarray(len(documents)),
        n_truncated=np.asarray(llm.n_truncated),
        seed=np.asarray(seed),
    )
    print(f"Representations   : {npz_path}  ({npz_path.stat().st_size / 1e6:.1f} MB)")

    write_run_metadata(
        output_dir,
        start_time=start_time,
        dataset=dataset,
        model=model_key,
        model_id=model_cfg["model_id"],
        key_terms=key_terms,
        seed=seed,
        n_documents=len(documents),
        n_rows=int(n),
        rows_per_term=collected_per_term,
        predicted_rows_per_term=predicted_per_term,
        n_truncated=llm.n_truncated,
        truncated_docs=llm.truncated_docs,
        hidden_size=llm.hidden_size,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect key-term representations (NNsight/RepresentationLM).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", required=True, help="Dataset name (e.g. 'pond').")
    p.add_argument(
        "--model", required=True, choices=sorted(REPRESENTATION_LM_REGISTRY),
        help=f"Model key. Available: {sorted(REPRESENTATION_LM_REGISTRY)}",
    )
    p.add_argument(
        "--key-terms", required=True, nargs="+", metavar="TERM",
        help="Base key terms to match (case-insensitive, whole-word, simple plural).",
    )
    p.add_argument("--date", default=None, help="Output date tag YYYY_mm_dd (default: today).")
    p.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Only process the first N documents (smoke / tiny-e2e).",
    )
    p.add_argument(
        "--verify-read-point", action="store_true",
        help=(
            "Smoke-only gate: before collecting, prove on one document that "
            "the collected vector is the post-final-norm state (vs the "
            "pre-norm residual and vs the model's own ln_final) and that two "
            "identical passes are bitwise equal. Aborts on failure."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    # DatasetConfig paths (metadata_file, data_dir, ...) are repo-root-relative.
    os.chdir(_REPO_ROOT)

    cfg = load_config()
    seed = cfg["defaults"]["seed"]  # no default — CLAUDE.md's no-magic-number rule
    set_seeds(seed)

    output_dir = paths.representation_lm(args.dataset, args.model, args.date)

    print(f"\nDataset          : {args.dataset}")
    print(f"Model            : {args.model}")
    print(f"Seed             : {seed}")
    print(f"Output           : {output_dir}\n")

    run_representation_lm(
        dataset=args.dataset,
        model_key=args.model,
        key_terms=args.key_terms,
        output_dir=output_dir,
        seed=seed,
        limit=args.limit,
        verify_read_point=args.verify_read_point,
    )


if __name__ == "__main__":
    main()
