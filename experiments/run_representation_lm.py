"""Key-term representation collection (NNsight / RepresentationLM).

Passes every non-excluded document of a dataset through a base
(non-instruction-tuned) model as raw next-token prediction and collects the
hidden state at a configurable list of layers (``--layers``) at the token
positions inside a whole-word occurrence of a supplied key term. One forward
pass per document yields every requested layer. Writes one ``n x d`` array per
layer plus the shared label / provenance arrays.

Layer indices: ``0`` = token embeddings, ``1..n_layers-1`` = residual stream
after that many blocks, ``n_layers`` (32 for llama-3.1-8b-base) =
post-final-norm (the vector the unembedding sees). Layers ``0..n_layers-1``
are pre-norm residuals; only ``n_layers`` is post-norm — row-L2 norms are not
comparable across that boundary.

This is exploratory groundwork for the ``naacl-27`` direction — see
``notes/scholarlm/builds/2026-08-31-representation-lm-01.md``. No probing /
clustering / separation analysis here; this runner stops at the artifact.

Output path (id-addressed, like every other Tier-1 type):
    experiments/results/{dataset}/representation_lm/{experiment_id}/

Saves:
  - ``representations.npz`` — one ``rep_layer_{L:02d}`` array per collected
    layer (float32 [n, d]), ``layers`` (int64 [k]), ``labels``, ``doc_ids``,
    ``char_starts``, ``char_ends``, ``token_indices`` (all length n), plus
    scalars ``key_terms``, ``model_name``, ``hidden_size``, ``n_documents``,
    ``n_truncated``, ``seed``.  (``savez_compressed`` decompresses per member,
    so a per-layer probe reads one array, not all of them.)
  - ``run_metadata.json``

Usage
-----
    python experiments/run_representation_lm.py experiments/experiment-configs/pond/representation_lm/<id>/<id>.yaml

Required params: dataset, model, key_terms (list), layers (list of int; no
default -- CLAUDE.md's no-magic-numbers rule; the module's own default is
[0, 8, 16, 24, 32]).
Optional params: limit, verify_read_point (bool).

Available models: the YAML files in experiments/model-configs/representation_lm/.
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

import utils as paths
from run_extraction import load_dataset_config, load_papers
from utils import load_config, set_seeds, write_run_metadata


def _load_documents(dataset_config, limit: int | None) -> dict[str, str]:
    """OCR text for every non-excluded paper, via run_extraction.load_papers.

    load_papers applies paper_filter -> paper_exclude -> paper_subset (unlike
    judge_prompts.load_documents_for_dataset, which skips paper_exclude).
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
    layers: list[int],
    output_dir: Path,
    seed: int,
    limit: int | None = None,
    verify_read_point: bool = False,
) -> None:
    model_cfg = paths.load_model_config("representation_lm", model_key)

    dataset_config = load_dataset_config(dataset)
    documents = _load_documents(dataset_config, limit)
    print(f"Documents        : {len(documents)}")
    print(f"Key terms        : {key_terms}")
    print(f"Layers           : {layers}")

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
        layers=layers,
        nnsight_kwargs=model_cfg["nnsight_kwargs"],
        hf_cache_dir=os.environ.get("HF_CACHE"),
        verbose=True,
    )
    # RepresentationLM validated/sorted/deduped the list against the model.
    resolved_layers = list(llm.layers)
    print(f"Resolved layers  : {resolved_layers}")

    if verify_read_point:
        print("\n--- verify_read_point (per-layer determinism + read-point gate) ---")
        llm.verify_read_point(next(iter(documents.values())), key_terms)
        print("--- verify_read_point passed ---\n")

    start_time = time.time()
    out = llm.collect(documents, key_terms)

    reps = out["representations"]  # {layer: float32 [n, d]}
    assert sorted(reps) == resolved_layers, (sorted(reps), resolved_layers)
    assert out["layers"].tolist() == resolved_layers
    n = len(out["labels"])
    collected_per_term = {
        t: int((out["labels"] == t).sum()) for t in key_terms
    }
    print(f"Collected rows   : {n}  {collected_per_term}")

    # Every occurrence is collected unless it fell in a truncated tail. The row
    # set is shared across layers, so this row-count gate is layer-independent.
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

    per_layer_norms: dict[str, float] = {}
    for L in resolved_layers:
        assert reps[L].shape == (n, llm.hidden_size), (L, reps[L].shape)
        assert np.isfinite(reps[L]).all(), f"non-finite representations at layer {L}"
        per_layer_norms[str(L)] = float(np.linalg.norm(reps[L], axis=1).mean())
        print(f"  layer {L:>2d}: mean row L2 {per_layer_norms[str(L)]:.4g}")

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "representations.npz"
    npz_arrays = {
        f"rep_layer_{L:02d}": reps[L] for L in resolved_layers
    }
    np.savez_compressed(
        npz_path,
        layers=out["layers"],
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
        **npz_arrays,
    )
    print(f"Representations   : {npz_path}  ({npz_path.stat().st_size / 1e6:.1f} MB)")

    write_run_metadata(
        output_dir,
        start_time=start_time,
        dataset=dataset,
        model=model_key,
        model_id=model_cfg["model_id"],
        key_terms=key_terms,
        layers=resolved_layers,
        seed=seed,
        n_documents=len(documents),
        n_rows=int(n),
        rows_per_term=collected_per_term,
        predicted_rows_per_term=predicted_per_term,
        mean_row_l2_by_layer=per_layer_norms,
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
    p.add_argument("config", help="Path to an experiment-configs/.../<id>.yaml.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config_path = Path(args.config)
    cfg = paths.load_experiment_config(config_path)
    params = cfg["params"]
    paths.require_params(params, "dataset", "model", "key_terms", "layers", config_path=config_path)

    model_key = params["model"]
    paths.load_model_config("representation_lm", model_key)  # fail loud on an unknown model before any work starts

    # DatasetConfig paths (metadata_file, data_dir, ...) are repo-root-relative.
    os.chdir(_REPO_ROOT)

    repo_seed = load_config()["defaults"]["seed"]  # no default -- CLAUDE.md's no-magic-number rule
    if cfg["seed"] != repo_seed:
        raise ValueError(
            f"{config_path}: seed ({cfg['seed']}) does not match experiments/config.yaml "
            f"defaults.seed ({repo_seed}) -- the repo's seed is a fixed, repo-wide value, "
            "not a per-run knob."
        )
    seed = cfg["seed"]
    set_seeds(seed)

    dataset = params["dataset"]
    output_dir = paths.result_dir(dataset, "representation_lm", cfg["id"])

    print(f"\nDataset          : {dataset}")
    print(f"Model            : {model_key}")
    print(f"Seed             : {seed}")
    print(f"Output           : {output_dir}\n")

    run_representation_lm(
        dataset=dataset,
        model_key=model_key,
        key_terms=params["key_terms"],
        layers=params["layers"],
        output_dir=output_dir,
        seed=seed,
        limit=params.get("limit"),
        verify_read_point=params.get("verify_read_point", False),
    )


if __name__ == "__main__":
    main()
