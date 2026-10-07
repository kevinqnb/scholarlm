"""experiments/combine_ablation_parts.py <config>

Ad-hoc: merge the outputs of an ablation run that was split across several jobs
(each a normal run_ablation.py config restricted to a disjoint ``params.paper_subset``)
into the single result a one-job run would have produced, under the combined
config's own id.

The combined config is an ordinary ablation config (same dataset/model/ablation/
ocr_dir as the parts) plus ``params.parts``: the list of part experiment ids. Run
this directly (it is CPU-only, seconds) once every part has finished -- do NOT
``submit.sh`` the combined id; that would launch the full single-job run again.

Fails loud, never repairs. It raises if: a part's config disagrees with the combined
config; the parts' papers overlap or do not cover exactly the papers a one-job run
would have processed; a part has no final.json / run_metadata.json; the parts were run
at different git commits; any record's document_id is outside its part, or its
batch-local ``entity_id`` index (``doc_<i>_entity_<j>``, ``i`` = the document's
position within *that part's* run) does not point at the record's own document; or the
output directory already has content.

What it rewrites, to match a one-job run: ``entity_id``'s document index becomes the
paper's index in the full sorted paper list, records are ordered by paper (then by
their order within the part), and ``measurement_id`` is renumbered 0..n-1. Everything
else is copied unchanged.

Order: MeasurementLM.fit() builds ``text_values + table_values``, and _deduplicate()
keeps first occurrences and never mixes sources, so a one-job final.json is a block of
text-sourced records followed by a block of table-sourced ones, each ordered by document.
The merge reproduces that (a record whose ``source`` is mixed or not text/table raises).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import utils as paths
from run_extraction import load_dataset_config, load_papers
from scholarlm.measurementlm import NumpyEncoder
from utils import write_run_metadata

_ENTITY_ID_RE = re.compile(r"doc_(\d+)_entity_(\d+)")

# Config keys that must be identical between the combined config and every part.
_SHARED_PARAMS = (
    "dataset", "model", "ablation", "ocr_dir",
    "include_qualifiers", "parse_quantities_context", "standardize_context",
)
# run_metadata keys that must be identical across parts.
_SHARED_METADATA = (
    "dataset", "model", "model_id", "hf_revision", "ablation", "include_qualifiers",
    "parse_quantities_context", "standardize_context", "commit",
)


def combine_records(expected_papers: list[str], parts: list[dict]) -> list[dict]:
    """Merge per-part final.json records into one-job order. Pure -- no I/O.

    Args:
        expected_papers: Every paper (document_id) a one-job run would process, in
            its processing (sorted-filename) order.
        parts: One dict per part: ``papers`` (that part's papers in the order its
            run processed them -- this defines the meaning of the batch-local
            ``doc_<i>_...`` ids in its records) and ``records`` (its final.json).
    """
    claimed = [p for part in parts for p in part["papers"]]
    if len(claimed) != len(set(claimed)):
        raise ValueError(f"parts overlap: {sorted({p for p in claimed if claimed.count(p) > 1})}")
    if set(claimed) != set(expected_papers):
        raise ValueError(
            f"parts do not cover the expected papers exactly: "
            f"missing={sorted(set(expected_papers) - set(claimed))}, "
            f"extra={sorted(set(claimed) - set(expected_papers))}"
        )
    global_index = {p: i for i, p in enumerate(expected_papers)}

    keyed = []  # (text/table block, global paper index, part number, position in part, record)
    for part_no, part in enumerate(parts):
        local_papers = part["papers"]
        for pos, rec in enumerate(part["records"]):
            doc = rec["document_id"]
            if doc not in local_papers:
                raise ValueError(f"part {part_no}: record {pos} has document_id {doc!r} outside the part")
            m = _ENTITY_ID_RE.fullmatch(rec["entity_id"])
            if m is None:
                raise ValueError(f"part {part_no}: record {pos} entity_id {rec['entity_id']!r} is not doc_<i>_entity_<j>")
            local_i = int(m.group(1))
            if local_i >= len(local_papers) or local_papers[local_i] != doc:
                raise ValueError(
                    f"part {part_no}: record {pos} entity_id {rec['entity_id']!r} points at "
                    f"local document {local_i}, but its document_id is {doc!r}"
                )
            kinds = set(rec["source"])
            if len(kinds) != 1 or not kinds <= {"text", "table"}:
                raise ValueError(f"part {part_no}: record {pos} has source {rec['source']!r}; expected all-text or all-table")
            block = 0 if "text" in kinds else 1
            new = dict(rec, entity_id=f"doc_{global_index[doc]}_entity_{m.group(2)}")
            keyed.append((block, global_index[doc], part_no, pos, new))

    keyed.sort(key=lambda t: t[:4])
    out = [dict(rec, measurement_id=i) for i, (*_, rec) in enumerate(keyed)]
    assert len(out) == sum(len(p["records"]) for p in parts), "row count changed in merge"
    assert len({r["measurement_id"] for r in out}) == len(out)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Combine split ablation parts into one result.")
    ap.add_argument("config", help="Path to the combined experiment-configs/.../<id>.yaml (has params.parts).")
    config_path = Path(ap.parse_args(argv).config)
    cfg = paths.load_experiment_config(config_path)
    params = cfg["params"]
    paths.require_params(params, "dataset", "model", "ablation", "ocr_dir", "parts", config_path=config_path)
    dataset = params["dataset"]

    out_dir = paths.result_dir(dataset, "ablation", cfg["id"])
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} already has content; refusing to overwrite.")

    dataset_config = load_dataset_config(dataset)
    _, full_info = load_papers(dataset_config, params["ocr_dir"], None)
    expected_papers = [i["document_id"] for i in full_info]

    parts, metas = [], []
    for part_id in params["parts"]:
        part_cfg_path = paths.find_experiment_config(part_id)
        part_cfg = paths.load_experiment_config(part_cfg_path)
        pp = part_cfg["params"]
        for key in _SHARED_PARAMS:
            if pp.get(key) != params.get(key):
                raise ValueError(f"{part_cfg_path}: params.{key}={pp.get(key)!r} != combined {params.get(key)!r}")
        if part_cfg["seed"] != cfg["seed"]:
            raise ValueError(f"{part_cfg_path}: seed {part_cfg['seed']} != combined {cfg['seed']}")
        subset = pp["paper_subset"]
        _, info = load_papers(dataset_config, params["ocr_dir"], subset)
        papers = [i["document_id"] for i in info]
        if sorted(papers) != sorted(subset):
            raise ValueError(f"{part_id}: paper_subset not fully loaded (dropped: {sorted(set(subset) - set(papers))})")

        part_dir = paths.result_dir(dataset, "ablation", part_id)
        with open(part_dir / "final.json") as f:
            records = json.load(f)
        with open(part_dir / "run_metadata.json") as f:
            metas.append(json.load(f))
        parts.append({"id": part_id, "papers": papers, "records": records})

    for key in _SHARED_METADATA:
        values = {json.dumps(m[key], sort_keys=True) for m in metas}
        if len(values) != 1:
            raise ValueError(f"parts disagree on run_metadata[{key!r}]: {sorted(values)}")

    combined = combine_records(expected_papers, parts)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "final.json", "w") as f:
        json.dump(combined, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

    step_names = sorted({s for m in metas for s in m["step_seconds"]})
    token_keys = sorted({k for m in metas for k in m["token_usage"]})
    write_run_metadata(
        out_dir,
        step_seconds={
            s: {"ran": True, "seconds": round(sum(m["step_seconds"][s]["seconds"] for m in metas if s in m["step_seconds"]), 1)}
            for s in step_names
        },
        token_usage={k: sum(m["token_usage"].get(k, 0) for m in metas) for k in token_keys},
        max_prompt_tokens=max(m["max_prompt_tokens"] for m in metas),
        **{k: metas[0][k] for k in _SHARED_METADATA if k != "commit"},
        parts_commit=metas[0]["commit"],
        parts_dirty=[m["dirty"] for m in metas],
        runtime_seconds_sum_of_parts=round(sum(m["runtime_seconds"] for m in metas), 1),
        step_seconds_note="summed across parts (total compute, not wall-clock)",
        combined_from=[
            {"id": p["id"], "n_papers": len(p["papers"]), "n_records": len(p["records"]),
             "run_timestamp": m["run_timestamp"], "runtime_seconds": m["runtime_seconds"]}
            for p, m in zip(parts, metas)
        ],
    )
    print(f"Combined {len(parts)} parts -> {out_dir / 'final.json'}: "
          f"{len(combined)} records from {len(expected_papers)} papers.")


if __name__ == "__main__":
    main()
