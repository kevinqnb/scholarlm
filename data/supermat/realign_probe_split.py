"""
Realign the supermat v3 probe files to the v1/v2 paper split ("v3s").

Why
---
v3 was built from ``ground_truth_qualifiers.json`` (142 papers); v1/v2 from
``ground_truth.json`` (136). ``sample_valid_set`` shuffles the paper list with
``Random(42)``, so the different paper list produced a different split: 38 v1-train
papers landed in v3 test and 26 v1-test papers in v3 train. The manual validation
sets (``data/validation/``, and the scholarlm-validation site) were drawn from the
v1 test split, so the v3 probe was trained on papers the validation sets evaluate on.

What
----
Rearranges existing rows -- no regeneration, no gpt-oss, no judge calls -- so the
test split is exactly the v1 test split. The 6 papers new in v3 (no v1 assignment)
go to train. Per paper, by (v3 side -> new side):

    train -> train   its v3 train rows stay in train.
    test  -> test    its v3 primary and diag rows stay.
    test  -> train   ONLY its v3 diag rows go to train; its primary rows are dropped.
    train -> test    all its v3 train rows go to diag; its GT valids and the
                     negatives built from them (``augment_axis is None``: original
                     paper, no gpt-oss content) also go to primary.

Each file is then trimmed to 1:1 (seeded, logged): surplus valids are dropped only
from synthetic valids (GT valids are always carried, as in the builder); surplus
negatives are dropped from the most frequent ``modification_type`` first. Rows are
reshuffled and ``measurement_id`` renumbered 0..N-1; the log maps every new row to
its source file and source ``measurement_id``.

Judge-interp outputs are moved the same way (``judge`` subcommand): every new row
takes the response and activations of its source row. Valid because the judge
prompt is a pure function of the row -- identical rows in the v3 primary and diag
runs have bit-exact activations and probabilities for all three judges.

Note: a fresh ``create_probe_dataset.py --augment --split-file`` build will NOT
byte-match these files (the RNG stream is consumed differently); this script, run on
the v3 files, is what reproduces them.

Usage (repo root)
-----------------
    python data/supermat/realign_probe_split.py split   --out data/supermat/probe_split_v3s.json
    python data/supermat/realign_probe_split.py realign --split-file ... --seed 342 --out-dir data/supermat
    python data/supermat/realign_probe_split.py judge   --log ... --train-run ... --primary-run ... \\
        --diag-run ... --out-train ... --out-primary ... --out-diag ...
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent

FILES = ("train", "primary", "diag")
V3_NAMES = {"train": "probe_dataset_v3.json",
            "primary": "probe_dataset_test_v3.json",
            "diag": "probe_dataset_test_v3_diag.json"}
V3S_NAMES = {"train": "probe_dataset_v3s.json",
             "primary": "probe_dataset_test_v3s.json",
             "diag": "probe_dataset_test_v3s_diag.json"}
LOG_NAME = "probe_dataset_v3s_realign_log.json"

# Judge-visible content of a row (the builder's _row_signature uses the same GT
# columns; the stamped full paper stands in for the edited context).
_SIG_COLS = ("document_id", "name", "attribute", "value", "units", "context_override")
# Fields run_judge_interp adds to a row in responses.json.
JUDGE_FIELDS = frozenset({"judgement", "judgement_prob", "judgement_p_true",
                          "judgement_p_false", "judgement_logit_p_true",
                          "judgement_logit_p_false", "judgement_model"})
NPZ_NAMES = ("attention_outputs.npz", "layer_outputs.npz")


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


def make_split(v1_train: set[str], v1_test: set[str], all_docs: set[str]) -> dict[str, str]:
    """v1 assignment for every v1 paper; every paper v1 never saw goes to train."""
    assert v1_train.isdisjoint(v1_test), f"v1 split overlaps: {sorted(v1_train & v1_test)}"
    assert (v1_train | v1_test) <= all_docs, \
        f"v1 papers missing from the GT: {sorted((v1_train | v1_test) - all_docs)}"
    return {d: ("test" if d in v1_test else "train") for d in sorted(all_docs)}


def _v3_side(v3: dict[str, list[dict]]) -> dict[str, str]:
    train_docs = {r["document_id"] for r in v3["train"]}
    prim_docs = {r["document_id"] for r in v3["primary"]}
    diag_docs = {r["document_id"] for r in v3["diag"]}
    assert prim_docs == diag_docs, \
        f"v3 primary/diag paper sets differ: {sorted(prim_docs ^ diag_docs)}"
    assert train_docs.isdisjoint(prim_docs), \
        f"paper in both v3 train and test: {sorted(train_docs & prim_docs)}"
    return {**{d: "train" for d in train_docs}, **{d: "test" for d in prim_docs}}


# ---------------------------------------------------------------------------
# Realign
# ---------------------------------------------------------------------------


def _sig(r: dict) -> tuple:
    return tuple(r[c] for c in _SIG_COLS)


def _trim_to_balance(cands: list[tuple[str, dict]], rng: random.Random, name: str
                     ) -> tuple[list[tuple[str, dict]], list[tuple[str, dict]]]:
    """Drop the larger class down to the smaller. Returns (kept, dropped)."""
    valid = [c for c in cands if c[1]["label"] == "valid"]
    invalid = [c for c in cands if c[1]["label"] == "invalid"]
    assert len(valid) + len(invalid) == len(cands), f"[{name}] label not in {{valid, invalid}}"
    surplus = len(valid) - len(invalid)
    drop_ids: set[int] = set()
    if surplus > 0:
        synth = [c for c in valid if c[1]["augment_axis"] is not None]
        assert len(synth) >= surplus, \
            f"[{name}] {surplus} surplus valids but only {len(synth)} synthetic valids to drop"
        drop_ids = {id(c) for c in rng.sample(synth, surplus)}
    elif surplus < 0:
        by_type: dict[str, list] = {}
        for c in invalid:
            by_type.setdefault(c[1]["modification_type"], []).append(c)
        for _ in range(-surplus):
            # most frequent type first; ties broken by name for determinism
            t = max(sorted(by_type), key=lambda k: len(by_type[k]))
            pick = by_type[t].pop(rng.randrange(len(by_type[t])))
            drop_ids.add(id(pick))
    kept = [c for c in cands if id(c) not in drop_ids]
    dropped = [c for c in cands if id(c) in drop_ids]
    return kept, dropped


def realign_rows(v3: dict[str, list[dict]], split: dict[str, str], rng: random.Random
                 ) -> tuple[dict[str, list[dict]], dict]:
    """Pure core. ``v3``: {"train"|"primary"|"diag": rows}. ``split``: doc -> "train"|"test".

    Returns ({file: new rows}, log) where ``log[file]["provenance"][i]`` gives the
    source of new row ``measurement_id == i``.
    """
    side = _v3_side(v3)
    assert set(split.values()) <= {"train", "test"}, f"bad split values: {set(split.values())}"
    assert set(split) == set(side), (
        f"split vs v3 papers differ: unassigned {sorted(set(side) - set(split))}, "
        f"not in v3 {sorted(set(split) - set(side))}")

    to_test = lambda r: split[r["document_id"]] == "test"
    cands = {
        "train": [("train", r) for r in v3["train"] if not to_test(r)]
                 + [("diag", r) for r in v3["diag"] if not to_test(r)],
        "primary": [("primary", r) for r in v3["primary"] if to_test(r)]
                   + [("train", r) for r in v3["train"]
                      if to_test(r) and r["augment_axis"] is None],
        "diag": [("diag", r) for r in v3["diag"] if to_test(r)]
                + [("train", r) for r in v3["train"] if to_test(r)],
    }

    out: dict[str, list[dict]] = {}
    log: dict = {"moves": {
        f"{a}->{b}": sorted(d for d in split if side[d] == a and split[d] == b)
        for a in ("train", "test") for b in ("train", "test")}}
    for name in FILES:
        c = cands[name]
        sigs = [_sig(r) for _, r in c]
        assert len(set(sigs)) == len(sigs), f"[{name}] duplicate rows after the move"
        kept, dropped = _trim_to_balance(c, rng, name)
        rng.shuffle(kept)
        rows, prov = [], []
        for i, (src, r) in enumerate(kept):
            rows.append({**r, "measurement_id": i})
            prov.append({"measurement_id": i, "src_file": src,
                         "src_measurement_id": r["measurement_id"]})
        out[name] = rows
        log[name] = {
            "n_rows": len(rows),
            "n_valid": sum(r["label"] == "valid" for r in rows),
            "provenance": prov,
            "dropped": [{"src_file": s, "src_measurement_id": r["measurement_id"],
                         "label": r["label"], "modification_type": r["modification_type"],
                         "augment_axis": r["augment_axis"]} for s, r in dropped],
        }
        assert log[name]["n_valid"] * 2 == len(rows), f"[{name}] not balanced after trim"

    want_test = {d for d, s in split.items() if s == "test"}
    for name in ("primary", "diag"):
        got = {r["document_id"] for r in out[name]}
        assert got == want_test, f"[{name}] papers != split test: {sorted(got ^ want_test)}"
    got_train = {r["document_id"] for r in out["train"]}
    assert got_train == set(split) - want_test, \
        f"[train] papers != split train: {sorted(got_train ^ (set(split) - want_test))}"
    return out, log


# ---------------------------------------------------------------------------
# Judge-interp transplant
# ---------------------------------------------------------------------------


def transplant_judge(provenance: list[dict], new_rows: list[dict],
                     src_responses: dict[str, list[dict]],
                     src_arrays: dict[str, dict[str, dict[str, np.ndarray]]]
                     ) -> tuple[list[dict], dict[str, dict[str, np.ndarray]]]:
    """Build one new judge-interp run from the source runs' outputs.

    ``src_responses[src_file]``: that run's responses.json rows.
    ``src_arrays[src_file][npz_name]``: {str(measurement_id): array}.
    Every new row must equal its source response on every non-judgement field
    (except ``measurement_id``).
    """
    assert [p["measurement_id"] for p in provenance] == list(range(len(new_rows)))
    by_mid: dict[str, dict[int, dict]] = {}
    for src, resp in src_responses.items():
        by_mid[src] = {r["measurement_id"]: r for r in resp}
        assert len(by_mid[src]) == len(resp), f"{src}: duplicate measurement_id in responses"
        for npz in NPZ_NAMES:
            keys = set(src_arrays[src][npz].keys())
            assert keys == {str(m) for m in by_mid[src]}, \
                f"{src}/{npz}: measurement_id set != responses.json"

    responses: list[dict] = []
    arrays: dict[str, dict[str, np.ndarray]] = {npz: {} for npz in NPZ_NAMES}
    for p, row in zip(provenance, new_rows):
        src, old = p["src_file"], p["src_measurement_id"]
        resp = by_mid[src][old]
        content = {k: v for k, v in resp.items() if k not in JUDGE_FIELDS}
        assert set(resp) - set(content) == JUDGE_FIELDS, \
            f"{src} mid {old}: judgement fields {sorted(set(resp) - set(content))}"
        assert {**content, "measurement_id": row["measurement_id"]} == row, \
            f"{src} mid {old} -> {row['measurement_id']}: response row != new row"
        responses.append({**resp, "measurement_id": row["measurement_id"]})
        for npz in NPZ_NAMES:
            arrays[npz][str(row["measurement_id"])] = src_arrays[src][npz][str(old)]
    return responses, arrays


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def _dump(obj, path: Path) -> None:
    with open(path, "x") as f:          # never overwrite
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"wrote {path}")


def _cmd_split(args) -> None:
    ids = lambda p: {r["document_id"] for r in _load(p)}
    split = make_split(ids(BASE / "probe_dataset.json"), ids(BASE / "probe_dataset_test.json"),
                       {str(r["document_id"]) for r in _load(Path(args.gt_file))})
    n_test = sum(s == "test" for s in split.values())
    print(f"{len(split)} papers: {len(split) - n_test} train / {n_test} test")
    _dump(split, Path(args.out))


def _cmd_realign(args) -> None:
    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    v3 = {k: _load(in_dir / n) for k, n in V3_NAMES.items()}
    split = _load(Path(args.split_file))
    out, log = realign_rows(v3, split, random.Random(args.seed))
    log["seed"] = args.seed
    log["split_file"] = str(args.split_file)
    log["inputs"] = {k: str(in_dir / n) for k, n in V3_NAMES.items()}
    for k in FILES:
        print(f"[{k}] {log[k]['n_rows']} rows ({log[k]['n_valid']} valid), "
              f"{len(log[k]['dropped'])} dropped in trim")
        _dump(out[k], out_dir / V3S_NAMES[k])
    _dump(log, out_dir / LOG_NAME)


_IDENTITY_KEYS = ("dataset", "extraction_id", "judge_model", "judge_model_id")


def judge_run_metadata(src_meta: dict[str, dict], runs: dict[str, str], log_path: str,
                       data_file: str) -> dict:
    """run_metadata.json for a realigned run. The identity keys the analysis loaders
    check (``judge_model``, ``dataset``, ...) stay top-level and must agree across
    every source run; everything run-specific stays nested per source."""
    ident = {k: src_meta["train"][k] for k in _IDENTITY_KEYS}
    for s, m in src_meta.items():
        assert {k: m[k] for k in _IDENTITY_KEYS} == ident, \
            f"source run {s} identity {({k: m[k] for k in _IDENTITY_KEYS})} != train's {ident}"
    return {**ident, "realigned_from": runs, "realign_log": log_path, "data_file": data_file,
            "source_run_metadata": src_meta}


def _cmd_judge(args) -> None:
    log = _load(Path(args.log))
    data_dir = Path(args.data_dir)
    runs = {"train": Path(args.train_run), "primary": Path(args.primary_run),
            "diag": Path(args.diag_run)}
    src_responses = {k: _load(d / "responses.json") for k, d in runs.items()}
    src_arrays = {k: {n: np.load(d / n) for n in NPZ_NAMES} for k, d in runs.items()}
    src_meta = {k: _load(d / "run_metadata.json") for k, d in runs.items()}
    outs = {"train": Path(args.out_train), "primary": Path(args.out_primary),
            "diag": Path(args.out_diag)}
    for k in FILES:
        new_rows = _load(data_dir / V3S_NAMES[k])
        resp, arrays = transplant_judge(log[k]["provenance"], new_rows, src_responses, src_arrays)
        outs[k].mkdir(parents=True, exist_ok=False)
        _dump(resp, outs[k] / "responses.json")
        for n, a in arrays.items():
            np.savez(outs[k] / n, **a)
            print(f"wrote {outs[k] / n}")
        _dump(judge_run_metadata(src_meta, {s: str(d) for s, d in runs.items()},
                                 str(args.log), str(data_dir / V3S_NAMES[k])),
              outs[k] / "run_metadata.json")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--gt-file", required=True)
    s.add_argument("--out", required=True)
    r = sub.add_parser("realign")
    r.add_argument("--split-file", required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--in-dir", required=True)
    r.add_argument("--out-dir", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--log", required=True)
    j.add_argument("--data-dir", required=True)
    for k in FILES:
        j.add_argument(f"--{k}-run", required=True)
        j.add_argument(f"--out-{k}", required=True)
    args = p.parse_args(argv)
    {"split": _cmd_split, "realign": _cmd_realign, "judge": _cmd_judge}[args.cmd](args)


if __name__ == "__main__":
    main()
