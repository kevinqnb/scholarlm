"""
Evaluate validity assessment methods for ScholarlM extractions.

Arm 1 (--synthetic): judge models vs known labels in probe_dataset_test.json.
Arm 2 (--human):     match / voted-judge / combined vs human labels from validation.py.

Both arms run by default; pass --synthetic or --human to run only one.

Usage
-----
    python analysis/validity_evaluation.py
    python analysis/validity_evaluation.py --synthetic
    python analysis/validity_evaluation.py --human --extraction-model gemma-3-27b
    python analysis/validity_evaluation.py --output results.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))
sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

import numpy as np
import pandas as pd

import paths
from run_extraction import load_dataset_config
from analysis.loaders import load_ground_truth, cached_match
from analysis.ablation import get_matching_rules, process_extraction_df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    n = len(y_true)
    if n == 0:
        nan = float("nan")
        return dict(N=0, N_pos=0, accuracy=nan, precision=nan, recall=nan, f1=nan)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1   = 2 * prec * rec / (prec + rec) if (not np.isnan(prec) and not np.isnan(rec) and prec + rec > 0) else float("nan")
    return dict(N=n, N_pos=int(np.sum(y_true)), accuracy=acc, precision=prec, recall=rec, f1=f1)


# ---------------------------------------------------------------------------
# Arm 1: Synthetic
# ---------------------------------------------------------------------------

_JUDGE_MODELS = ["gpt-oss-120b", "qwen-2.5-72b", "llama-3.3-70b"]


def evaluate_synthetic(datasets: list[str]) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        probe_path = _REPO_ROOT / "data" / dataset / "probe_dataset_test.json"
        if not probe_path.exists():
            print(f"  [SKIP] probe_dataset_test.json not found: {probe_path}")
            continue

        with open(probe_path) as f:
            probe = pd.DataFrame(json.load(f))
        probe["label_bool"] = probe["label"] == "valid"

        judge_dfs: dict[str, pd.DataFrame] = {}
        for jm in _JUDGE_MODELS:
            try:
                resp_path = paths.find_synthetic_responses(dataset, jm, split="test")
            except FileNotFoundError:
                print(f"  [SKIP] No synthetic test predictions: dataset={dataset} judge={jm}")
                continue
            with open(resp_path) as f:
                responses = pd.DataFrame(json.load(f))
            judge_dfs[jm] = probe[["measurement_id", "label_bool"]].merge(
                responses[["measurement_id", "judgement"]].rename(columns={"judgement": f"j_{jm}"}),
                on="measurement_id", how="inner",
            )
            print(f"  Loaded: dataset={dataset} judge={jm} N={len(judge_dfs[jm])}")

        if not judge_dfs:
            continue

        for jm, df in judge_dfs.items():
            valid = df.dropna(subset=[f"j_{jm}"])
            if not valid.empty:
                rows.append({"dataset": dataset, "evaluator": jm,
                             **_binary_metrics(valid["label_bool"].to_numpy(bool),
                                               valid[f"j_{jm}"].to_numpy(bool))})

        if len(judge_dfs) >= 2:
            base = list(judge_dfs.values())[0][["measurement_id", "label_bool"]].copy()
            for jm, df in judge_dfs.items():
                base = base.merge(df[["measurement_id", f"j_{jm}"]], on="measurement_id", how="inner")
            jcols = [f"j_{jm}" for jm in judge_dfs]
            base["majority"] = base.apply(
                lambda r: sum(r[c] for c in jcols if pd.notna(r[c])) > sum(pd.notna(r[c]) for c in jcols) / 2,
                axis=1,
            )
            valid_mv = base.dropna(subset=["majority"])
            if not valid_mv.empty:
                rows.append({"dataset": dataset, "evaluator": f"majority ({', '.join(judge_dfs)})",
                             **_binary_metrics(valid_mv["label_bool"].to_numpy(bool),
                                               valid_mv["majority"].to_numpy(bool))})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Arm 2: Human
# ---------------------------------------------------------------------------

def evaluate_human(datasets: list[str], extraction_model: str = "gemma-3-27b") -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        try:
            responses_path, ext_date = paths.find_human_responses(dataset, extraction_model)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        print(f"  Human responses: {responses_path}")
        with open(responses_path) as f:
            human_records = [r for r in json.load(f) if r.get("judgement") is not None]
        if not human_records:
            print(f"  [SKIP] No non-skipped human judgements for {dataset}/{extraction_model}")
            continue
        print(f"  N non-skipped: {len(human_records)}")

        human_df = pd.DataFrame(human_records)
        y_human = human_df["judgement"].to_numpy(bool)

        # Match labels
        match_labels = gt_df = None
        try:
            config = load_dataset_config(dataset)
            gt_df = load_ground_truth(config)
            strict, fuzzy, threshold = get_matching_rules(dataset)
            ext_df = process_extraction_df(human_df.copy(), dataset, config)
            _, edges, edge_weights = cached_match(gt_df, ext_df, strict_matching=strict, fuzzy_matching=fuzzy)
            match_labels = np.zeros(len(ext_df), dtype=bool)
            for i, (_, ex_idx) in enumerate(edges):
                if edge_weights[i] > threshold:
                    match_labels[ex_idx] = True
            rows.append({"dataset": dataset, "extraction_model": extraction_model, "method": "match",
                         **_binary_metrics(y_human, match_labels)})
        except Exception as e:
            print(f"  [WARN] Match evaluation failed: {e}")

        # Combined judge labels
        try:
            with open(paths.find_combined(dataset, extraction_model, ext_date)) as f:
                combined_by_id = {r["measurement_id"]: r for r in json.load(f)}

            raw = [combined_by_id.get(r["measurement_id"], {}).get("judgement_combined") for r in human_records]
            valid_mask = np.array([j is not None for j in raw])
            judge_labels = np.array([bool(j) if j is not None else False for j in raw])

            if valid_mask.sum() > 0:
                rows.append({"dataset": dataset, "extraction_model": extraction_model, "method": "combined_judge",
                             **_binary_metrics(y_human[valid_mask], judge_labels[valid_mask])})

                if match_labels is not None:
                    combined = (match_labels | judge_labels)[valid_mask]
                    rows.append({"dataset": dataset, "extraction_model": extraction_model, "method": "match_or_judge",
                                 **_binary_metrics(y_human[valid_mask], combined)})

                    # Diagnostic: human=Valid, match=Invalid, judge=Valid
                    case_idxs = np.where(y_human & ~match_labels & judge_labels & valid_mask)[0]
                    print(f"\n  --- Human=Valid, Match=Invalid, Judge=Valid: {len(case_idxs)} cases ---")
                    for idx in case_idxs[:5]:
                        r = human_records[int(idx)]
                        print(f"\n  [{idx}] mid={r.get('measurement_id')}  doc={r.get('document_id')}  attribute={r.get('attribute')}")
                        print(f"        extracted:   value={r.get('value')}  units={r.get('units')}  name={r.get('name')}")
                        if gt_df is not None:
                            gt_rows = gt_df[(gt_df["document_id"] == r.get("document_id")) & (gt_df["attribute"] == r.get("attribute"))]
                            if gt_rows.empty:
                                print(f"        ground truth: (no rows for this doc+attribute)")
                            else:
                                for _, gt_row in gt_rows.iterrows():
                                    print(f"        ground truth: value={gt_row.get('value')}  units={gt_row.get('units')}")
                    if len(case_idxs) > 5:
                        print(f"  ... ({len(case_idxs) - 5} more)")

        except FileNotFoundError as e:
            print(f"  [WARN] No combined.json — skipping judge evaluation: {e}")
        except Exception as e:
            print(f"  [WARN] Judge evaluation failed: {e}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _fmt(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no results)"
    return df.to_string(index=False, formatters={c: "{:.3f}".format for c in ["accuracy", "precision", "recall", "f1"]})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["pond", "nfix"])
    p.add_argument("--extraction-model", default="gemma-3-27b")
    p.add_argument("--synthetic", action="store_true", default=False)
    p.add_argument("--human", action="store_true", default=False)
    p.add_argument("--output", default=None, metavar="CSV")
    args = p.parse_args(argv)

    run_synthetic = args.synthetic or not args.human
    run_human = args.human or not args.synthetic

    all_frames = []

    if run_synthetic:
        print("\n=== Synthetic evaluation ===")
        df = evaluate_synthetic(args.datasets)
        print(_fmt(df))
        if not df.empty:
            all_frames.append(df.assign(arm="synthetic"))

    if run_human:
        print("\n=== Human validation evaluation ===")
        df = evaluate_human(args.datasets, extraction_model=args.extraction_model)
        print(_fmt(df))
        if not df.empty:
            all_frames.append(df.assign(arm="human"))

    if args.output and all_frames:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(all_frames, ignore_index=True).to_csv(out, index=False)
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
