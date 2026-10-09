"""Row identity and provenance for the 'real' cells of calibration predictions.pkl.

analysis/calibration.py stores, with every real cell's score arrays, the
measurement_id / document_id / attribute of each scored row, the sha256 of the
final.json and combined.json they were scored from, and the Platt sample's ids.
Consumers (common/meta_inputs.stored_prediction_rows, decision_threshold.py) then join scores to rows by
measurement_id and verify all of it, instead of rebuilding the row selection by
position. Import-side-effect free so it can be unit tested on a hand-built fixture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.common.provenance import sha256_file

ID_COLS = ("measurement_id", "document_id", "attribute")
# Keys a real cell must carry besides probe_probs / ntp_probs / labels / platt.
PROVENANCE_KEYS = ("measurement_ids", "document_ids", "attributes", "final_sha256", "combined_sha256",
                   "calibration_config_id", "seed", "platt_measurement_ids", "excluded_documents")


def real_cell_provenance(final_df, idx, platt_idx, excluded_docs, final_path, combined_path, calibration_config_id, seed) -> dict:
    """Provenance for one real cell. ``idx`` are the final_df positions scored in the
    cell, ``platt_idx`` the positions of the Platt-fit sample, ``excluded_docs`` the
    documents the cell excludes. Asserts the identities are consistent before storing."""
    assert final_df["measurement_id"].is_unique, "final.json measurement_id is not unique"
    sub = final_df.iloc[np.asarray(idx)]
    mids = sub["measurement_id"].to_numpy(dtype=np.int64)
    pmids = final_df["measurement_id"].iloc[np.asarray(platt_idx)].to_numpy(dtype=np.int64)
    assert not set(mids.tolist()) & set(pmids.tolist()), "test rows overlap the Platt sample"
    assert not sub["document_id"].isin(excluded_docs).any(), "scored rows include an excluded document"
    return {
        "measurement_ids": mids,
        "document_ids": sub["document_id"].tolist(),
        "attributes": sub["attribute"].tolist(),
        "final_sha256": sha256_file(final_path),
        "combined_sha256": sha256_file(combined_path),
        "calibration_config_id": calibration_config_id,
        "seed": seed,
        "platt_measurement_ids": pmids,
        "excluded_documents": sorted(excluded_docs),
    }


def check_real_cell(cell: dict, final_df: pd.DataFrame, excluded_docs: set, final_sha256: str,
                    calibration_config_id: str) -> pd.DataFrame:
    """Verify a stored real cell against the final.json now in use and return its rows
    (measurement_id, document_id, attribute, ntp_prob, probe_prob, label), in the
    cell's order. Raises on any mismatch -- there is no positional fallback."""
    missing = [k for k in ("probe_probs", "ntp_probs", "labels", *PROVENANCE_KEYS) if k not in cell]
    if missing:
        raise ValueError(f"stored cell lacks {missing} -- predictions.pkl predates row provenance; rerun calibration")
    if cell["calibration_config_id"] != calibration_config_id:
        raise ValueError(f"cell was built by {cell['calibration_config_id']!r}, asked for {calibration_config_id!r}")
    if cell["final_sha256"] != final_sha256:
        raise ValueError("final.json changed since predictions.pkl was built (sha256 mismatch) -- rerun calibration")
    if set(cell["excluded_documents"]) != set(excluded_docs):
        raise ValueError("the cell's excluded documents differ from the probe's synthetic-training documents")
    n = len(cell["measurement_ids"])
    lens = {k: len(cell[k]) for k in ("probe_probs", "ntp_probs", "labels", "document_ids", "attributes")}
    if any(v != n for v in lens.values()):
        raise ValueError(f"stored cell arrays disagree in length (ids {n}): {lens}")
    if not final_df["measurement_id"].is_unique:
        raise ValueError("final.json measurement_id is not unique")
    out = pd.DataFrame({"measurement_id": np.asarray(cell["measurement_ids"], dtype=np.int64),
                        "document_id": cell["document_ids"], "attribute": cell["attributes"],
                        "ntp_prob": np.asarray(cell["ntp_probs"], dtype=float),
                        "probe_prob": np.asarray(cell["probe_probs"], dtype=float),
                        "label": np.asarray(cell["labels"], dtype=bool)})
    if not out["measurement_id"].is_unique:
        raise ValueError("stored measurement_ids are not unique")
    expect = final_df.loc[~final_df["document_id"].isin(excluded_docs), list(ID_COLS)]
    if set(out["measurement_id"]) != set(expect["measurement_id"]):
        raise ValueError(f"stored ids ({len(out)}) are not the final.json ids outside the excluded documents ({len(expect)})")
    joined = out.merge(expect, on="measurement_id", how="left", suffixes=("", "__final"), validate="one_to_one")
    for c in ("document_id", "attribute"):
        bad = joined[c] != joined[c + "__final"]
        if bad.any():
            raise ValueError(f"{int(bad.sum())} stored rows disagree with final.json on {c!r} "
                             f"(first measurement_id {joined.loc[bad, 'measurement_id'].iloc[0]})")
    if set(cell["platt_measurement_ids"].tolist()) & set(out["measurement_id"]):
        raise ValueError("the Platt sample overlaps the scored rows")
    if not (np.isfinite(out["ntp_prob"]).all() and np.isfinite(out["probe_prob"]).all()):
        raise ValueError("stored predictions contain non-finite values")
    return out
