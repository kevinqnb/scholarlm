"""
Human validation Streamlit app for extraction results.

Presents a random sample from a final.json extraction run, showing the source PDF
page, OCR text, and extracted data point.  Collects binary valid / invalid
judgements; invalid points receive an error-type label.

Output: responses.json written to the standard judge path, compatible with
run_judge_combine.py (judge key "human").

Output path:
    data/experiments/{dataset}/judge/{extraction_model}/{extraction_date}/human/{date}/responses.json

Usage
-----
    # Run from the repo root:
    streamlit run experiments/validation.py -- \\
        --dataset pond --extraction-model gemma-3-27b

    # Pin extraction date and sample size:
    streamlit run experiments/validation.py -- \\
        --dataset pond --extraction-model gemma-3-27b \\
        --extraction-date 2026_04_01 --n-sample 50

    # Override PDF or OCR directory:
    streamlit run experiments/validation.py -- \\
        --dataset pond --extraction-model gemma-3-27b \\
        --pdf-dir /path/to/pdfs --ocr-dir /path/to/ocr_output_raw
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — must come before any local imports
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXPERIMENTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_EXPERIMENTS_DIR))

# Ensure relative paths in dataset configs resolve correctly.
os.chdir(_REPO_ROOT)

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import judge_common
from run_extraction import load_dataset_config
import paths

try:
    from pdf2image import convert_from_path as _pdf_convert
    _PDF2IMAGE_OK = True
except ImportError:
    _PDF2IMAGE_OK = False


# ---------------------------------------------------------------------------
# Error-type vocabulary
# ---------------------------------------------------------------------------

_ERROR_TYPES: dict[str, tuple[str, str]] = {
    "hallucination": (
        "Hallucination",
        "Value absent from or not supported by the source text.",
    ),
    "disorientation": (
        "Disorientation",
        "Value present but attributed to the wrong entity, page, or attribute.",
    ),
    "deviation": (
        "Deviation",
        "Value present but incorrect — wrong number, unit confusion, or out of scope.",
    ),
}


# ---------------------------------------------------------------------------
# Argument parsing (parsed once per process)
# ---------------------------------------------------------------------------


@st.cache_resource
def _get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--dataset", required=True, help="Dataset name (e.g. 'pond', 'nfix').")
    p.add_argument("--extraction-model", required=True, help="Extraction model short name.")
    p.add_argument("--extraction-date", default=None, help="Date tag YYYY_mm_dd.")
    p.add_argument("--n-sample", type=int, default=100, help="Number of data points to sample (default: 100).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42).")
    p.add_argument("--pdf-dir", default=None, metavar="DIR",
                   help="PDF directory. Defaults to {data_dir}/pdfs/.")
    p.add_argument("--ocr-dir", default=None, metavar="DIR",
                   help="OCR directory. Defaults to {data_dir}/ocr_output_raw/.")
    p.add_argument("--judge-date", default=None, help="Date tag for output directory (default: today).")
    args, _ = p.parse_known_args(sys.argv[1:])
    return args


# ---------------------------------------------------------------------------
# Data loading (cached across reruns)
# ---------------------------------------------------------------------------


@st.cache_data
def _load_sample(
    dataset: str,
    model: str,
    date: str | None,
    n: int,
    seed: int,
    ocr_dir: str | None,
) -> tuple[list[dict], list[str], str]:
    """Return (sample, documents, resolved_extraction_date)."""
    input_file = paths.find_extraction_final(dataset, model, date)
    resolved_date = input_file.parent.name

    with open(input_file) as f:
        all_records: list[dict] = json.load(f)

    cfg = load_dataset_config(dataset)
    data_dir = Path(cfg.data_dir)
    if not data_dir.is_absolute():
        data_dir = _REPO_ROOT / data_dir
    effective_ocr = ocr_dir or str(data_dir / "ocr_output_raw")
    docs = judge_common.load_documents_for_dataset(cfg, effective_ocr)

    rng = random.Random(seed)
    sample = rng.sample(all_records, k=min(n, len(all_records)))
    sample.sort(key=lambda r: (r.get("document_id", 0), _first_page(r) or 0))
    return sample, docs, resolved_date


@st.cache_data
def _load_pdf_page_bytes(pdf_path: str, page: int) -> bytes | None:
    """Render one PDF page to PNG bytes (0-indexed page number). Cached."""
    if not _PDF2IMAGE_OK:
        return None
    try:
        imgs = _pdf_convert(pdf_path, dpi=130, first_page=page + 1, last_page=page + 1)
        if not imgs:
            return None
        buf = io.BytesIO()
        imgs[0].save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Page-number helpers
# ---------------------------------------------------------------------------


def _first_page(r: dict) -> int | None:
    pn = r.get("page_number")
    if isinstance(pn, list):
        return next((p for p in pn if p is not None), None)
    return pn if pn is not None else None


def _all_pages(r: dict) -> list[int]:
    pn = r.get("page_number")
    if isinstance(pn, list):
        return [p for p in pn if p is not None]
    return [pn] if pn is not None else []


# ---------------------------------------------------------------------------
# Persistence — load / save responses.json
# ---------------------------------------------------------------------------


def _out_path(dataset: str, model: str, extraction_date: str, judge_date: str | None) -> Path:
    return paths.judge(dataset, model, extraction_date, "human", judge_date) / "responses.json"


def _load_existing(out: Path) -> dict[int, dict]:
    """Load partial results from a prior session, keyed by measurement_id."""
    if not out.exists():
        return {}
    try:
        with open(out) as f:
            records: list[dict] = json.load(f)
    except Exception:
        return {}
    return {
        r["measurement_id"]: {
            "measurement_id": r["measurement_id"],
            "judgement": r.get("judgement"),
            "error_type": r.get("error_type"),
        }
        for r in records
        if "measurement_id" in r
    }


def _save(sample: list[dict], results: dict[int, dict], out: Path) -> None:
    """Write all validated records to responses.json."""
    output = []
    for record in sample:
        mid = record.get("measurement_id")
        if mid not in results:
            continue
        res = results[mid]
        output.append(record | {
            "judgement": res["judgement"],
            "judgement_model": "human",
            "error_type": res.get("error_type"),
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Data card
# ---------------------------------------------------------------------------


def _show_data_card(record: dict, cfg) -> None:
    """Render the extracted data point as a structured info panel."""
    attr = record.get("attribute", "")
    attr_info = cfg.attribute_info_dict.get(attr, {})

    st.markdown(f"#### `{attr}`")
    if desc := attr_info.get("description", ""):
        st.caption(desc[:300])

    value = record.get("value")
    units = record.get("units")
    value_str = str(value) + (f"  {units}" if units else "")
    st.markdown(f"**Value:** `{value_str}`")

    # Entity + event fields
    entity_keys = set(cfg.entity_schema.model_fields.keys())
    event_keys = (
        set(cfg.measurement_event_schema.model_fields.keys())
        if cfg.measurement_event_schema is not None
        else set()
    )
    entity_vals = {k: record[k] for k in sorted(entity_keys | event_keys) if record.get(k) is not None}
    if entity_vals:
        st.markdown("**Entity / Event:**")
        for k, v in entity_vals.items():
            st.markdown(f"- `{k}` : {v}")

    # Provenance
    pages = _all_pages(record)
    src = record.get("source", [])
    src_str = ", ".join(src) if isinstance(src, list) else str(src or "—")
    page_str = ", ".join(str(p) for p in pages) if pages else "—"
    st.markdown(f"**Source:** {src_str} &nbsp;·&nbsp; **Page(s):** {page_str}")

    table_numbers = record.get("table_number")
    if isinstance(table_numbers, list):
        tnums = [t for t in table_numbers if t is not None]
        if tnums:
            st.markdown(f"**Table(s):** {', '.join(str(t) for t in tnums)}")

    with st.expander("Paper metadata", expanded=False):
        for k in ("document_id", "title", "author", "year"):
            if v := record.get(k):
                st.markdown(f"- **{k}:** {v}")
        st.markdown(f"- **measurement_id:** `{record.get('measurement_id')}`")


# ---------------------------------------------------------------------------
# Record a judgement and advance
# ---------------------------------------------------------------------------


def _record_and_advance(
    record: dict,
    judgement: bool | None,
    error_type: str | None,
    sample: list[dict],
    out: Path,
) -> None:
    mid = record["measurement_id"]
    st.session_state.results[mid] = {
        "measurement_id": mid,
        "judgement": judgement,
        "error_type": error_type,
    }
    _save(sample, st.session_state.results, out)
    st.session_state.idx += 1
    st.session_state.awaiting_error = False
    st.rerun()


# ---------------------------------------------------------------------------
# Item view
# ---------------------------------------------------------------------------


def _show_item(
    idx: int,
    n_total: int,
    record: dict,
    docs: list[str],
    pdf_dir: Path,
    cfg,
    sample: list[dict],
    out: Path,
) -> None:
    pages = _all_pages(record)
    first = pages[0] if pages else None
    doc_id = record.get("document_id", "")
    document = docs.get(doc_id, "") if isinstance(docs, dict) else ""
    page_text = judge_common.extract_page_text(document, pages)

    st.markdown(f"### Item {idx + 1} of {n_total}")
    st.divider()

    left, right = st.columns([0.55, 0.45])

    with left:
        if _PDF2IMAGE_OK and first is not None:
            pdf_path = str(pdf_dir / f"{doc_id}.pdf")
            img_bytes = _load_pdf_page_bytes(pdf_path, first)
            if img_bytes is not None:
                st.image(img_bytes, caption=f"{doc_id} · page {first}", use_container_width=True)
            else:
                st.warning(f"PDF not found or unreadable: `{pdf_path}`")
        elif not _PDF2IMAGE_OK:
            st.info("Install `pdf2image` for PDF preview.")

        with st.expander("OCR text (relevant pages)", expanded=True):
            display_text = page_text[:8000] + ("\n…[truncated]" if len(page_text) > 8000 else "")
            st.code(display_text, language=None)

    with right:
        _show_data_card(record, cfg)

    st.divider()

    # Validation controls
    if not st.session_state.awaiting_error:
        st.markdown("**Is this extraction correct?**")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("✓  Valid", type="primary", use_container_width=True, key="btn_valid"):
                _record_and_advance(record, True, None, sample, out)
        with c2:
            if st.button("✗  Invalid", type="secondary", use_container_width=True, key="btn_invalid"):
                st.session_state.awaiting_error = True
                st.rerun()
        with c3:
            if st.button("—  Skip", use_container_width=True, key="btn_skip"):
                _record_and_advance(record, None, None, sample, out)
    else:
        st.markdown("**Select the error type:**")
        for key, (label, desc) in _ERROR_TYPES.items():
            if st.button(f"**{label}** — {desc}", use_container_width=True, key=f"err_{key}"):
                _record_and_advance(record, False, key, sample, out)
        if st.button("← Back", key="btn_back"):
            st.session_state.awaiting_error = False
            st.rerun()


# ---------------------------------------------------------------------------
# Summary view
# ---------------------------------------------------------------------------


def _show_summary(
    results: dict[int, dict],
    out: Path,
    n_reviewed: int,
    n_total: int,
) -> None:
    n_valid = sum(1 for r in results.values() if r.get("judgement") is True)
    n_invalid = sum(1 for r in results.values() if r.get("judgement") is False)
    n_skip = sum(1 for r in results.values() if r.get("judgement") is None)

    if n_reviewed >= n_total:
        st.success("All data points reviewed!")
    else:
        st.info(f"Session ended — {n_reviewed} of {n_total} data points reviewed.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Valid", n_valid)
    c2.metric("Invalid", n_invalid)
    c3.metric("Skipped", n_skip)

    if n_invalid > 0:
        by_type: dict[str, int] = {}
        for r in results.values():
            if r.get("judgement") is False:
                et = r.get("error_type") or "unclassified"
                by_type[et] = by_type.get(et, 0) + 1
        st.write("**Error type breakdown:**")
        for et, cnt in sorted(by_type.items()):
            st.write(f"- {et}: {cnt}")

    if results:
        st.caption(f"Results saved to `{out}`")

    if st.button("Start Over"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(layout="wide", page_title="Validation", initial_sidebar_state="collapsed")

    args = _get_args()

    try:
        cfg = load_dataset_config(args.dataset)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    try:
        sample, docs, extraction_date = _load_sample(
            args.dataset, args.extraction_model, args.extraction_date,
            args.n_sample, args.seed, args.ocr_dir,
        )
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    out = _out_path(args.dataset, args.extraction_model, extraction_date, args.judge_date)
    n_total = len(sample)

    data_dir = Path(cfg.data_dir)
    if not data_dir.is_absolute():
        data_dir = _REPO_ROOT / data_dir
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else data_dir / "pdfs"

    # Session state init — runs once per browser session; auto-resumes from disk.
    if "initialized" not in st.session_state:
        existing = _load_existing(out)
        st.session_state.results = existing
        validated_ids = set(existing.keys())
        st.session_state.idx = next(
            (i for i, r in enumerate(sample) if r.get("measurement_id") not in validated_ids),
            n_total,
        )
        st.session_state.awaiting_error = False
        st.session_state.done = False
        st.session_state.initialized = True

    n_validated = len(st.session_state.results)
    idx = st.session_state.idx

    # Header
    st.title("Extraction Validation")
    h1, h2 = st.columns([5, 1])
    with h1:
        st.caption(
            f"Dataset: **{args.dataset}** · Model: **{args.extraction_model}** · "
            f"Date: **{extraction_date}** · Sample: **{n_total}**"
        )
    with h2:
        if st.button("Save & Exit", use_container_width=True):
            _save(sample, st.session_state.results, out)
            st.session_state.done = True
            st.rerun()

    st.progress(
        n_validated / n_total if n_total else 0,
        text=f"{n_validated} / {n_total} validated",
    )

    if st.session_state.done or idx >= n_total:
        _show_summary(st.session_state.results, out, n_validated, n_total)
        return

    _show_item(idx, n_total, sample[idx], docs, pdf_dir, cfg, sample, out)


if __name__ == "__main__":
    main()
