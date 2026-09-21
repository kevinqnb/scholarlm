"""Official MeasEval (SemEval-2021 Task 8) scoring for measeval extraction runs.

``analysis/ablation.py``'s ``get_matching_rules``/``recovery_rate``/``validity_rate``
compute this repo's own recovery/validity methodology, which is NOT the same
number as the published MeasEval leaderboard score -- see
``data/measeval/README.md``'s "Train/trial/eval and comparability" section. This
module instead exports a run's predictions into the exact TSV format the
official, unmodified scorer (``data/measeval/raw/eval/measeval-eval.py``)
expects, and invokes that script directly, to get a number that IS comparable
to published results.

Scope -- deliberately narrow, confirmed with Kevin before writing any of this
--------------------------------------------------------------------------
Only three of MeasEval's nine scored components are attempted here: **Quantity**,
**Unit**, and **MeasuredEntity**. Everything else is out of scope for this
module:

- **MeasuredProperty** (our ``property`` field) is never emitted.
- **Qualifier** -- a MeasEval annotation type, not to be confused with anything
  below -- is an open-text span (e.g. "under standard conditions") that nothing
  in this pipeline extracts. Not attempted.
- **Modifiers** -- MeasEval's name for the tag list our pipeline calls
  ``qualifiers`` (``IsMean``/``IsRange``/``HasTolerance``/etc., produced by
  ``MeasurementLM._parse_quantities()``). Also not attempted here, despite
  being cheap to add later (our atomic tags would need a mapping table onto
  MeasEval's fixed 11-value *compound* vocabulary, e.g.
  ``["IsMean","HasSD"]`` -> ``"IsMeanHasSD"``).
- **Relations** (HasQuantity, HasProperty, Qualifies) are not computed or
  reported. They cannot be *omitted* from the submission TSV, though: the
  official script unconditionally JSON-parses every MeasuredEntity row's
  ``other`` field for relation scoring and crashes on an empty one
  (``json.loads('')`` raises). Every MeasuredEntity row we emit therefore
  carries a placeholder ``other = {"HasQuantity": "<sibling Quantity annotId>"}``
  purely so the (unmodified, never-patched) third-party script doesn't crash --
  the HasQuantity/HasProperty/Qualifies numbers it computes as a side effect are
  never read back or reported by this module.

Span-identification heuristic -- documented in full, per Kevin's explicit request
-----------------------------------------------------------------------------
The official scorer keys everything on character offsets. Our extraction
pipeline never computes one -- it only copies text. This module recovers
offsets by searching for the extracted text, verbatim, in
``data/measeval/ocr_output_raw/{document_id}.txt`` (with the ``<page
number="N">...</page>`` wrapper stripped exactly as
``data/measeval/preprocessing.py`` wrote it, so offsets land on the same
coordinate system as ``ground_truth.json``'s own ``entity_start``/etc. columns).

- **Quantity.** Our pipeline splits the original phrase into separate
  ``value``/``units`` fields (e.g. ``"25"`` / ``"°C"``), but MeasEval's gold
  Quantity span is the one contiguous original phrase (``"25 °C"``). We try,
  in order: ``f"{value} {units}"``, then ``f"{value}{units}"``, then ``value``
  alone if ``units`` is null or neither concatenation is found verbatim. The
  first candidate that appears in the text is used.
- **MeasuredEntity.** We search ``name`` verbatim.
- **Disambiguation.** Measured directly against real ground truth (see
  ``data/measeval/README.md``): 38% of gold ``name`` spans and 16% of gold
  ``quantity`` spans occur more than once in their own paragraph, so a plain
  first-match ``str.find`` mislocates roughly a third of them. When a candidate
  string has more than one occurrence, the Quantity span is resolved first
  (leftmost occurrence when still ambiguous -- there is no other information to
  break the tie), then the Entity span is resolved by picking whichever of its
  occurrences is *nearest* to the chosen Quantity offset (gold entity/quantity
  pairs are always within the same sentence). This is a heuristic tie-break, not
  a guarantee of correctness.
- **No match found means the record is dropped, not guessed.** Every drop is
  counted and surfaced in the coverage report (see ``Coverage`` below) rather
  than silently shrinking the submission -- see CLAUDE.md's "quietly wrong
  result" failure mode.

Not attempted: improving this heuristic (e.g. fuzzy/normalized matching) if its
recovery rate turns out low on real predictions. That would be a measured
limitation to report, not something to loosen until numbers "look right."
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'experiments'))
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

import utils as paths
from experiments.run_extraction import load_dataset_config

MEASEVAL_ROOT = REPO_ROOT / "data" / "measeval"
OCR_DIR = MEASEVAL_ROOT / "ocr_output_raw"
RAW_DATA_DIR = MEASEVAL_ROOT / "raw" / "data"
DIRECTORY_FILE = MEASEVAL_ROOT / "directory.json"
EVAL_SCRIPT = MEASEVAL_ROOT / "raw" / "eval" / "measeval-eval.py"

# Matches exactly how data/measeval/preprocessing.py wrote the file
# (f'<page number="0">\n{text}\n</page>\n'), so stripping it recovers the same
# coordinate system ground_truth.json's own offset columns are computed against.
_PAGE_WRAPPER_RE = re.compile(r'^<page number="\d+">\n|\n</page>\s*$')

# Values the pipeline sometimes emits as the literal string "None" instead of a
# real null (an observed model-output quirk, not a serialization issue) --
# treated the same as a missing field for span lookup.
_NULLISH = {None, "None", ""}


# ---------------------------------------------------------------------------
# Text loading
# ---------------------------------------------------------------------------

def load_doc_text(document_id: str) -> str:
    """Load a measeval document's reference text, offsets-compatible with ground_truth.json."""
    raw = (OCR_DIR / f"{document_id}.txt").read_text(encoding="utf-8")
    return _PAGE_WRAPPER_RE.sub("", raw)


def doc_split(document_id: str) -> str:
    """train/trial/eval split for a document_id, per data/measeval/directory.json."""
    directory = json.loads(DIRECTORY_FILE.read_text())
    if document_id not in directory:
        raise KeyError(f"{document_id!r} not found in {DIRECTORY_FILE}")
    return directory[document_id]["source_split"]


# ---------------------------------------------------------------------------
# Span identification
# ---------------------------------------------------------------------------

def _find_all(haystack: str, needle: str) -> list[tuple[int, int]]:
    """All (start, end) occurrences of ``needle`` in ``haystack``, non-overlapping."""
    positions = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        positions.append((idx, idx + len(needle)))
        start = idx + 1
    return positions


def _nearest(candidates: list[tuple[int, int]], anchor: tuple[int, int] | None) -> tuple[int, int]:
    """Pick the candidate span closest (by midpoint) to ``anchor``; leftmost if no anchor."""
    if anchor is None or len(candidates) == 1:
        return candidates[0]
    anchor_mid = (anchor[0] + anchor[1]) / 2
    return min(candidates, key=lambda c: abs((c[0] + c[1]) / 2 - anchor_mid))


def resolve_quantity_span(doc_text: str, value, units) -> tuple[int, int, str] | None:
    """Locate the Quantity span for an extracted (value, units) pair.

    Tries the contiguous "value units" / "valueunits" forms first (matching
    MeasEval's own single-span Quantity convention), falling back to ``value``
    alone. Returns (start, end, text) for the first candidate found verbatim in
    ``doc_text``, or None if none of them appear at all.
    """
    if value in _NULLISH:
        return None
    value = str(value)
    candidates = []
    if units not in _NULLISH:
        units_str = str(units)
        candidates.append(f"{value} {units_str}")
        candidates.append(f"{value}{units_str}")
    candidates.append(value)

    for candidate in candidates:
        positions = _find_all(doc_text, candidate)
        if positions:
            start, end = _nearest(positions, anchor=None)  # leftmost tie-break
            return start, end, doc_text[start:end]
    return None


def resolve_entity_span(
    doc_text: str, name, anchor: tuple[int, int] | None
) -> tuple[int, int, str] | None:
    """Locate the MeasuredEntity span for an extracted ``name``, nearest to ``anchor``."""
    if name in _NULLISH:
        return None
    positions = _find_all(doc_text, str(name))
    if not positions:
        return None
    start, end = _nearest(positions, anchor)
    return start, end, doc_text[start:end]


# ---------------------------------------------------------------------------
# TSV export
# ---------------------------------------------------------------------------

@dataclass
class Coverage:
    """Records how many candidate rows survived span-recovery, and why the rest didn't.

    Printed alongside the official scorer's numbers so a shrunk submission is
    visible, not silently folded into a lower recall.
    """

    total_records: int = 0
    quantity_dropped_no_value: int = 0
    quantity_dropped_unlocatable: int = 0
    quantity_located: int = 0
    entity_dropped_no_name: int = 0
    entity_dropped_unlocatable: int = 0
    entity_located: int = 0

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def build_submission_tsv(df: pd.DataFrame, out_dir: Path) -> Coverage:
    """Write one MeasEval-format TSV per document into ``out_dir``.

    ``df`` must have ``document_id``, ``name``, ``value``, ``units`` columns
    (one row per extracted measurement). Only Quantity and MeasuredEntity rows
    are emitted -- see module docstring for scope. Every emitted row's ``text``
    field is the literal substring at its offsets (required by the official
    scorer's length validator), never the model's own copy.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coverage = Coverage()

    for document_id, group in df.groupby("document_id"):
        doc_text = load_doc_text(document_id)
        rows = []
        annot_set = 0

        for _, rec in group.iterrows():
            coverage.total_records += 1

            quantity = resolve_quantity_span(doc_text, rec.get("value"), rec.get("units"))
            if quantity is None:
                if rec.get("value") in _NULLISH:
                    coverage.quantity_dropped_no_value += 1
                else:
                    coverage.quantity_dropped_unlocatable += 1
                continue
            coverage.quantity_located += 1
            qstart, qend, qtext = quantity

            annot_set += 1
            # annotId must fully match vladiate's r'T?\d*-?\d+' -- letters not
            # allowed. Follows gold's own "T<type-slot>-<annotSet>" convention
            # (see data/measeval/README.md's worked example): 1=Quantity,
            # 2=MeasuredEntity.
            q_annot_id = f"T1-{annot_set}"
            units = rec.get("units")
            other = json.dumps({"unit": str(units)}) if units not in _NULLISH else ""
            rows.append({
                "docId": document_id, "annotSet": annot_set, "annotType": "Quantity",
                "startOffset": qstart, "endOffset": qend, "annotId": q_annot_id,
                "text": qtext, "other": other,
            })

            name = rec.get("name")
            if name in _NULLISH:
                coverage.entity_dropped_no_name += 1
                continue
            entity = resolve_entity_span(doc_text, name, anchor=(qstart, qend))
            if entity is None:
                coverage.entity_dropped_unlocatable += 1
                continue
            coverage.entity_located += 1
            estart, eend, etext = entity
            e_annot_id = f"T2-{annot_set}"
            # Placeholder relation only -- see module docstring "Relations" section.
            rows.append({
                "docId": document_id, "annotSet": annot_set, "annotType": "MeasuredEntity",
                "startOffset": estart, "endOffset": eend, "annotId": e_annot_id,
                "text": etext, "other": json.dumps({"HasQuantity": q_annot_id}),
            })

        if rows:
            pd.DataFrame(rows, columns=[
                "docId", "annotSet", "annotType", "startOffset", "endOffset",
                "annotId", "text", "other",
            ]).to_csv(out_dir / f"{document_id}.tsv", sep="\t", index=False)

    return coverage


# ---------------------------------------------------------------------------
# Official scorer invocation
# ---------------------------------------------------------------------------

def run_official_eval(submission_dir: Path, gold_dir: Path, mode: str = "class") -> str:
    """Invoke the unmodified official scorer as a subprocess. Returns its stdout.

    measeval-eval.py opens "../fileCategories.txt" as a path relative to its own
    location, so it must be run with cwd=EVAL_SCRIPT.parent; -g/-s are passed as
    absolute paths so that requirement doesn't also constrain them. Never passes
    -l/--limit: that flag silently restricts gold to docs present in the
    submission, which is locally valid but not leaderboard-comparable.
    """
    submission_dir = Path(submission_dir).resolve()
    gold_dir = Path(gold_dir).resolve()
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "-i", "",
        "-g", str(gold_dir) + "/",
        "-s", str(submission_dir) + "/",
        "-m", mode,
    ]
    result = subprocess.run(
        cmd, cwd=EVAL_SCRIPT.parent, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"measeval-eval.py exited {result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    # The script calls a bare exit() (code 0) on a validation failure -- a
    # nonzero returncode alone won't catch it, so check the message too.
    if "You have invalid tsv data in your submission" in result.stdout:
        raise RuntimeError(
            f"measeval-eval.py rejected the submission TSVs as invalid "
            f"(exited 0, but reported invalid data):\n{result.stdout}"
        )
    return result.stdout


# ---------------------------------------------------------------------------
# Calibration placeholder rows -- a second, independent workaround from the
# HasQuantity placeholder above. Discovered empirically: measeval-eval.py
# crashes (ValueError: Columns must be same length as key) when a submission
# has literally zero MeasuredProperty or Qualifier rows -- which is guaranteed
# under this module's scope, since we never emit either. The crash is
# `propertyMatches.apply(lambda x: calcF1(x), axis=1)` (and the identical
# pattern for qualifierMatches): pandas can't safely probe calcF1 (which reads
# row.aText/row.gText) on a genuinely empty frame, and falls back to returning
# the frame unchanged instead of an empty Series. Not a pandas-version issue
# (verified) -- a latent bug in the 2021 script that a normal MeasEval
# submission would never trigger, since normal submissions guess *something*
# for every component.
#
# Fix (confirmed with Kevin): inject one small, fully self-authored
# calibration document into both submission and gold dirs on every run, with a
# Quantity+MeasuredProperty+Qualifier triple that matches itself perfectly (we
# control both sides, so alignment is exact by construction). This keeps
# propertyMatches/qualifierMatches non-empty so the script completes. It does
# NOT touch MeasuredEntity (the calibration doc has no MeasuredEntity row), so
# MeasuredEntity's reported numbers are never contaminated. It DOES add
# exactly one guaranteed true positive to Quantity and Unit, which
# `_correct_for_calibration` below subtracts back out before reporting, so the
# numbers Kevin sees reflect only the real run, not our own scaffolding.
#
# S0927024813002961 is a real MeasEval article_id (from data/measeval's own
# worked README example) reused here only so the script's unconditional
# subject-category lookup (`cats[docId.split("-")[0]]`, which KeyErrors on an
# unrecognized prefix) succeeds; "-9999" is not a real MeasEval document id
# (verified against data/measeval/directory.json).
# ---------------------------------------------------------------------------

_CALIBRATION_DOC_ID = "S0927024813002961-9999"
_CALIBRATION_TEXT = "The control sample weighed 5 g under standard conditions."


def _calibration_tsv_rows() -> list[dict]:
    text = _CALIBRATION_TEXT

    def span(s: str) -> tuple[int, int]:
        i = text.index(s)
        return i, i + len(s)

    (qs, qe), (ps, pe), (qls, qle) = span("5 g"), span("weighed"), span("under standard conditions")
    return [
        {"docId": _CALIBRATION_DOC_ID, "annotSet": 1, "annotType": "Quantity",
         "startOffset": qs, "endOffset": qe, "annotId": "T1-1",
         "text": text[qs:qe], "other": json.dumps({"unit": "g"})},
        {"docId": _CALIBRATION_DOC_ID, "annotSet": 1, "annotType": "MeasuredProperty",
         "startOffset": ps, "endOffset": pe, "annotId": "T3-1",
         "text": text[ps:pe], "other": json.dumps({"HasQuantity": "T1-1"})},
        {"docId": _CALIBRATION_DOC_ID, "annotSet": 1, "annotType": "Qualifier",
         "startOffset": qls, "endOffset": qle, "annotId": "T4-1",
         "text": text[qls:qle], "other": json.dumps({"Qualifies": "T1-1"})},
    ]


def write_calibration_doc(directory: Path) -> None:
    """Write the calibration doc's TSV into ``directory`` (used for both gold and submission)."""
    rows = _calibration_tsv_rows()
    pd.DataFrame(rows, columns=[
        "docId", "annotSet", "annotType", "startOffset", "endOffset", "annotId", "text", "other",
    ]).to_csv(directory / f"{_CALIBRATION_DOC_ID}.tsv", sep="\t", index=False)


def _correct_for_calibration(scores: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Subtract the calibration doc's known, exact contribution from Quantity/Unit.

    The calibration doc contributes exactly one perfect match (EM=1.0, F1=1.0,
    true positive, no corresponding false positive/negative) to Quantity and
    Unit specifically -- never to MeasuredEntity, which the calibration doc
    doesn't touch. Recomputes precision/recall/F-measure and the EM/F1 means
    from the corrected counts rather than trusting the script's own printed
    aggregate, since those means can't be corrected by simple subtraction.
    """
    corrected = {k: dict(v) for k, v in scores.items()}
    for annot_type in ("Quantity", "Unit"):
        raw = scores.get(annot_type, {})
        tp = raw.get("True positives (matching rows)")
        fp = raw.get("False positives (submission only)")
        fn = raw.get("False negatives (gold only)")
        em_mean = raw.get("Exact Match Score")
        f1_mean = raw.get("F1 (Overlap) Score")
        if None in (tp, fp, fn, em_mean, f1_mean) or tp < 1:
            # Nothing to correct (e.g. no data at all) -- leave as printed.
            continue

        n_total_raw = tp + fp + fn
        n_total = n_total_raw - 1
        tp_c = tp - 1

        c = corrected[annot_type]
        c["True positives (matching rows)"] = tp_c
        c["False positives (submission only)"] = fp
        c["False negatives (gold only)"] = fn
        c["Precision"] = tp_c / (tp_c + fp) if (tp_c + fp) > 0 else float("nan")
        c["Recall"] = tp_c / (tp_c + fn) if (tp_c + fn) > 0 else float("nan")
        if c["Precision"] + c["Recall"] > 0:
            c["F-measure"] = 2 * c["Precision"] * c["Recall"] / (c["Precision"] + c["Recall"])
        else:
            c["F-measure"] = float("nan")
        c["Exact Match Score"] = (em_mean * n_total_raw - 1.0) / n_total if n_total > 0 else float("nan")
        c["F1 (Overlap) Score"] = (f1_mean * n_total_raw - 1.0) / n_total if n_total > 0 else float("nan")
    return corrected


_CLASS_LINE_RE = re.compile(
    r"^(True positives \(matching rows\)|False positives \(submission only\)|"
    r"False negatives \(gold only\)|Precision|Recall|F-measure|"
    r"Exact Match Score|F1 \(Overlap\) Score) for (.+?): (.+)$"
)

# The 3 components this module attempts; everything else the official scorer
# also prints (MeasuredProperty, Qualifier, modifier, HasQuantity, HasProperty,
# Qualifies) is intentionally discarded here -- see module docstring "Scope".
_IN_SCOPE_TYPES = {"Quantity", "MeasuredEntity", "Unit"}


def parse_class_scores(stdout: str) -> dict[str, dict[str, float]]:
    """Pull only the Quantity/MeasuredEntity/Unit metric lines out of -m class stdout."""
    scores: dict[str, dict[str, float]] = {t: {} for t in _IN_SCOPE_TYPES}
    for line in stdout.splitlines():
        m = _CLASS_LINE_RE.match(line.strip())
        if not m:
            continue
        metric, annot_type, value = m.groups()
        if annot_type not in _IN_SCOPE_TYPES:
            continue
        try:
            scores[annot_type][metric] = float(value)
        except ValueError:
            scores[annot_type][metric] = value  # e.g. "nan"
    return scores


# ---------------------------------------------------------------------------
# Loading extraction results
# ---------------------------------------------------------------------------

def load_extraction_df(experiment_id: str) -> pd.DataFrame:
    """Load an extraction run's final.json into a (document_id, name, value, units) frame.

    Fails loud if the run isn't a measeval run -- this module is measeval-only.
    Deliberately does NOT run analysis.baselines.normalize_baseline_extraction:
    that rewrites value/units into ground-truth notation (parses value to a
    float, canonicalizes units) for this repo's own fuzzy/strict matching, which
    would destroy the verbatim text this module's span search depends on.
    """
    run_dir = paths.find_result_dir(experiment_id)
    metadata = paths.load_run_metadata(run_dir)
    if metadata is None or metadata.get("dataset") != "measeval":
        raise ValueError(
            f"{experiment_id!r} is not a measeval run "
            f"(run_metadata.json dataset={metadata.get('dataset') if metadata else None!r}); "
            f"analysis/measeval_evaluation.py is measeval-only."
        )
    with open(run_dir / "final.json") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    missing = {"document_id", "name", "value", "units"} - set(df.columns)
    if missing:
        raise ValueError(f"final.json is missing required columns: {missing}")
    return df[["document_id", "name", "value", "units"]].copy()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def evaluate(
    experiment_id: str,
    *,
    dev: bool = False,
    out_dir: Path | None = None,
) -> dict:
    """Export an extraction run to MeasEval TSVs and score it with the official scorer.

    dev=False (default): every document in the run must belong to the official
    `eval` split (raw/data/eval/tsv/) -- scored against the FULL eval gold set,
    for a number that is actually comparable to published results. Raises if any
    document isn't in `eval`, rather than silently mixing splits.

    dev=True: plumbing-only mode for runs against train/trial documents (e.g.
    the tinye2e smoke configs). Gold is restricted to exactly the documents
    present in this run (copied from their real train/trial/eval tsv), so the
    printed numbers are about this run's own documents, not swamped by ~230
    other documents this run made no predictions for. NOT a comparable number --
    never report a dev=True result as a benchmark figure.
    """
    df = load_extraction_df(experiment_id)
    doc_ids = sorted(df["document_id"].unique())
    splits = {doc_id: doc_split(doc_id) for doc_id in doc_ids}

    if dev:
        gold_source_dirs = {doc_id: RAW_DATA_DIR / split / "tsv" for doc_id, split in splits.items()}
    else:
        non_eval = {doc_id: split for doc_id, split in splits.items() if split != "eval"}
        if non_eval:
            raise ValueError(
                f"evaluate(dev=False) requires every document to be in the official "
                f"`eval` split for a leaderboard-comparable number; found non-eval "
                f"documents: {non_eval}. Pass dev=True for a plumbing-only check."
            )
        gold_source_dirs = {doc_id: RAW_DATA_DIR / "eval" / "tsv" for doc_id in doc_ids}

    out_dir = Path(out_dir) if out_dir else REPO_ROOT / "analysis" / "out" / "measeval" / experiment_id
    submission_dir = out_dir / "submission"
    gold_dir = out_dir / "gold"
    for d in (submission_dir, gold_dir):
        if d.exists():
            for f in d.glob("*.tsv"):
                f.unlink()
        d.mkdir(parents=True, exist_ok=True)

    coverage = build_submission_tsv(df, submission_dir)

    if dev:
        # Copy only this run's own documents' real gold tsv -- not -l/--limit
        # (which the eval script applies to ALL gold files present anywhere),
        # so this stays legible as "the doc(s) we ran," not a leaderboard claim.
        for doc_id, split_dir in gold_source_dirs.items():
            src = split_dir / f"{doc_id}.tsv"
            if src.exists():
                (gold_dir / f"{doc_id}.tsv").write_text(src.read_text())
    else:
        for src in (RAW_DATA_DIR / "eval" / "tsv").glob("*.tsv"):
            (gold_dir / src.name).write_text(src.read_text())

    # Required so the script doesn't crash -- see "Calibration placeholder
    # rows" above. Written identically to both sides so it's a perfect,
    # self-contained match; its known contribution is subtracted back out of
    # Quantity/Unit below before these numbers are reported.
    write_calibration_doc(submission_dir)
    write_calibration_doc(gold_dir)

    stdout = run_official_eval(submission_dir, gold_dir, mode="class")
    raw_scores = parse_class_scores(stdout)
    scores = _correct_for_calibration(raw_scores)

    return {
        "experiment_id": experiment_id,
        "dev": dev,
        "documents": doc_ids,
        "coverage": coverage.as_dict(),
        "scores": scores,
        "raw_scores_including_calibration_doc": raw_scores,
        "raw_stdout": stdout,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--dev", action="store_true",
                         help="Plumbing-only mode for non-eval-split runs. Not a comparable score.")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    result = evaluate(args.experiment_id, dev=args.dev, out_dir=args.out_dir)

    print(f"experiment_id: {result['experiment_id']}")
    print(f"dev mode: {result['dev']}" + (" (NOT a comparable score)" if result["dev"] else ""))
    print(f"documents: {result['documents']}")
    print(f"coverage: {result['coverage']}")
    print("(Quantity/Unit below have the calibration placeholder doc's 1 known "
          "true positive subtracted out -- see module docstring.)")
    print()
    for annot_type, metrics in result["scores"].items():
        print(f"-- {annot_type} --")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
        print()


if __name__ == "__main__":
    main()
