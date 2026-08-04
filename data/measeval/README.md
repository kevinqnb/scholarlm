# data/measeval — MeasEval (SemEval-2021 Task 8)

Ground truth for the measeval dataset, adapted from
[MeasEval](https://github.com/harperco/MeasEval) (Harper et al., SemEval-2021 Task 8):
a CC-BY-licensed corpus of short scientific-text excerpts (one paragraph each, drawn
from Elsevier OA-STM articles) annotated with quantities, the entities/properties
they measure, and qualifying context.

Unlike pond/nfix/supermat, MeasEval ships **plain text and gold TSV annotations
directly** — there is no PDF/OCR step, and no need for the fuzzy page-attribution
machinery those datasets use, since MeasEval's own annotations already carry exact
character offsets into the source text.

## Directory structure

```
data/measeval/
  download_measeval.py   - shallow-clones harperco/MeasEval into raw/ (not committed)
  raw/                    - vendored clone of the source repo (not shared, regenerable)
  directory.json          - document registry: source_split, article_id (title/author/year unavailable)
  ocr_output_raw/         - {document_id}.txt, each wrapped as a single <page number="0"> block (not shared, regenerable)
  preprocessing.py        - builds ground_truth.json and ground_truth_ten.json
  ground_truth.json       - ground truth dataset
  ground_truth_ten.json   - top-10 document (by row count) development subset
```

`raw/` and `ocr_output_raw/` are gitignored (regenerable from the public source via
`download_measeval.py` + `preprocessing.py`), same as `pdfs/`/`ocr_output_raw/` in
the other datasets.

## Build

```bash
python data/measeval/download_measeval.py
python data/measeval/preprocessing.py
```

## Document granularity and splits

Each MeasEval `docId` (a single paragraph excerpt, e.g. `S0006322312001096-1136`)
is treated as its own `document_id`, matching MeasEval's own unit of annotation.
`train` (248 text files / 233 annotated), `trial` (65/65 — note trial's text lives
under `raw/data/trial/txt/`, not `text/`), and `eval` (135/130) are combined into a
single ground truth. `data/iaa/` (992 files, a separate inter-annotator-agreement
re-annotation of a subset of `train`) is **excluded** — including it would
double-count rows for documents already present in `train`. 15 documents in `train`
and 5 in `eval` have no `.tsv` at all; these are real documents with zero annotated
measurements (still given a `directory.json` entry and an `ocr_output_raw/*.txt`,
just no ground-truth rows).

### Train/trial/eval and comparability with published results

Per MeasEval's own README (`raw/README.md`, `raw/eval/README.md`), **`eval` is the
official held-out gold test set** used for the SemEval-2021 Task 8 / CodaLab
leaderboard — it's the split to filter to (`split == "eval"`) for any comparison
against published MeasEval results. `train`/`trial`/`eval` are verified disjoint
(`preprocessing.py` raises if it ever finds a `document_id` in more than one split).
Every `ground_truth.json` row carries a `split` column directly (not just
`directory.json`) specifically so eval-only filtering doesn't require a join and
can't be silently skipped. `ground_truth_ten.json`, the quick-iteration dev subset,
is built from `train`/`trial` only and never draws from `eval`, so prompt/config
development can't accidentally peek at held-out test answers.

**Caveat — this only guarantees you're using the right *documents*, not a
numerically comparable *score*.** The official leaderboard scores submissions with
`raw/eval/measeval-eval.py`: Exact Match and SQuAD-style token-overlap F1 across 9
components (Quantity, MeasuredProperty, MeasuredEntity, Qualifier, Unit, Modifiers,
HasQuantity, HasProperty, Qualifies), matched via BRAT-style span alignment. This
repo's own `analysis/metrics.py` (recovery rate / hallucination rate via
`match_datasets`' strict/fuzzy row matching) is a different methodology built for
pond/nfix/supermat's closed-attribute schema. Evaluating on `split == "eval"` with
this repo's own metrics gives numbers that are comparable *across models run
through this repo*, but **not** directly comparable to published MeasEval
leaderboard/paper numbers unless the official scorer is run separately against
`raw/eval/`. That's a decision for whoever builds the extraction config and
evaluation step, not something resolved by this ground truth alone.

## Ground truth schema

One row per MeasEval `annotSet`. Verified across all 1663 annotSets in
train+trial+eval that every annotSet contains at most one `Quantity`, one
`MeasuredEntity`, one `MeasuredProperty`, and 0–3 `Qualifier`s — so grouping by
`(docId, annotSet)` is sufficient; no relation-graph traversal across annotSets is
needed.

| field | source |
|---|---|
| `document_id` | tsv/text filename stem |
| `split` | `"train"`, `"trial"`, or `"eval"` — `"eval"` is the official held-out test set, see above |
| `quantity` | `Quantity.text` — the raw quantity span, units included as written (`"54.8 years"`, `"5318"`). Gold counterpart of the extraction side's entity field under the quantity-first design (see below); carried for traceability, not used by matching, which keys on the parsed `value`/`units` |
| `name` | `MeasuredEntity.text`, or `None` if the Quantity attaches to no entity |
| `attribute` | constant `"measurement"` for every row — matches the single abstract attribute bucket in `experiments/configs/measeval.py`'s `attribute_info_dict`, so ground truth and extraction output strict-match trivially on this field (see below) |
| `property` | `MeasuredProperty.text`, or `None` if the Quantity attaches directly to an entity with no property span — the actual open-vocabulary property name, intended for fuzzy rather than exact matching |
| `value` | numeric value parsed from `Quantity.text` |
| `units` | `Quantity.other["unit"]`, or `None` |
| `additional_details` | all linked `Qualifier.text` spans joined with `"; "`, or `None` |
| `mods` | `Quantity.other["mods"]`, kept verbatim for reference/filtering |
| `annot_set` | source MeasEval `annotSet` id, for traceability back to the raw tsv |
| `entity_start`/`entity_end`, `property_start`/`property_end`, `quantity_start`/`quantity_end` | character offsets of each span into `ocr_output_raw/{document_id}.txt`'s inner text — exact gold provenance, replacing the `page_number`/`page_score`/`page_confidence` fuzzy-attribution columns used by the other three datasets |

### Attribute is free text, not a closed catalog

pond (7 attributes) and supermat (1 attribute) both use a small, closed
`attribute_info_dict`. MeasEval's `MeasuredProperty` spans are open vocabulary
(`"mean (pressure averaged) temperature"`, `"paleolatitude"`, `"grew"`, ...), so they
can't be mapped onto a fixed set the way pond/supermat's attributes are.

`experiments/configs/measeval.py` resolves this by collapsing extraction to a single
abstract `attribute_info_dict` bucket, `"measurement"` — used only as a coarse
per-document gate — while the real property name is resolved per measurement, as
`MeasurementEventSchema.property`.

### Quantity-first enumeration

The unit of enumeration on the extraction side is the **quantity**, matching this
ground truth's own unit of annotation: `EntitySchema` has the single field
`quantity` (one item per directly reported number, span copied verbatim), and what
that number was measured on (`name`), of what (`property`), and under what
circumstances (`additional_details`) are resolved afterwards as measurement-event
fields. Value extraction then only splits the known span into `value` + `units`.

This inverted an earlier design in which entities were (subject, property) pairs and
the quantity was whatever the value step found for them. It was changed because on
the ten-document dev subset that design recovered 0.370 against a *value-only*
ceiling of 0.446 — 55% of gold rows were lost because their number never appeared in
the output at all, before subject/property/unit matching entered into it (units cost
a further ~5 points, subject/property ~1). Enumerating subjects first asks the model
to solve the hard half first, and every pair it fails to name takes that pair's
numbers down with it. See the config's module docstring for the full rationale and
for the two earlier designs this replaces.

The ground truth schema mirrors the current design field-for-field: `quantity` is
the raw `Quantity.text` span, `attribute` the constant `"measurement"`, `name` the
`MeasuredEntity` span, and `property` the `MeasuredProperty` span (or `None`).

`analysis/ablation.py`'s `get_matching_rules(dataset)` (imported by `analysis/baselines.py`
as well) has a `measeval` branch: `attribute` joins in the strict-match set as-is
(it's constant on both sides, a no-op), and `property` goes in the *fuzzy* set
alongside `name`, since it's open text and ~36% of rows have it as `None` on the
ground-truth side. Fuzzy threshold is `1/6`, matching nfix's 2-fuzzy-field case.
These rules were deliberately left untouched by the quantity-first change (both
fields are plain columns of the final record whether they come from the entity or
the event), so runs from before and after it are directly comparable. Adding
`quantity` to the fuzzy set is a separate, still-open decision.

Note: `analysis/calibration.py` is a different, judge/probe-specific file (it eagerly
loads trained probes and combined judge outputs at import time) — it is **not** the
place for a measeval branch, since this dataset has no judge pipeline (see "No judge
pipeline" below). Don't be misled by the module docstring in
`experiments/configs/measeval.py`, which still names it as the "known follow-up";
`get_matching_rules` in `analysis/ablation.py` is the actual, working integration point.

Two matching caveats worth knowing before reading the first results table:
- **~1.1% of ground-truth rows (10/951) have both `name` and `property` null.**
  `match_datasets` drops a candidate edge outright when every fuzzy field is null on
  either side (see `src/scholarlm/utils/data.py`), so these rows are structurally
  unrecoverable regardless of fuzzy threshold — a hard ceiling on recovery, not a
  bug. (Also: unlike supermat, a null `name` here does *not* fall back to
  strict-only matching — `property` still has to carry the fuzzy score, or the
  edge is dropped.)
- **Units are strict-matched, but open free text.** `attribute_info_dict["measurement"]["units"]`
  is `[]` (no fixed vocabulary to prompt the model with, unlike pond/nfix/supermat),
  so a model's unit spelling has no vocabulary to converge toward. Treat measeval
  validity/recovery as a conservative lower bound until unit normalization is
  revisited (see `analysis/baselines.py`'s `normalize_baseline_extraction`, written
  for exactly this "not shown the ground truth's unit vocabulary" situation).

### Ground truth value policy

Matches the policy documented in `data/supermat/README.md`: ground truth `value`
must be a single, unambiguous reported number. Quantity rows are **dropped** when
their `mods` field contains `IsRange`, `IsApproximate`, or `IsList` (matched by
substring, since the corpus also contains garbled concatenations like
`IsRangeHasTolerance` or `IsMeanIsRange`). `IsCount`, `IsMean`, `IsMedian`, and
`HasTolerance` are **kept** — each is still a single definite number as written.
Rows with no digit at all in the Quantity span (spelled-out numbers like `"four"`,
`"twice"`) are also dropped rather than parsed via word-to-number inference.

Value extraction tries `<mantissa> × 10<exp>` scientific notation first (e.g.
`"3.7 × 106"` → `3.7e6`, `"4.3 × 10−8"` → `4.3e-8`), then falls back to the first
plain numeric token in the span (mirroring supermat's `_parse_tcvalue`). A handful
of Quantity spans describe a *product* of two numbers rather than a single value
(e.g. `"121 × 53"`, `"60 × 10 × 3 mm3"`, physical dimensions) — these are not
scientific notation, so only the first factor is captured as `value`. This is a
known, accepted limitation rather than something worth special-casing for a
handful of rows; the neither-entity-nor-property-linked edge case (5 annotSets
across the whole corpus) and dimension-product rows are the main places the
schema doesn't perfectly fit MeasEval's raw annotation.

Running `preprocessing.py` prints the drop counts for both categories on every run.

## Running experiments

`experiments/configs/measeval.py` exists, and `analysis/ablation.py`'s `get_matching_rules`
has a `measeval` branch (see "Attribute is free text, not a closed catalog" above), so the
same core workflow used for pond/nfix/supermat runs here too:

```bash
# Main pipeline. measeval is plain text, not PDF -- always pass --ocr-dir, or
# run_extraction.py falls back to its PDF table-cleaning path and raises
# looking for a nonexistent data/measeval/processed_pdfs/.
python experiments/run_extraction.py --dataset measeval --model gemma-3-27b \
    --ocr-dir data/measeval/ocr_output_raw

# Ablations 1, 3-6 (same --ocr-dir requirement). Ablation 2 is intentionally
# disabled for measeval -- see experiments/configs/measeval.py's comment next
# to ablation2_entity_schema=None -- there is no interesting closed-attribute
# choice to ablate with a single "measurement" bucket.
python experiments/run_ablation.py --dataset measeval --model gemma-3-27b --ablation 1 \
    --ocr-dir data/measeval/ocr_output_raw

# Text-based external baselines. Neither goes through the PDF table-cleaning
# path, so --ocr-dir is not required for either -- but it's worth knowing they
# differ: chatextract accepts an optional --ocr-dir (defaults to
# ocr_output_raw already, same as above); gliner has no --ocr-dir flag at all
# and always reads ocr_output_raw directly.
python experiments/run_baseline_chatextract.py --dataset measeval --model gemma-3-27b
python experiments/run_baseline_gliner.py --dataset measeval
```

`run_baseline_nuextract.py` does **not** work here: NuExtract-2.0-8B is vision-only and
reads rendered PDF page images from `processed_pdfs/`, which requires
`experiments/process_pdfs.py` — and measeval has no PDFs to render in the first place
(see "Directory structure" above). There is no input this baseline could consume for
this dataset.

No judge pipeline is used or needed here (unlike pond/nfix/supermat): the ground truth
was built directly from MeasEval's own gold character-offset annotations, thorough enough
to score validity by ground-truth matching alone (`analysis/metrics.py`'s `validity_rate`
already supports this — pass no `judged_df` and it falls back to match-only labels).
`run_judge_*` scripts are simply never run for this dataset.

Analysis (recovery/validity vs. ground truth, no judge needed):

```bash
# Fill in real dates in analysis/ablation.py's ablation_configs['measeval'] and
# analysis/baselines.py's baseline_configs['measeval'] as runs are produced, then:
uv run python analysis/ablation.py
uv run python analysis/baselines.py
```
