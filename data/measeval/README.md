# data/measeval — MeasEval (SemEval-2021 Task 8)

Adapted from [MeasEval](https://github.com/harperco/MeasEval) (Harper et al., SemEval-2021 Task 8):
a CC-BY corpus of short scientific-text excerpts (one paragraph each, from Elsevier OA-STM
articles) annotated with quantities, the entities and properties they measure, and qualifying
context.

Unlike pond/nfix/supermat there is no PDF or OCR step: MeasEval ships plain text and gold TSV
annotations with exact character offsets, so no page attribution is needed.

All commands below run from the repo root, in the `uv sync` environment.

## Tracked files

| File | What it is / who uses it |
|---|---|
| `ground_truth.json` | **The ground truth used by the pipeline** (`DatasetConfig.ground_truth_file`): one row per MeasEval `annotSet`, ranges/approximations/lists dropped (see below). 951 rows, all three splits. |
| `ground_truth_qualifiers.json` | Same annotSets, undropped (1663 rows): the raw Quantity span text as `value`, null qualifier fields. Built by `preprocessing.py --qualifiers`. |
| `ground_truth_ten.json` | Ten-document development subset (92 rows), from `train` and `trial` only, never `eval`. |
| `directory.json` | Document registry: `source_split`, `article_id` (title/author/year unavailable). `DatasetConfig.metadata_file`. |
| `download_measeval.py`, `preprocessing.py` | Builders, see below. |

Regenerable and gitignored: `raw/` (clone of the MeasEval repo, including the official scorer and gold TSVs) and `ocr_output_raw/` (`{document_id}.txt`, each wrapped in a single `<page number="0">` block).

## Reproducing the files

```bash
python data/measeval/download_measeval.py            # shallow-clones MeasEval into raw/ (--force to re-clone)
python data/measeval/preprocessing.py                # ground_truth.json, ground_truth_ten.json, directory.json, ocr_output_raw/
python data/measeval/preprocessing.py --qualifiers   # ground_truth_qualifiers.json only
```

## Documents and splits

Each MeasEval `docId` (one paragraph excerpt, e.g. `S0006322312001096-1136`) is its own
`document_id`. `train`, `trial` (its text lives under `raw/data/trial/txt/`) and `eval` are
combined; they are verified disjoint (`preprocessing.py` raises otherwise). `data/iaa/`, a
separate inter-annotator re-annotation of part of `train`, is excluded because it would
double-count rows. Documents with no `.tsv` have a `directory.json` entry and an
`ocr_output_raw` file but no ground-truth rows.

`eval` is the official held-out gold test set (SemEval-2021 Task 8 / CodaLab leaderboard). Every
ground-truth row carries a `split` column, so eval-only filtering needs no join.

## Ground truth schema

One row per `annotSet`. Every annotSet has at most one `Quantity`, one `MeasuredEntity`, one
`MeasuredProperty` and 0–3 `Qualifier`s, so grouping by `(docId, annotSet)` is sufficient.

| field | source |
|---|---|
| `document_id`, `split` | text/tsv filename stem; `"train"`, `"trial"` or `"eval"` |
| `quantity` | `Quantity.text`, the raw span with units as written (`"54.8 years"`). Carried for traceability; matching keys on `value`/`units` |
| `name` | `MeasuredEntity.text`, or `None` |
| `attribute` | constant `"measurement"`, the single abstract bucket in `experiments/dataset-configs/measeval.py` |
| `property` | `MeasuredProperty.text`, or `None`. Open vocabulary, intended for fuzzy matching |
| `value` | number parsed from `Quantity.text` |
| `units` | `Quantity.other["unit"]`, or `None` |
| `additional_details` | linked `Qualifier.text` spans joined with `"; "`, or `None` |
| `mods` | `Quantity.other["mods"]`, verbatim |
| `annot_set` | source `annotSet` id |
| `entity_*`, `property_*`, `quantity_*` (`_start`/`_end`) | character offsets into `ocr_output_raw/{document_id}.txt`'s inner text (exact gold provenance) |

`attribute` is not a closed catalog as in pond/supermat: units and properties are open free
text, so there is no unit vocabulary to prompt with.

## Ground truth value policy

`value` is a single, unambiguous reported number, as in `data/supermat/README.md`. Rows whose
`mods` contain `IsRange`, `IsApproximate` or `IsList` (matched by substring, since the corpus has
garbled concatenations like `IsRangeHasTolerance`) are dropped, as are rows with no digit in the
Quantity span (`"four"`, `"twice"`). `IsCount`, `IsMean`, `IsMedian` and `HasTolerance` are kept.
Scientific notation (`"3.7 × 106"` → `3.7e6`) is parsed first, then the first plain numeric token.
Product spans (`"121 × 53"`) keep only the first factor. `preprocessing.py` prints the drop counts
on every run.

## Running experiments

measeval is plain text, so every config pins `ocr_dir: data/measeval/ocr_output_raw`; the PDF
table-cleaning path does not apply. `run_baseline_nuextract.py` does not work here (it reads
rendered PDF page images). No judge pipeline is used: the gold annotations are scored by
ground-truth matching alone.

```bash
python experiments/run_extraction.py experiments/experiment-configs/measeval/extraction/2026-09-21-measeval-extraction-gemma27b-full-01/2026-09-21-measeval-extraction-gemma27b-full-01.yaml
bash experiments/submit.sh 2026-09-21-measeval-extraction-gemma27b-full-01 [--dry-run]    # or via SGE
```

Other runners (ablations, GLiNER, LangExtract, NuExtract3, ChatExtract) follow the same
`<runner> <config>` pattern; configs are under `experiments/experiment-configs/measeval/`.

## Evaluation on the official MeasEval scorer

`analysis/measeval_evaluation.py` exports a run as MeasEval TSVs and scores it with the
unmodified official scorer, `raw/eval/measeval-eval.py`, so the numbers are comparable to
published MeasEval results. Use `dev: false` (the eval split) for leaderboard comparison. It
reports Quantity, Unit and MeasuredEntity; MeasuredProperty, Qualifier, modifiers and relations
are not reported. Spans are recovered verbatim from the OCR text, and records whose spans cannot
be located are dropped and counted in `Coverage`.

```bash
python analysis/measeval_evaluation.py --config analysis/analysis-configs/measeval/2026-10-09-measeval-official-eval-01.yaml
python analysis/measeval_evaluation.py --experiment-id <id> [--dev] [--out-dir DIR]
```

The repo's own recovery/validity metrics (strict/fuzzy row matching against `ground_truth.json`)
are a separate evaluation, used across models run through this repo.
