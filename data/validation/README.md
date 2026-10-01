# Validation sets for the manual-validation site

The `validation_set_*.json` files in `data/{pond,nfix,supermat}/` are the
inputs to the human-validation site
([scholarlm-validation](../../../scholarlm-validation)). They are **not tracked
in git** (several MB each; see `.gitignore`) and are rebuilt from the scripts
here. Every set is a deterministic function of pinned inputs, so rebuilding
reproduces the deployed files byte-for-byte.

## What is on the site

| Site dataset | File | Built by |
|---|---|---|
| `pond__pilot-gemma27b-full`, `nfix__pilot-gemma27b-full`, `supermat__pilot-gemma27b-full` | `data/<ds>/validation_set_pilot-gemma27b-full.json` | `build_pilot_validation_sets.py` |
| `pond__pilot-langextract-gemma27b` | `data/pond/validation_set_pilot-langextract-gemma27b.json` | `build_pilot_validation_sets.py` |
| `pond__main-gemma27b-full`, `supermat__main-gemma27b-full` | `data/<ds>/validation_set_main-gemma27b-full.json` | `build_main_validation_sets.py` |

How many judges each dataset needs is configured on the site (`judges_required`),
not in these files. The main set is disjoint from the pilot set of the same dataset, so
pilot + main together form a ~500-measurement sample.

## Inputs (untracked experiment outputs)

The builders read, relative to the repo root:

- `data/<ds>/probe_dataset_test.json` and `probe_dataset.json` — the test/train
  splits (tracked). Sampling is restricted to the test split.
- The pinned extraction output, `experiments/results/<ds>/.../final.json`,
  one per set. The exact paths are in `_configs()` / `_configs_other()` in
  `build_pilot_validation_sets.py`; the main set reuses the pilot's.
- The **table-cleaned OCR** the extraction actually ran against
  (`experiments/results/<ds>/drop_references/2026-09-18-<ds>-table-cleaning-drop-references-01`).
  Always use the OCR dir the extraction consumed, never the raw OCR: validators
  must read the text the method saw.

`experiments/results/` is not in git, so these are produced by running the
extraction experiments (or obtained from the authors).

## Rebuilding

Run from the repo root. Order matters: the main sets read the pilot sets.

```bash
python data/validation/build_pilot_validation_sets.py              # pond, nfix, supermat
python data/validation/build_main_validation_sets.py pond supermat
```

Both accept an output root so you can build into a scratch directory and
compare before touching `data/`:

```bash
python data/validation/build_pilot_validation_sets.py --out-dir /tmp/vs
python data/validation/build_main_validation_sets.py pond supermat --data-dir /tmp/vs
cmp /tmp/vs/pond/validation_set_pilot-gemma27b-full.json data/pond/validation_set_pilot-gemma27b-full.json
```

**Do not change the seed, the sampling constants, or the pinned inputs for a
set that is already live.** The site stores judgements keyed on
`(dataset, measurement_id)`; a different draw would attach existing judgements
to the wrong measurements. A changed set must go in as a new dataset name.

### How the pilot sample is drawn

1. **Papers.** `N_PAPERS = 20` papers are drawn from the test split with
   `Random(SEED)` (`SEED = 342`).
2. **Measurements.** For each method independently, up to `N_PER_PAPER = 5`
   rows per paper are drawn from that method's own `final.json`.
3. **Top-up.** Papers can have fewer than 5 rows, so if a method ends up below
   `TARGET_MEASUREMENTS = 100`, further test-split papers (those with at least
   one row in that method's output) are drawn in a fresh seeded shuffle until
   the target is met. The last paper may overshoot (hence 102 / 104). The
   phase-1 rows are never altered. Top-up papers are listed under
   `topup_document_ids` in the output file.

Rows with no real page provenance (missing or `None` `page_number`) get the
sentinel page `-1`, and the site shows the whole paper with no page highlighted.

### How the main sample is drawn

Papers are visited in sorted order over the whole test split; per-paper quotas
differ for papers that are in the pilot vs. not (`PER_PAPER`), then single
extra rows are drawn at random from papers with rows left until pilot + main
reaches `TARGET_TOTAL = 500`. No pilot `measurement_id` is ever drawn.

## Loading into the site

Each file carries its own `dataset` name and a full-paper `full_text` per
document. Ingest with the validation repo (see its README). Put each dataset's
tracked `directory.json` next to its set files, so paper titles and authors are
loaded too.

```bash
uv run python -m scholarlm_validation.ingest data/pond/validation_set_pilot-gemma27b-full.json ... --db "$DB"
```

Use `--append` to add new measurements to a DB that already holds judgements;
ingesting a dataset name that already exists is refused rather than
overwritten.

## Analysis

`scholarlm-validation/analysis/pipeline.py` finds a site dataset's extraction
run through `extraction_final_json`, recorded inside each set file, so keep
the files in `data/<ds>/` under their `validation_set_<method_key>.json` names.
