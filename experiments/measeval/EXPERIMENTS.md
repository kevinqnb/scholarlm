# MeasEval Dataset — Experiment Steps

All commands are run from the **repository root**. This mirrors `experiments/pond/`'s
job-script framework, with the steps that don't apply to measeval removed (see
"How this differs from pond" below).

## Dataset configuration

`experiments/configs/measeval.py` is the single source of truth for all measeval-specific
extraction settings. Unlike pond/nfix/supermat, measeval has no fixed attribute catalog --
see that file's module docstring and `data/measeval/README.md` for the "measurement" bucket
and the quantity-first design (entities are the reported *quantities*; subject/property are
resolved per quantity as measurement-event fields). Matching against ground truth (recovery/validity) is handled by
`analysis/ablation.py`'s `get_matching_rules('measeval')`, not `analysis/calibration.py`.

## How this differs from pond

- **No OCR / PDF pre-processing / table-cleaning steps.** measeval ships plain text with
  gold character-offset annotations directly (`data/measeval/ocr_output_raw/`,
  pre-built by `data/measeval/preprocessing.py`). Every job below reads that directory
  straight -- there is no `processed_pdfs/` and no `ocr_output_cleaned_*` variant to build.
- **No judge steps.** The ground truth was built from MeasEval's own gold annotations,
  thorough enough to score validity by ground-truth matching alone
  (`analysis/metrics.py::validity_rate` with no `judged_df`). `run_judge_*` scripts are
  never run for this dataset -- there is intentionally no `judge/` or `judge_interp/`
  subdirectory here.
- **Ablation 2 is disabled.** `experiments/configs/measeval.py` sets
  `ablation2_entity_schema=None` -- with a single "measurement" bucket there's no
  interesting closed-attribute choice to ablate. `run_ablation.py --ablation 2` will
  raise for this dataset; there is no `measeval_ablation_2_*.sh` script.
- **NuExtract baseline is not runnable.** It's vision-only (reads rendered PDF page
  images via `processed_pdfs/`), and measeval has no PDFs to render. There is no
  `measeval_extract_nuextract*.sh` script for this reason -- not an oversight.
- **Every measeval command needs `--ocr-dir data/measeval/ocr_output_raw`** (already baked
  into every script below). Without it, `run_extraction.py`/`run_ablation.py` fall into
  their PDF table-cleaning path by default and raise looking for a nonexistent
  `data/measeval/processed_pdfs/`.

## Step 1 — Serve the model(s)

The full-suite arms (extraction, ablations 1/3/4/5/6, chatextract) all run on
**gemma-3-27b** and share one server. Extraction is additionally run standalone on
**llama-3.1-8b** and **gpt-oss-120b** for a same-pipeline cross-model comparison (no
ablations/baselines for those two yet -- add scripts the same way if that's wanted
later). Each model is its own server job on its own GPU; nothing here shares a server
across models.

```bash
qsub experiments/serve_gemma_3_27b.sh      # -> job ID A
qsub experiments/serve_llama_3_1_8b.sh     # -> job ID B
qsub experiments/serve_gpt_oss_120b.sh     # -> job ID C
```

Note the three job IDs SGE prints (e.g. `Your job 1234567 (...) has been submitted`) --
the scripts below need the one matching their model.

## Step 2 — Point every job script at its server job

Every script that talks to a vLLM server (everything except `gliner`, which uses its own
GPU directly) has a placeholder `#$ -hold_jid <PLACEHOLDER>`, one token per model so a
single substitution can't accidentally cross-wire a script to the wrong server:

| Model | Placeholder | Files |
|---|---|---|
| gemma-3-27b | `HOLD_JID_PLACEHOLDER` | `extraction/measeval_extract_gemma_3_27b.sh`, `ablation/gemma/measeval_ablation_{1,3,4,5,6}_gemma_3_27b.sh`, `baseline/measeval_chatextract_gemma_3_27b.sh` |
| llama-3.1-8b | `LLAMA_HOLD_JID_PLACEHOLDER` | `extraction/measeval_extract_llama_3_1_8b.sh` |
| gpt-oss-120b | `GPTOSS_HOLD_JID_PLACEHOLDER` | `extraction/measeval_extract_gpt_oss_120b.sh` |

```bash
cd experiments/measeval
grep -rl HOLD_JID_PLACEHOLDER .       | xargs sed -i "s/HOLD_JID_PLACEHOLDER/<JOB_ID_A>/"
grep -rl LLAMA_HOLD_JID_PLACEHOLDER . | xargs sed -i "s/LLAMA_HOLD_JID_PLACEHOLDER/<JOB_ID_B>/"
grep -rl GPTOSS_HOLD_JID_PLACEHOLDER . | xargs sed -i "s/GPTOSS_HOLD_JID_PLACEHOLDER/<JOB_ID_C>/"
```

(This mirrors pond's checked-in scripts, which hardcode a real `hold_jid` from whichever
run produced them -- expect to repeat this substitution for each new run.)

## Step 3 — Submit extraction, ablations, and baselines

```bash
cd /projectnb/mcnet/kevin/coastal/scholarlm

qsub experiments/measeval/extraction/measeval_extract_gemma_3_27b.sh
qsub experiments/measeval/extraction/measeval_extract_llama_3_1_8b.sh
qsub experiments/measeval/extraction/measeval_extract_gpt_oss_120b.sh

qsub experiments/measeval/ablation/gemma/measeval_ablation_1_gemma_3_27b.sh
qsub experiments/measeval/ablation/gemma/measeval_ablation_3_gemma_3_27b.sh
qsub experiments/measeval/ablation/gemma/measeval_ablation_4_gemma_3_27b.sh
qsub experiments/measeval/ablation/gemma/measeval_ablation_5_gemma_3_27b.sh
qsub experiments/measeval/ablation/gemma/measeval_ablation_6_gemma_3_27b.sh

qsub experiments/measeval/baseline/measeval_chatextract_gemma_3_27b.sh

# No server dependency -- can be submitted any time, independently of Step 1/2:
qsub experiments/measeval/baseline/measeval_extract_gliner.sh
```

The gemma-3-27b arms all `hold_jid` on the same server job and read `--ocr-dir
data/measeval/ocr_output_raw` directly, so once that server is healthy they run
concurrently against it (vLLM handles the concurrent request load; no need to serialize
them or spin up a second gemma server instance the way some pond scripts do). The
llama-3.1-8b and gpt-oss-120b extraction jobs each wait on their own separate server.

Each job polls its `.vllm_endpoint_*.txt` file and the server's `/health` endpoint before
running (see any script for the exact logic), so submission order relative to the server
finishing its load doesn't matter -- just submit the servers first so the endpoint files
exist to poll for.

Outputs land at:
- `data/experiments/measeval/extraction/gemma-3-27b/YYYY_mm_dd/final.json`
- `data/experiments/measeval/extraction/llama-3.1-8b/YYYY_mm_dd/final.json`
- `data/experiments/measeval/extraction/gpt-oss-120b/YYYY_mm_dd/final.json`
- `data/experiments/measeval/ablations/ablation{1,3,4,5,6}/gemma-3-27b/YYYY_mm_dd/final.json`
- `data/experiments/measeval/extraction/chatextract-gemma-3-27b/YYYY_mm_dd/final.json`
- `data/experiments/measeval/extraction/gliner-large-v1/YYYY_mm_dd/final.json`

## Step 4 — Analysis (recovery / validity vs. ground truth, no judge needed)

Fill in the resolved dates from Step 3 into `analysis/ablation.py`'s
`ablation_configs['measeval']` and `analysis/baselines.py`'s `baseline_configs['measeval']`
(both currently `None` placeholders), then:

```bash
uv run python analysis/ablation.py    # -> results/ablation/ablation_measeval.csv
uv run python analysis/baselines.py   # -> results/baselines/baselines_measeval.csv
```

If you change `get_matching_rules('measeval')` (e.g. tune the `1/6` fuzzy threshold)
after a prior analysis run, delete `data/experiments/measeval/**/match_cache*.pkl` first --
`cached_match` reuses a stale cache keyed only on the extraction file's mtime, not on the
matching rules, so a rule change silently won't take effect otherwise.

## Known caveats (see `data/measeval/README.md` for detail)

- Fuzzy threshold `1/6` is chosen by analogy to nfix's 2-fuzzy-field case; unvalidated
  against measeval itself, revisit once these runs land.
- ~1.1% of ground-truth rows (10/951) have both `name` and `property` null and are
  structurally unrecoverable by this matching scheme -- a hard ceiling on recovery.
- Units are strict-matched but open free text for measeval (no fixed per-attribute
  vocabulary to prompt the model with, unlike pond/nfix/supermat) -- treat recovery/validity
  as a conservative lower bound until unit normalization is revisited.
- `paper_filter=None` extracts over all 448 docs (train+trial+eval combined). Restricting
  to `split == "eval"` for leaderboard comparability, if wanted, is a slice at the analysis
  layer (ground truth carries `split` per-row) -- not done here, left as a call for whoever
  needs that comparison.
