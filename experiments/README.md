# Experiments

All scripts are run from the **repository root**. Per-dataset configs live in
`experiments/dataset-configs/`; per-model configs live in
`experiments/model-configs/{kind}/<model>.yaml`; per-experiment configs live in
`experiments/experiment-configs/{dataset}/{type}/<id>/<id>.yaml`.

## Scripts

| Script | Purpose |
|---|---|
| `run_extraction.py` | Run the 7-step extraction pipeline |
| `run_ablation.py` | Run an ablation variant (1–6) |
| `run_table_cleaning.py` | Pre-clean OCR tables with a vLLM model |
| `run_judge_interp.py` | Interpretability judge (NNsight, collects activations) |
| `run_judge_local.py` | Local judge via vLLM server |
| `run_judge_combine.py` | Majority-vote combination of judge outputs → `combined.json` |
| `run_jacobian_lens.py` | NNsight, Jacobian-lens j-scores |
| `run_representation_lm.py` | NNsight, key-term last-layer representation collection |
| `run_attribution.py` | Attribution tracking |
| `run_baseline_gliner.py` / `run_baseline_nuextract.py` / `run_baseline_chatextract.py` | Baseline extractors |
| `run_ocr.py` | Run OLMo-OCR on PDFs |
| `process_pdfs.py` | Pre-process PDFs (requires separate environment) |
| `run_probe_augment.py` | Probe-dataset augmentation |

Every runner takes one positional `config` argument — the path to its
`experiment-configs/.../<id>.yaml` — and no override flags. Run any of them with
`--help` for the full flag set.

There is no script left that produces human-validation `responses.json` files
(`validation.py`, a Streamlit app, was removed) — `analysis/loaders.py`'s
`load_human_judgements` and `experiments/utils.py`'s `find_human_responses`
still read any that already exist on disk.

## Serving a model

`run_extraction.py`, `run_table_cleaning.py`, and `run_judge_local.py` require a
running vLLM server, by default at `http://localhost:8081/v1`. You don't need to
bring one up by hand: `bash experiments/submit.sh <id>` resolves the experiment's
model-config, starts (or reuses) a vLLM server as part of the submitted SGE job, and
waits for `/health` before calling the runner. Serve parameters live in
`experiments/model-configs/{extraction,vllm_judge}/<model>.yaml`'s `serve:` block —
there is no more standalone `gen_serve_script.py` or hand-written `serve_*.sh`.

## Basic workflow

```bash
# Extract (run directly)
python experiments/run_extraction.py \
    experiments/experiment-configs/pond/extraction/<id>/<id>.yaml

# ...or submit to SGE, which brings up the vLLM server for you
bash experiments/submit.sh <id>

# Judge (run one or more)
python experiments/run_judge_interp.py \
    experiments/experiment-configs/pond/judge_interp/<id>/<id>.yaml
python experiments/run_judge_local.py \
    experiments/experiment-configs/pond/judge_local/<id>/<id>.yaml

# Combine judge outputs (explicit extraction_id/judge_ids in the config,
# not directory-scanning)
python experiments/run_judge_combine.py \
    experiments/experiment-configs/pond/judge_combine/<id>/<id>.yaml
```

## Ablations

```bash
python experiments/run_ablation.py \
    experiments/experiment-configs/pond/ablation/<id>/<id>.yaml
```

Judging an ablation run uses the same judge runners above, pointed at that
ablation's own extraction output via the judge config's `extraction_id`.
