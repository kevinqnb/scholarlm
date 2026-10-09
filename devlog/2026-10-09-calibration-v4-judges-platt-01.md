---
id: 2026-10-09-calibration-v4-judges-platt-01
kind: build
---

## Session 2026-10-09

### Prompts
- "I want to produce calibration results for the llama8b and mistral12b interp judges. This should be identical to 2026-10-08-calibration-v4-gemma27b-qwen-2.5-7b-intercept-fit-sample-01, but using those other judges. In addition to the qwen judgement already implemented, that's 3 judge interp models over the gemma-3-27b extractions. In addition to that, I want two more sets of 3 configs (for all three interp models), for the full pipeline llama-3.1-8b and gpt-oss-120b extraction models. Can you help me produce configs to do that?"
- "train the 6 llama/mistral probes" -- chose to train them as Platt, via a config key.
- "Why would we have `use_platt_scaling: false` if we're using `probe_variant: platt` (as we should). What does the first flag control?"
- "Maybe we could rerun some tests with qwen to see which is the right choice? Can you make two new configs for qwen so I can compare side by side?" / "The issue is that I have recently updated to using calibration v4."
- "Submit full runs now"
- "Do the flip and commit. Then submit the 8 full runs."
- "Can you help me run 2026-10-08-calibration-v4-gptoss120b-qwen-2.5-7b-intercept-fit-sample-01 again but time with platt fitting? I.e. make a new config for that and run it -- just this setting."
- "Can you also submit one for an oracle prior shift?"

### Implemented
Added eight calibration v4 configs, `analysis/analysis-configs/2026-10-08-calibration-v4-{gemma27b,gptoss120b,llama8b-reppen}-{qwen-2.5-7b,llama-3.1-8b,mistral-nemo-12b}-intercept-fit-sample-01.yaml`, with params identical to the gemma27b x qwen reference. `analysis/synthetic_probe_train.py`'s hardcoded Platt switch is now a required `params.use_platt_scaling` key in every synthetic-probe config (`analysis/analysis-configs/*-synthetic-probe-*.yaml`). Retraining a qwen probe with it set reproduced the existing Platt pickles exactly, and the llama/mistral v3 probes were then trained with it. A platt/noplatt pair (`...-gemma27b-qwen-2.5-7b-intercept-fit-sample-{platt,noplatt}-01`) compared the two probe variants under v4 and served as a seed-determinism control; Platt was kept. `analysis/calibration_updated_v4.py` gains a `platt_fit` recalibration (`analysis/recalibration.py`'s `platt_fit_map`, which rejects separable or non-converged fits) in a standalone eval commit. A smoke run reproduced the existing intercept-fit results exactly, so no prior numbers change. That commit was followed by gptoss120b x qwen `platt-fit-sample-01` and `prior-shift-oracle-01` configs and their smoke configs.

### Commits
- `1aed5fe` synthetic_probe_train: train-side Platt scaling is a required config key
- `b84a1b8` calibration v4 configs: 3 extractions x 3 interp judges, plus platt/noplatt pair
- `f100c75` calibration v4: add platt_fit recalibration (slope + intercept by logistic MLE)
- `ec8dbdc` calibration v4 configs: gptoss120b x qwen platt_fit full run + platt/intercept smokes
- `23262c9` calibration v4 config: gptoss120b x qwen prior-shift oracle (diagnostic)
