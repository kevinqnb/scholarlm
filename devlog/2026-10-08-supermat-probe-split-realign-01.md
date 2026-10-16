---
id: 2026-10-08-supermat-probe-split-realign-01
kind: build
---

## Session 2026-10-08

### Prompts
- "Please help me make a surgical update to the supermat dataset. In `../scholarlm-validation` we built a validation set for the supermat data, which was intended to be from the test split of documents. The problem is, that after we built this the train test splits shifted. Specifically, I think the difference arose between v2 and v3 of the supermat probe datasets. The goal now is to perform a surgical procedure where we re-align the v3 split so that its test set agrees with the test set for `../scholarlm-validation`. ... If a data point moves from the train set into the test set, we'll need to make sure that we move its raw values to the primary set, and its full synthetically supplemented values to the diag set. If a point moves from the test set to the training set, ONLY its diag entry should move into the training set. The scripts for building the probe dataset should also be adjusted to make the new split reproducible. ... The judge-interp outputs on supermat v3 ... need to be moved around. First investigate: is this error clear? ... Is there anything else we would need to change to make this consistent? ... just highlight to me which ones need re-running."
- "(1) I think the 6 new papers should go into the training set. Otherwise wouldn't the building of the validation sets in this repo and the bucket/pool sets in `scholarlm-validation` be affected? (2) Can you explain this more to me? What exists in the train file and how would it move to the primary negatives? (3) Good suggestion. (4) Good. (5) Don't worry about prevalence validation -- that is being dropped."
- "Go with (A), start with the fixture test" (primary rows for papers moving into test reuse their already-judged GT valids and GT-parent negatives from the train file)
- "How come v3 had 6 extra papers?"

### Implemented
The supermat v3 probe was built from `ground_truth_qualifiers.json`. That file has 6 more papers than `ground_truth.json` (papers whose only Tc values are ranges, approximations or bounds), so the seeded paper shuffle gave a different train/test split from v1/v2, the split the manual validation sets were drawn from. New `data/supermat/realign_probe_split.py` rearranges the existing v3 rows onto the v1 split (committed as `data/supermat/probe_split_v3s.json`; the 6 new papers go to train). It then trims each file to 1:1 with a seed and logs each row's source, writing `probe_dataset*_v3s*.json`. It also assembles judge-interp outputs for the realigned files from the v3 runs, checking each row against its source. Configs: `experiments/experiment-configs/supermat/judge_interp/2026-10-08-supermat-v3s-{qwen-2.5-7b,llama-3.1-8b,mistral-nemo-12b}-synthetic-judge-{train,test-primary,test-diag}-01.yaml`. `create_probe_dataset.py` gains `--split-file`, which pins the split for future builds. Supermat synthetic-probe and calibration results computed from the v3 runs need regenerating against the v3s runs.

### Commits
8d6a192 supermat: realign v3 probe split to the v1/v2 (validation-set) split as v3s
