<!-- Public devlog entry: scholarlm/devlog/2026-09-29-nfix-supermat-probe-v3-rebuild-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-09-29-nfix-supermat-probe-v3-rebuild-01
kind: build
config: experiments/experiment-configs/nfix/probe_augment/2026-09-29-nfix-probe-augmentation-v3-rung3-01/2026-09-29-nfix-probe-augmentation-v3-rung3-01.yaml
---

## Session 2026-09-29

### Prompts

- Fix up the nfix and supermat probe-dataset generators too: some of the pond changes were pond-specific, but others, like the field structure, should apply here.
- Build supermat from the dataset config's ground truth; change nfix substrate "wc" to "water column" and use the prompt's date format (leave the plural ecosystem types); treat N vs N2 units as ambiguous and exclude them from the wrong-units negatives.
- Keep v3 as is for now (a proposed extra kind of negative for range and tolerance values is deferred).
- Submit the rung-3 jobs, and commit the gpt-oss response cache following the v2 convention.

### Implemented

Rebuilt the `--augment` path of `data/nfix/create_probe_dataset.py` and `data/supermat/create_probe_dataset.py`, reusing the shared helpers from the pond rebuild unchanged. Rows now carry only the fields each dataset's judge sees, papers and contexts come from the table-cleaned OCR (`--ocr-dir` is required in `experiments/run_probe_augment.py`), nfix gets the prompt's substrate wording and date format plus a dimensional unit-equivalence check for the wrong-units negatives, and supermat builds from its qualifier ground truth via a new `--qualifiers` flag with string values kept verbatim in every file. The default v1 path is byte-identical to before. Tests are in `tests/test_nfix_probe_v3.py` and `tests/test_supermat_probe_v3.py` plus updates to two existing files; rungs 1 and 2 passed and the two rung-3 yield checks (`experiments/experiment-configs/{nfix,supermat}/probe_augment/2026-09-29-{nfix,supermat}-probe-augmentation-v3-rung3-01/`) were submitted via `experiments/submit.sh`. Probe numbers on the v2 nfix and supermat files are not comparable to v3.

### Commits

- 495c162 Pond v3 probe augmentation: gpt-oss response cache from rung-3 job 7792591
- 9965700 nfix/supermat probe v3: judge-visible columns, cleaned-OCR context, equivalence-aware negatives
