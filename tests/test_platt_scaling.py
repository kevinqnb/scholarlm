"""Rung-1 unit tests for analysis/platt_scaling.py helpers and load_platt_sweep_config."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "experiments"))

from analysis import analysis_config as ac  # noqa: E402
from analysis import platt_scaling as ps  # noqa: E402

_CFG = _REPO / "analysis/analysis-configs/2026-10-05-platt-scaling-sweep-gemma27b-qwen-2.5-7b-v3-01.yaml"


def test_trial_order_nested_prefixes_and_determinism():
    docs = np.repeat(np.arange(10), 5)                     # 10 docs x 5 rows
    a = ps.trial_order(docs, seed=0, trial=3)
    assert sorted(a.tolist()) == list(range(50))
    assert set(a[:10]) <= set(a[:25])                      # prefixes nest by construction
    assert (a == ps.trial_order(docs, seed=0, trial=3)).all()
    assert not (a == ps.trial_order(docs, seed=0, trial=4)).all()
    assert not (a == ps.trial_order(docs, seed=1, trial=3)).all()


def test_trial_order_spreads_over_documents():
    # docs A,B,C with 6,2,1 rows. Ordering: round-robin over a random doc order.
    docs = np.array(list("AAAAAABBC"))
    for trial in range(20):
        o = ps.trial_order(docs, seed=0, trial=trial)
        first3 = docs[o[:3]]
        assert sorted(first3.tolist()) == ["A", "B", "C"]   # one from each doc in pass 1
        assert sorted(docs[o[3:5]].tolist()) == ["A", "B"]  # pass 2: C exhausted, skipped
        assert docs[o[5:]].tolist() == ["A"] * 4            # then only A remains
        # same document cycle each pass
        assert [d for d in docs[o[3:5]]] == [d for d in docs[o[:3]] if d != "C"]
    # row choice within a doc is random: A's first pick varies across trials
    firsts = {int(ps.trial_order(docs, 0, t)[np.flatnonzero(docs[ps.trial_order(docs, 0, t)] == "A")[0]]) for t in range(50)}
    assert len(firsts) > 1


def test_smece_known_answer_matches_relplot_diagram():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 4000)
    y = rng.uniform(0, 1, 4000) < p                       # perfectly calibrated -> smECE ~ 0
    y_bad = rng.uniform(0, 1, 4000) < (1 - p)             # anti-calibrated -> large
    ce, ce_bad = ps.smece(p, y), ps.smece(p, y_bad)
    assert ce < 0.05 < 0.2 < ce_bad
    assert ps.smece(p, y) == ce                            # deterministic
    d = ps.relplot.prepare_rel_diagram(p, y, num_bootstrap=5, plot_confidence_band=False, plot_bag_lines=False)
    assert abs(d["ce"] - ce) < 1e-9                        # same number v3 reports


def test_summarize_trials_hand_values():
    import pandas as pd
    rows = [dict(zip(("Train dataset", "Test dataset", "Type", "Platt N", "SmECE"), ("a", "a", "Probe", 10, v)))
            for v in (0.0, 0.1, 0.2, 0.3, 0.4)]
    out = ps.summarize_trials(pd.DataFrame(rows), 0.5)
    r = out.iloc[0]
    assert r["n_used"] == 5 and abs(r["SmECE"] - 0.2) < 1e-12
    assert abs(r["SmECE_lo"] - 0.1) < 1e-12 and abs(r["SmECE_hi"] - 0.3) < 1e-12   # 25th/75th pct


def test_committed_config_loads():
    cfg = ac.load_platt_sweep_config(_CFG)
    assert cfg["params"]["platt_ns"] == [10, 50, 100, 250, 500, 1000]
    assert cfg["params"]["n_trials"] == 100 and cfg["params"]["ci"] == 0.95
    assert cfg["params"]["single_class_policy"] == "drop"


@pytest.mark.parametrize("mutate", [
    lambda p: p.pop("platt_ns"),
    lambda p: p.pop("n_trials"),
    lambda p: p.pop("ci"),
    lambda p: p.pop("single_class_policy"),
    lambda p: p.update(platt_ns=[]),
    lambda p: p.update(platt_ns=[10, 10, 50]),
    lambda p: p.update(platt_ns=[50, 10]),
    lambda p: p.update(platt_ns=[0, 10]),
    lambda p: p.update(platt_ns=[True, 10]),
    lambda p: p.update(n_trials=1),
    lambda p: p.update(ci=1.0),
    lambda p: p.update(single_class_policy="skip"),
    lambda p: p.update(n_boot=200),   # old key not allowed
    lambda p: p.update(platt_n=100),   # v3 key not allowed
])
def test_loader_rejects_malformed(tmp_path, mutate):
    cfg = copy.deepcopy(yaml.safe_load(_CFG.read_text()))
    mutate(cfg["params"])
    p = tmp_path / f"{cfg['id']}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        ac.load_platt_sweep_config(p)
