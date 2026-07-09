"""Compare MeasurementLM against external baselines (e.g. NuExtract-2.0-8B).

Mirrors analysis/ablation.py's structure and matching logic exactly, but both
arms are loaded via load_extraction() (baselines are registered as pseudo
"extraction models" under extraction/{model}/{date}/, so no new loader or
path-resolution code is needed).
"""
import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'experiments'))
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np
from analysis.loaders import load_extraction, load_combined_judgements, load_ground_truth, cached_match
from analysis.metrics import recovery_rate, validity_rate
from analysis.ablation import get_matching_rules, process_extraction_df
from experiments.run_extraction import load_dataset_config
import paths


def _load_and_score(dataset, config, model, date, ground_truth_df, strict_matching, fuzzy_matching, fuzzy_threshold, cache_tag):
    """Load an extraction run, match against ground truth, and return (recovery, validity) triples."""
    path = paths.find_extraction_final(dataset, model, date)
    resolved_date = Path(path).parent.name

    records = load_extraction(dataset, model, resolved_date)
    df = pd.DataFrame(records)
    df = process_extraction_df(df, dataset, config)

    cache_path = paths.extraction(dataset, model, resolved_date) / f'match_cache_{cache_tag}.pkl'

    try:
        judged = pd.DataFrame(load_combined_judgements(dataset, model, resolved_date))
    except FileNotFoundError:
        judged = None

    cached_match(
        ground_truth_df, df,
        strict_matching=strict_matching,
        fuzzy_matching=fuzzy_matching,
        fuzzy_threshold=0.0,
        cache_path=cache_path,
    )

    recov, recov_lo, recov_hi = recovery_rate(
        ground_truth_df, df,
        strict_matching=strict_matching,
        fuzzy_matching=fuzzy_matching,
        fuzzy_threshold=fuzzy_threshold,
        cache_path=cache_path,
        return_ci=True,
    )
    valid, valid_lo, valid_hi = validity_rate(
        ground_truth_df, df,
        strict_matching=strict_matching,
        fuzzy_matching=fuzzy_matching,
        fuzzy_threshold=fuzzy_threshold,
        judged_df=judged,
        cache_path=cache_path,
        return_ci=True,
    )
    return resolved_date, (recov, recov_lo, recov_hi), (valid, valid_lo, valid_hi), (judged is not None)


def compute_baseline_metrics(dataset, baseline_configs):
    """Compute recovery and validity metrics comparing MeasurementLM models against baselines.

    Args:
        dataset: Dataset name.
        baseline_configs: dict mapping mlm_model -> {'mlm_date': str, 'nuextract_date': str}.
    """
    config = load_dataset_config(dataset)
    ground_truth_df = load_ground_truth(config)
    strict_matching, fuzzy_matching, fuzzy_threshold = get_matching_rules(dataset)

    results = []

    for mlm_model, dates in baseline_configs.items():
        print(f"\n  Processing model: {mlm_model}")
        row = {'dataset': dataset, 'mlm_model': mlm_model}

        try:
            resolved_date, (recov, recov_lo, recov_hi), (valid, valid_lo, valid_hi), has_judge = _load_and_score(
                dataset, config, mlm_model, dates['mlm_date'],
                ground_truth_df, strict_matching, fuzzy_matching, fuzzy_threshold,
                cache_tag='mlm',
            )
            row['mlm_recovery'] = recov
            row['mlm_recovery_ci_lo'] = recov_lo
            row['mlm_recovery_ci_hi'] = recov_hi
            row['mlm_validity'] = valid
            row['mlm_validity_ci_lo'] = valid_lo
            row['mlm_validity_ci_hi'] = valid_hi
            row['mlm_has_judge'] = has_judge
            print(f"    MeasurementLM ({mlm_model}, {resolved_date}): "
                  f"recovery={recov:.3f} [{recov_lo:.3f}, {recov_hi:.3f}], "
                  f"validity={valid:.3f} [{valid_lo:.3f}, {valid_hi:.3f}]"
                  f"{'' if has_judge else ' (no judge — validity is a lower bound)'}")
        except Exception as e:
            print(f"    MeasurementLM ERROR: {e}")
            row['mlm_recovery'] = np.nan
            row['mlm_validity'] = np.nan

        if dates.get('nuextract_date') is None:
            row['nuextract_recovery'] = np.nan
            row['nuextract_validity'] = np.nan
            results.append(row)
            continue

        try:
            resolved_date, (recov, recov_lo, recov_hi), (valid, valid_lo, valid_hi), has_judge = _load_and_score(
                dataset, config, 'nuextract-2.0-8b', dates['nuextract_date'],
                ground_truth_df, strict_matching, fuzzy_matching, fuzzy_threshold,
                cache_tag='nuextract',
            )
            row['nuextract_recovery'] = recov
            row['nuextract_recovery_ci_lo'] = recov_lo
            row['nuextract_recovery_ci_hi'] = recov_hi
            row['nuextract_validity'] = valid
            row['nuextract_validity_ci_lo'] = valid_lo
            row['nuextract_validity_ci_hi'] = valid_hi
            row['nuextract_has_judge'] = has_judge
            print(f"    NuExtract-2.0-8B ({resolved_date}): "
                  f"recovery={recov:.3f} [{recov_lo:.3f}, {recov_hi:.3f}], "
                  f"validity={valid:.3f} [{valid_lo:.3f}, {valid_hi:.3f}]"
                  f"{'' if has_judge else ' (no judge — validity is a lower bound)'}")
        except FileNotFoundError:
            print(f"    NuExtract-2.0-8B: not found, skipping.")
            row['nuextract_recovery'] = np.nan
            row['nuextract_validity'] = np.nan
        except Exception as e:
            print(f"    NuExtract-2.0-8B ERROR: {e}")
            row['nuextract_recovery'] = np.nan
            row['nuextract_validity'] = np.nan

        results.append(row)

    return pd.DataFrame(results)


def main():
    # Fill in with the extraction dates you want to compare, per dataset.
    baseline_configs = {
        'pond': {
            'gemma-3-27b': {'mlm_date': '2026_05_05', 'nuextract_date': None},
        },
        'nfix': {
            'gemma-3-27b': {'mlm_date': '2026_05_06', 'nuextract_date': None},
        },
    }

    output_dir = Path('results/baselines/')
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset, configs in baseline_configs.items():
        print(f"Processing dataset: {dataset}")
        results_df = compute_baseline_metrics(dataset, configs)

        output_path = output_dir / f'baselines_{dataset}.csv'
        results_df.to_csv(output_path, index=False)

        print(f"\n{'='*60}")
        print(f"Results saved to {output_path}")
        print(f"{'='*60}")
        print(results_df.round(3))


if __name__ == '__main__':
    main()
