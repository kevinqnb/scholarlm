import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'experiments'))
sys.path.insert(0, str(REPO_ROOT))

import re
import pandas as pd
import numpy as np
from analysis.loaders import load_extraction, load_ablation, load_combined_judgements, load_ground_truth, cached_match
from analysis.metrics import recovery_rate, hallucination_rate
from experiments.run_extraction import load_dataset_config
from scholarlm.utils.unit_conversion import apply_unit_conversion
import paths

def get_matching_rules(dataset):
    """Get strict matching, fuzzy matching, and fuzzy threshold based on dataset type."""
    if 'pond' in dataset:
        return (
            {'document_id': 'document_id', 'attribute': 'attribute', 'value': 'converted_value', 'units': 'units'},
            {'name': 'name', 'location': 'location', 'ecosystem': 'ecosystem'},
            1/3,
        )
    elif 'nfix' in dataset:
        return (
            {'document_id': 'document_id', 'attribute': 'attribute', 'value': 'converted_value', 'units': 'units'},
            {"name": "name", "site_type": "site_type"},
            1/6,
        )
    else:
        raise ValueError(f"Dataset not recognized: {dataset}")


def process_extraction_df(extraction_df, dataset, config):
    """Apply unit conversion and normalization to extraction dataframe."""
    extraction_df = apply_unit_conversion(extraction_df, {}) # NOTE: no longer using dataset-specific unit conversion rules, but could be added back if needed

    # Unitless attributes should have units set to None to avoid matching issues
    extraction_df.loc[extraction_df.attribute == 'ph', 'units'] = None
    
    if 'nfix' in dataset:
        extraction_df['attribute'] = extraction_df['attribute'].map({
            'nfix_rate_areal': 'nfix_rate',
            'nfix_rate_volumetric': 'nfix_rate',
            'nfix_rate_mass': 'nfix_rate',
            'nfix_rate': 'nfix_rate'
        })
    
    return extraction_df


def compute_ablation_metrics(dataset, ablations_config):
    """Compute recovery and hallucination metrics for all ablations in a dataset."""
    
    config = load_dataset_config(dataset)
    ground_truth_df = load_ground_truth(config)
    strict_matching, fuzzy_matching, fuzzy_threshold = get_matching_rules(dataset)
    
    results = []
    
    for model, ablation_dates in ablations_config.items():
        print(f"\n  Processing model: {model}")
        row = {'dataset': dataset, 'model': model}
        
        # Process baseline
        try:
            baseline_date = ablation_dates['baseline']
            baseline_path = paths.find_extraction_final(dataset, model, baseline_date)
            baseline_date = Path(baseline_path).parent.name

            baseline_records = load_extraction(dataset, model, baseline_date)
            baseline_df = pd.DataFrame(baseline_records)
            baseline_df = process_extraction_df(baseline_df, dataset, config)

            baseline_cache_path = paths.extraction(dataset, model, baseline_date) / 'match_cache.pkl'

            try:
                baseline_judged = pd.DataFrame(load_combined_judgements(dataset, model, baseline_date))
            except FileNotFoundError:
                baseline_judged = None

            cached_match(
                ground_truth_df, baseline_df,
                strict_matching=strict_matching,
                fuzzy_matching=fuzzy_matching,
                fuzzy_threshold=0.0,
                cache_path=baseline_cache_path,
            )

            baseline_recov = recovery_rate(
                ground_truth_df, baseline_df,
                strict_matching=strict_matching,
                fuzzy_matching=fuzzy_matching,
                fuzzy_threshold=fuzzy_threshold,
                cache_path=baseline_cache_path
            )
            baseline_hall = hallucination_rate(
                ground_truth_df, baseline_df,
                strict_matching=strict_matching,
                fuzzy_matching=fuzzy_matching,
                fuzzy_threshold=fuzzy_threshold,
                judged_df=baseline_judged,
                cache_path=baseline_cache_path
            )

            row['baseline_recovery'] = baseline_recov
            row['baseline_hallucination'] = baseline_hall
            print(f"    Baseline: recovery={baseline_recov:.3f}, hallucination={baseline_hall:.3f}")

        except Exception as e:
            print(f"    Baseline ERROR: {e}")
            row['baseline_recovery'] = np.nan
            row['baseline_hallucination'] = np.nan
        
        # Process ablations
        for ablation_n, ablation_date in ablation_dates.items():
            if ablation_n == 'baseline':
                continue
            
            if ablation_date is None:
                row[f'ablation_{ablation_n}_recovery'] = np.nan
                row[f'ablation_{ablation_n}_hallucination'] = np.nan
                continue
            
            try:
                ablation_path = paths.find_extraction_final(dataset, model, ablation_date, ablation_n)
                ablation_date = Path(ablation_path).parent.name

                records = load_ablation(dataset, ablation_n, model, ablation_date)
                if len(records) == 0:
                    row[f'ablation_{ablation_n}_recovery'] = np.nan
                    row[f'ablation_{ablation_n}_hallucination'] = np.nan
                    continue

                ablation_df = pd.DataFrame(records)
                ablation_df = process_extraction_df(ablation_df, dataset, config)

                ablation_cache_path = paths.ablation(dataset, ablation_n, model, ablation_date) / 'match_cache.pkl'

                try:
                    ablation_judged = pd.DataFrame(load_combined_judgements(dataset, model, ablation_date, ablation=ablation_n))
                except FileNotFoundError:
                    ablation_judged = None

                cached_match(
                    ground_truth_df, ablation_df,
                    strict_matching=strict_matching,
                    fuzzy_matching=fuzzy_matching,
                    fuzzy_threshold=0.0,
                    cache_path=ablation_cache_path,
                )

                ablation_recov = recovery_rate(
                    ground_truth_df, ablation_df,
                    strict_matching=strict_matching,
                    fuzzy_matching=fuzzy_matching,
                    fuzzy_threshold=fuzzy_threshold,
                    cache_path=ablation_cache_path
                )
                ablation_hall = hallucination_rate(
                    ground_truth_df, ablation_df,
                    strict_matching=strict_matching,
                    fuzzy_matching=fuzzy_matching,
                    fuzzy_threshold=fuzzy_threshold,
                    judged_df=ablation_judged,
                    cache_path=ablation_cache_path
                )

                row[f'ablation_{ablation_n}_recovery'] = ablation_recov
                row[f'ablation_{ablation_n}_hallucination'] = ablation_hall
                print(f"    Ablation {ablation_n}: recovery={ablation_recov:.3f}, hallucination={ablation_hall:.3f}")
                
            except FileNotFoundError:
                print(f"    Ablation {ablation_n}: not found, skipping.")
                row[f'ablation_{ablation_n}_recovery'] = np.nan
                row[f'ablation_{ablation_n}_hallucination'] = np.nan
            except Exception as e:
                print(f"    Ablation {ablation_n} ERROR: {e}")
                row[f'ablation_{ablation_n}_recovery'] = np.nan
                row[f'ablation_{ablation_n}_hallucination'] = np.nan
        
        results.append(row)
    
    return pd.DataFrame(results)


def main():
    # Define ablation configurations per dataset
    ablation_configs = {
        'pond': {
            'llama-3.1-8b': {'baseline': '2026_05_04', '1': '2026_05_04', '2': '2026_05_07', '3': '2026_05_08', '4': '2026_05_04', '5': '2026_05_05', '6': '2026_05_05'},
            'gemma-3-27b': {'baseline': '2026_05_05', '1': '2026_05_04', '2': '2026_05_05', '3': '2026_05_06', '4': '2026_05_07', '5': '2026_05_06', '6': '2026_05_06'},
            'gpt-oss-120b': {'baseline': '2026_05_02', '1': '2026_05_03', '2': '2026_05_03', '3': '2026_05_03', '4': '2026_05_03', '5': '2026_05_03', '6': '2026_05_03'},
        },
        'nfix': {
            'llama-3.1-8b': {'baseline': '2026_05_05', '1': '2026_05_07', '2': '2026_05_08', '3': '2026_05_07', '4': '2026_05_06', '5': '2026_05_07', '6': '2026_05_07'},
            'gemma-3-27b': {'baseline': '2026_05_06', '1': '2026_05_07', '2': '2026_05_07', '3': '2026_05_07', '4': '2026_05_07', '5': '2026_05_07', '6': '2026_05_07'},
            'gpt-oss-120b': {'baseline': '2026_05_03', '1': '2026_05_06', '2': '2026_05_06', '3': '2026_05_06', '4': '2026_05_06', '5': '2026_05_07', '6': '2026_05_08'},
        },
    }
    
    output_dir = Path('results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset, ablations in ablation_configs.items():
        print(f"Processing dataset: {dataset}")
        results_df = compute_ablation_metrics(dataset, ablations)
        
        output_path = output_dir / f'ablation_{dataset}.csv'
        results_df.to_csv(output_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"Results saved to {output_path}")
        print(f"{'='*60}")
        print(results_df.round(3))


if __name__ == '__main__':
    main()