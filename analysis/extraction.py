import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'experiments'))
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np
from analysis.loaders import load_extraction, load_combined_judgements, load_ground_truth, cached_match
from scholarlm.utils.unit_conversion import apply_unit_conversion
from analysis.metrics import recovery_rate, hallucination_rate
from experiments.run_extraction import load_dataset_config
import paths


def compute_metrics(dataset, model, extraction_date):
    """Compute recovery and hallucination metrics for a single extraction run."""
    
    # Load configuration and ground truth
    config = load_dataset_config(dataset)
    ground_truth_df = load_ground_truth(config)
    
    # Define matching rules based on dataset
    if 'pond' in dataset:
        STRICT_MATCHING = {'document_id': 'document_id', 'attribute': 'attribute', 'value': 'converted_value'}
        FUZZY_MATCHING = {'name': 'name', 'location': 'location', 'ecosystem': 'ecosystem'}
    elif 'nfix' in dataset:
        STRICT_MATCHING = {'document_id': 'document_id', 'attribute': 'attribute', 'value': 'converted_value'}
        FUZZY_MATCHING = {"name": "name", "site_type": "site_type"}
    else:
        raise ValueError(f"Dataset not recognized: {dataset}")
    
    # Load extraction data
    records = load_extraction(dataset, model, extraction_date)
    extraction_df = pd.DataFrame(records)
    extraction_df = apply_unit_conversion(extraction_df, config.unit_conversion_table)
    
    # Get cache path
    cache_path = paths.extraction(dataset, model, extraction_date) / 'match_cache.pkl'
    
    # Load judgements if available
    try:
        judgements = load_combined_judgements(dataset, model, extraction_date)
        judged_df = pd.DataFrame(judgements)
    except Exception:
        judged_df = None
    
    # Compute metrics with fuzzy threshold
    threshold = 1/3
    
    recovery = recovery_rate(
        extraction_df=extraction_df,
        ground_truth_df=ground_truth_df,
        strict_matching=STRICT_MATCHING,
        fuzzy_matching=FUZZY_MATCHING,
        fuzzy_threshold=threshold,
        cache_path=cache_path,
    )
    
    hallucination = hallucination_rate(
        extraction_df=extraction_df,
        ground_truth_df=ground_truth_df,
        judged_df=judged_df,
        strict_matching=STRICT_MATCHING,
        fuzzy_matching=FUZZY_MATCHING,
        fuzzy_threshold=threshold,
        cache_path=cache_path,
        label_col="judgement_combined"
    ) if judged_df is not None else np.nan
    
    return {
        'dataset': dataset,
        'model': model,
        'extraction_date': extraction_date,
        'recovery': recovery,
        'hallucination': hallucination,
    }


def main():
    # Define runs: list of (dataset, model, extraction_date) tuples
    runs = [
        ('pond', 'llama-3.1-8b', None),
        ('nfix', 'llama-3.1-8b', None),
        ('pond', 'gemma-3-27b', None),
        ('nfix', 'gemma-3-27b', None),
        ('pond', 'gpt-oss-120b', None), 
        ('nfix', 'gpt-oss-120b', None),
    ]
    
    results = []
    
    for dataset, model, extraction_date in runs:
        try:
            print(f"Computing metrics for {dataset} / {model} / {extraction_date}...")
            result = compute_metrics(dataset, model, extraction_date)
            results.append(result)
            print(f"  Recovery: {result['recovery']:.3f}, Hallucination: {result['hallucination']:.3f}")
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    output_path = Path('results/extraction.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    print(results_df)


if __name__ == '__main__':
    main()