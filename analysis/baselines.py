"""Compare MeasurementLM against external baselines (e.g. NuExtract-2.0-8B).

Mirrors analysis/ablation.py's structure and matching logic exactly, but all
arms (MeasurementLM extraction models and external baselines alike) are loaded
via load_extraction() (baselines are registered as pseudo "extraction models"
under extraction/{model}/{date}/, so no new loader or path-resolution code is
needed).

Unlike the ablation comparison (fixed set of arms per model), a dataset here
can compare an arbitrary set of models at once -- e.g. several MeasurementLM
backbones plus NuExtract, ChatExtract, and GLiNER -- so results are stored in
long/tidy form: one row per (dataset, model).
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
from scholarlm.utils.normalization import canonical_units, parse_value
from experiments.run_extraction import load_dataset_config
import paths


# External baselines, which are never shown the ground truth's unit vocabulary and so
# must be transcribed into it before matching. MeasurementLM arms are excluded
# deliberately: they are prompted with that vocabulary already, and normalising them
# too would only raise their scores, so leaving them raw keeps the reported gap a
# conservative one rather than one this step helped produce.
EXTERNAL_BASELINES = {
    'nuextract-2.0-8b',
    'chatextract-gemma-3-27b',
    'gliner-large-v1',
}


# Spellings of "this row carries no unit". Extractors emit these instead of a null.
_EMPTY_UNITS = {'', '-', '--', 'na', 'n/a', 'none', 'null', 'nan',
                'unknown', 'unitless', 'dimensionless'}


def build_unit_vocabulary(ground_truth_df):
    """Map each canonical unit form to the ground truth's own spelling of it."""
    vocabulary = {}
    for unit in ground_truth_df['units'].dropna().unique():
        canonical = canonical_units(unit)
        if canonical and canonical not in vocabulary:
            vocabulary[canonical] = unit
    return vocabulary


def unitless_attributes(ground_truth_df):
    """Attributes the ground truth records without any unit (e.g. pH).

    Read off the ground truth rather than hard-coded, so a new dataset needs no
    change here: an attribute counts as unitless only if *no* ground-truth row for
    it carries a unit.
    """
    with_units = ground_truth_df.groupby('attribute')['units'].apply(
        lambda s: s.notna().any()
    )
    return set(with_units[~with_units].index)


def _is_empty_units(u):
    if u is None or (isinstance(u, float) and np.isnan(u)):
        return True
    return str(u).strip().lower() in _EMPTY_UNITS


# NOTE (2026-07-14): drop_incomplete_measurements is currently unused. It withholds
# valueless/unitless rows from matching and forces them invalid in the validity
# denominator, which is the *right* behavior -- but it's what made validity_rate here
# diverge from analysis/ablation.py's numbers for llama-3.1-8b, which has a lot of such
# rows. Reverted the call site below to match ablation.py's unfiltered approach (where
# the judge's label alone decides those rows) until the two are reconciled on purpose.
def drop_incomplete_measurements(df, judged, unitless):
    """Withhold rows that do not state a measurement from matching.

    A row is a measurement only if it has a numeric value and -- unless its attribute
    is one the ground truth records without units -- a unit. Anything else is not a
    measurement at all, so it is removed from the pool of rows allowed to match the
    ground truth.

    It is *not* removed from the system's output. The returned ``n_total`` is the
    pre-filter row count, and callers pass it to ``validity_rate`` as ``denominator_n``
    so that withheld rows stay in the validity denominator and count as invalid. They
    have to: a row with no value is still something the system emitted, and excusing it
    would reward a model for flooding its output with unusable rows -- which is exactly
    what the external baselines do (GLiNER leaves 70% of its nfix units null). So this
    filter can only lower validity, never raise it, and it lowers it most for the arms
    that emit the most junk.

    Returns:
        ``(df, judged, n_total)`` -- filtered frames plus the pre-filter row count.
        ``judged`` is positionally aligned with ``df`` (``validity_rate`` requires equal
        lengths), so it is filtered by the same mask.
    """
    if judged is not None and len(judged) != len(df):
        raise ValueError(
            f"judgements ({len(judged)} rows) are not aligned with the extraction "
            f"({len(df)} rows); they are matched positionally, so filtering them "
            f"together would silently mislabel rows"
        )

    n_total = len(df)

    keep = df['converted_value'].notna()
    needs_units = ~df['attribute'].isin(unitless)
    keep &= ~(needs_units & df['units'].map(_is_empty_units))

    withheld = int((~keep).sum())
    if withheld:
        print(f"    {withheld}/{n_total} rows ({withheld/n_total:.1%}) lack a value or "
              f"a unit: withheld from matching, counted invalid")

    df = df[keep].reset_index(drop=True)
    if judged is not None:
        judged = judged[keep.to_numpy()].reset_index(drop=True)
    return df, judged, n_total


def normalize_baseline_extraction(df, unit_vocabulary):
    """Rewrite a baseline's ``value``/``units`` into the ground truth's notation.

    Matching compares these two columns by exact string equality, so a baseline that
    reported the right measurement in its own notation -- ``nmol N L−1 h−1`` for the
    ground truth's ``nmol N L⁻¹ h⁻¹``, or ``19.9 ± 2.3`` for ``19.9`` -- is scored as
    a miss over a spelling difference. This transcribes those two fields; the matching
    rules themselves stay exactly as they are, for every arm.

    A unit is only rewritten when its canonical form (see
    scholarlm.utils.normalization, which restates encoding but never repairs a missing
    exponent or analyte) coincides with that of a ground-truth unit. Anything with no
    counterpart keeps its original string and remains a genuine mismatch.
    """
    df = df.copy()
    df['value'] = df['value'].map(parse_value)
    df['units'] = df['units'].map(
        lambda u: unit_vocabulary.get(canonical_units(u), u)
    )
    return df


def f1_score(recovery, validity):
    """Harmonic mean of recovery and validity. 0.0 if both are 0."""
    if pd.isna(recovery) or pd.isna(validity) or recovery + validity == 0:
        return float('nan') if pd.isna(recovery) or pd.isna(validity) else 0.0
    return 2 * recovery * validity / (recovery + validity)


def _load_and_score(dataset, config, model, date, ground_truth_df, strict_matching, fuzzy_matching, fuzzy_threshold, cache_tag, unit_vocabulary):
    """Load an extraction run, match against ground truth, and return (recovery, validity) triples."""
    path = paths.find_extraction_final(dataset, model, date)
    resolved_date = Path(path).parent.name

    records = load_extraction(dataset, model, resolved_date)
    df = pd.DataFrame(records)
    if model in EXTERNAL_BASELINES:
        df = normalize_baseline_extraction(df, unit_vocabulary)
    df = process_extraction_df(df, dataset, config)

    try:
        judged = pd.DataFrame(load_combined_judgements(dataset, model, resolved_date))
    except FileNotFoundError:
        judged = None

    cache_path = paths.extraction(dataset, model, resolved_date) / f'match_cache_{cache_tag}_v2.pkl'

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


def compute_baseline_metrics(dataset, models_config):
    """Compute recovery, validity, and F1 metrics for a set of models on one dataset.

    Args:
        dataset: Dataset name.
        models_config: dict mapping model -> date (str). ``model`` is anything
            registered as a pseudo extraction model under extraction/{model}/,
            i.e. a MeasurementLM backbone (e.g. 'gemma-3-27b') or an external
            baseline (e.g. 'nuextract-2.0-8b', 'chatextract-gemma-3-27b',
            'gliner-large-v1'). A None date skips that model (row of NaNs).

    Returns:
        Long/tidy DataFrame with one row per model: dataset, model, recovery
        (+ CI), validity (+ CI), f1, has_judge.
    """
    config = load_dataset_config(dataset)
    ground_truth_df = load_ground_truth(config)
    strict_matching, fuzzy_matching, fuzzy_threshold = get_matching_rules(dataset)
    unit_vocabulary = build_unit_vocabulary(ground_truth_df)

    results = []

    for model, date in models_config.items():
        print(f"\n  Processing model: {model}")
        row = {'dataset': dataset, 'model': model}

        if date is None:
            print(f"    {model}: no date configured, skipping.")
            row['recovery'] = np.nan
            row['validity'] = np.nan
            row['f1'] = np.nan
            results.append(row)
            continue

        try:
            resolved_date, (recov, recov_lo, recov_hi), (valid, valid_lo, valid_hi), has_judge = _load_and_score(
                dataset, config, model, date,
                ground_truth_df, strict_matching, fuzzy_matching, fuzzy_threshold,
                cache_tag=model, unit_vocabulary=unit_vocabulary,
            )
            row['resolved_date'] = resolved_date
            row['recovery'] = recov
            row['recovery_ci_lo'] = recov_lo
            row['recovery_ci_hi'] = recov_hi
            row['validity'] = valid
            row['validity_ci_lo'] = valid_lo
            row['validity_ci_hi'] = valid_hi
            row['f1'] = f1_score(recov, valid)
            row['has_judge'] = has_judge
            print(f"    {model} ({resolved_date}): "
                  f"recovery={recov:.3f} [{recov_lo:.3f}, {recov_hi:.3f}], "
                  f"validity={valid:.3f} [{valid_lo:.3f}, {valid_hi:.3f}], "
                  f"f1={row['f1']:.3f}"
                  f"{'' if has_judge else ' (no judge — validity is a lower bound)'}")
        except FileNotFoundError:
            print(f"    {model}: not found, skipping.")
            row['recovery'] = np.nan
            row['validity'] = np.nan
            row['f1'] = np.nan
        except Exception as e:
            print(f"    {model} ERROR: {e}")
            row['recovery'] = np.nan
            row['validity'] = np.nan
            row['f1'] = np.nan

        results.append(row)

    return pd.DataFrame(results)


def main():
    # Fill in with the extraction dates you want to compare, per dataset.
    # Each dataset maps model -> extraction date; models can be MeasurementLM
    # backbones or external baselines (registered as pseudo extraction models).
    baseline_configs = {
        'pond': {
            'llama-3.1-8b': '2026_05_04',
            'gemma-3-27b': '2026_05_05',
            'gpt-oss-120b': '2026_05_02',
            'nuextract-2.0-8b': '2026_07_11',
            'chatextract-gemma-3-27b': '2026_07_11',
            'gliner-large-v1': '2026_07_11',
        },
        'nfix': {
            'llama-3.1-8b': '2026_05_05',
            'gemma-3-27b': '2026_05_06',
            'gpt-oss-120b': '2026_05_03',
            'nuextract-2.0-8b': '2026_07_11',
            'chatextract-gemma-3-27b': '2026_07_11',
            'gliner-large-v1': '2026_07_11',
        },
        'supermat': {
	    'llama-3.1-8b': '2026_07_13',
            'gemma-3-27b': '2026_07_09',
	    'gpt-oss-120b': '2026_07_13',
            'nuextract-2.0-8b': '2026_07_13',
            'chatextract-gemma-3-27b': '2026_07_13',
            'gliner-large-v1': '2026_07_13',
        },
        # nuextract-2.0-8b is deliberately omitted (not just None): it reads
        # rendered PDF page images via processed_pdfs/, and measeval is a
        # text-only dataset with no PDFs to render (see data/measeval/README.md)
        # -- there is no input run_baseline_nuextract.py could ever consume
        # here. All arms below are 2026_08_01 runs made under the
        # quantity-first design (see experiments/configs/measeval.py's module
        # docstring), so the comparison is apples-to-apples.
        'measeval': {
            'gemma-3-27b': '2026_08_01',
            'llama-3.1-8b': '2026_08_01',
            'gpt-oss-120b': '2026_08_01',
            'chatextract-gemma-3-27b': '2026_08_01',
            'gliner-large-v1': '2026_08_01',
        },
    }

    output_dir = Path('results/baselines/')
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset, models_config in baseline_configs.items():
        print(f"Processing dataset: {dataset}")
        results_df = compute_baseline_metrics(dataset, models_config)

        output_path = output_dir / f'baselines_{dataset}.csv'
        results_df.to_csv(output_path, index=False)

        print(f"\n{'='*60}")
        print(f"Results saved to {output_path}")
        print(f"{'='*60}")
        print(results_df.round(3))


if __name__ == '__main__':
    main()
