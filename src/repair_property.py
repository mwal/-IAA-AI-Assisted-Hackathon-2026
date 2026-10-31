"""Conservative, past-only checks for apparent 1,000x property-value errors.

No claim outcomes enter this repair. Unsupported extremes are left unchanged.
Run after assemble_data.py and before train_baselines.py.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent / 'data_audit'


def main():
    evidence_frames, summary = [], {}
    for product, threshold in [('buildings', 10_000_000.), ('contents', 2_000_000.)]:
        path = ROOT / f'assembled_{product}.parquet'
        d = pd.read_parquet(path)
        # Use a separately saved original on reruns, so the repair is idempotent.
        original_path = ROOT / f'original_sum_insured_{product}.parquet'
        if original_path.exists():
            original = pd.read_parquet(original_path).set_index('row_id').original_sum_insured
            assert set(original.index) == set(d.row_id.astype(str))
            d['sum_insured'] = d.row_id.astype(str).map(original).to_numpy()
        else:
            original = pd.DataFrame({'row_id': d.row_id.astype(str), 'original_sum_insured': d.sum_insured})
            original.to_parquet(original_path, index=False)
        ev = d[['row_id', 'policy_id', 'address_id', 'year', 'period_start', 'sum_insured']].copy()
        for col in ['row_id', 'policy_id', 'address_id']:
            ev[col] = ev[col].astype(str)
        ev = ev.rename(columns={'sum_insured': 'original_sum_insured'})
        ev['corrected_sum_insured'] = ev.original_sum_insured
        ev['screened_extreme'] = ev.original_sum_insured > threshold
        ev['corrected'] = False
        ev['reference_row_id'] = None
        ev['reference_year'] = np.nan
        ev['reference_sum_insured'] = np.nan
        ev['reference_period_start'] = pd.NaT
        ev['scaled_to_reference_ratio'] = np.nan
        for year in sorted(ev.year.unique()):
            current = ev.loc[ev.year.eq(year) & ev.screened_extreme]
            if current.empty:
                continue
            # Earlier calendar years only: information is available before the
            # annual forecast cutoff, even for mid-year coverage changes.
            previous = ev.loc[(ev.year < year) & ev.original_sum_insured.between(1, threshold)]
            previous = previous.sort_values(['period_start', 'row_id']).drop_duplicates(['policy_id', 'address_id'], keep='last')
            previous = previous[['policy_id', 'address_id', 'row_id', 'year', 'period_start', 'original_sum_insured']].rename(columns={
                'row_id': 'ref_row', 'year': 'ref_year', 'period_start': 'ref_start', 'original_sum_insured': 'ref_value'})
            matched = current.reset_index().merge(previous, on=['policy_id', 'address_id'], how='left', validate='many_to_one')
            ratio = (matched.original_sum_insured / 1000.) / matched.ref_value
            supported = matched.ref_value.notna() & ratio.between(.9, 1.1)
            selected = matched.loc[supported]
            ix = selected['index'].to_numpy()
            ev.loc[ix, 'corrected_sum_insured'] = selected.original_sum_insured.to_numpy() / 1000.
            ev.loc[ix, 'corrected'] = True
            ev.loc[ix, 'reference_row_id'] = selected.ref_row.to_numpy()
            ev.loc[ix, 'reference_year'] = selected.ref_year.to_numpy()
            ev.loc[ix, 'reference_sum_insured'] = selected.ref_value.to_numpy()
            ev.loc[ix, 'reference_period_start'] = selected.ref_start.to_numpy()
            ev.loc[ix, 'scaled_to_reference_ratio'] = ratio.loc[supported].to_numpy()
        corrected = ev.loc[ev.corrected]
        assert (corrected.reference_year < corrected.year).all()
        assert np.allclose(corrected.original_sum_insured / corrected.corrected_sum_insured, 1000.)
        assert corrected.scaled_to_reference_ratio.between(.9, 1.1).all()
        d['sum_insured'] = ev.corrected_sum_insured.to_numpy()
        # Case normalization is supported by literal spelling only. T and W stay distinct.
        if product == 'buildings':
            d['construction_type'] = d.construction_type.astype('object').str.strip().str.lower().astype('category')
            annual = d.sort_values(['period_start', 'period_end', 'row_id']).drop_duplicates(
                ['policy_id', 'address_id', 'year'], keep='last')
            prior_codes = annual.loc[annual.year.eq(2022), ['policy_id', 'address_id', 'construction_type']].rename(
                columns={'construction_type': 'code_2022'})
            next_codes = annual.loc[annual.year.eq(2023), ['policy_id', 'address_id', 'construction_type']].rename(
                columns={'construction_type': 'code_2023'})
            transitions = prior_codes.merge(next_codes, on=['policy_id', 'address_id'], validate='one_to_one')
            counts = transitions.groupby(['code_2022', 'code_2023'], observed=True).size().rename('rows').reset_index()
            (ROOT / 'construction_code_transitions.json').write_text(json.dumps({
                'same_policy_address_2022_to_2023': counts.to_dict('records'),
                'decision': 'Normalize spelling case only; keep t and w distinct. Transitions do not establish category semantics.'
            }, indent=2, default=str))
        temp = path.with_suffix('.temporary.parquet')
        d.to_parquet(temp, index=False)
        temp.replace(path)
        ev['product'] = product
        evidence_frames.append(ev.loc[ev.screened_extreme])
        summary[product] = {'screening_threshold_CU': threshold,
            'screened_extreme_rows': int(ev.screened_extreme.sum()),
            'supported_corrections': int(ev.corrected.sum()),
            'historical_corrections': int((ev.corrected & ev.year.lt(2025)).sum()),
            'test_corrections': int((ev.corrected & ev.year.eq(2025)).sum()),
            'unsupported_extreme_rows_unchanged': int((ev.screened_extreme & ~ev.corrected).sum()),
            'rule': 'Divide by 1000 only when scaled value is within 10% of latest plausible earlier-calendar-year record for same policy and address. No future records or claim outcomes used.'}
        print(product, summary[product], flush=True)
    pd.concat(evidence_frames, ignore_index=True).to_parquet(ROOT / 'property_sum_insured_evidence.parquet', index=False)
    (ROOT / 'property_repairs.json').write_text(json.dumps(summary, indent=2))
    # Independently classify unmatched claims relative to first supplied coverage.
    unmatched = pd.read_parquet(ROOT / 'unmatched_claims.parquet')
    records = []
    for product in ('motor', 'buildings', 'contents'):
        d = pd.read_parquet(ROOT / f'assembled_{product}.parquet', columns=['policy_id', 'period_start'])
        first = d.groupby('policy_id', observed=True).period_start.min()
        g = unmatched.loc[unmatched.line.eq(product)].copy()
        g['first_supplied_coverage_start'] = g.policy_id.map(first)
        g['loss_before_first_supplied_coverage'] = g.loss_date < g.first_supplied_coverage_start
        assert g.loss_before_first_supplied_coverage.all()
        records.append(g)
    classified = pd.concat(records, ignore_index=True)
    classified.to_parquet(ROOT / 'unmatched_claims_classified.parquet', index=False)
    audit_path = ROOT / 'audit.json'
    audit = json.loads(audit_path.read_text())
    audit['property_repairs'] = summary
    audit['unmatched_claims_investigation'] = {
        'total': len(classified), 'all_before_first_supplied_coverage_for_policy': True,
        'interpretation': 'Incomplete supplied policy history; not established claim-record errors.'}
    audit_path.write_text(json.dumps(audit, indent=2, default=str))


if __name__ == '__main__':
    main()
