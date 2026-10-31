"""Audit the original Parquet tables and construct coverage-row targets.

Run with the environment described in RUN_INSTRUCTIONS.txt. Original files are
read only. This script uses only the raw data, not analyses from other chats.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PRODUCTS = ('motor', 'buildings', 'contents')


def dump(value, path):
    path.write_text(json.dumps(value, indent=2, default=str))


def dates(frame):
    for col in ('period_start', 'period_end', 'loss_date', 'report_date'):
        if col in frame:
            frame[col] = pd.to_datetime(frame[col])
    return frame


def assemble(source, output):
    output.mkdir(parents=True, exist_ok=True)
    tables = {}
    for path in sorted(source.glob('*.parquet')):
        meta = pq.ParquetFile(path)
        tables[path.stem] = {
            'rows': meta.metadata.num_rows,
            'bytes': path.stat().st_size,
            'schema': {f.name: str(f.type) for f in meta.schema_arrow},
        }
    dump(tables, output / 'table_inventory.json')
    customers = pd.read_parquet(source / 'customers.parquet')
    addresses = pd.read_parquet(source / 'addresses.parquet')
    weather = pd.read_parquet(source / 'weather.parquet')
    claims = dates(pd.read_parquet(source / 'claims.parquet'))
    assert customers.customer_id.is_unique
    assert customers.customer_id.str.fullmatch(r'\d{8}').all()
    assert customers.customer_id.str.lstrip('0').is_unique
    assert addresses.address_id.is_unique
    assert not weather.duplicated(['address_id', 'year']).any()
    assert claims.claim_id.is_unique
    assert set(claims.status) <= {'open', 'closed', 'repudiated'}
    claims['eligible'] = claims.status.isin(['open', 'closed'])
    claims['net_incurred'] = claims.incurred_to_date - claims.recoveries
    assert claims[['paid_to_date', 'incurred_to_date', 'recoveries']].notna().all().all()
    assert np.isfinite(claims.net_incurred).all()
    assert (claims.net_incurred >= 0).all(), 'Investigate negative costs before training.'
    assert claims.loss_date.notna().all() and claims.report_date.notna().all()
    assert (claims.report_date >= claims.loss_date).all()
    assert claims.report_date.max() <= pd.Timestamp('2024-12-31')
    audit = {
        'source': str(source),
        'claim_status_counts': claims.status.value_counts().to_dict(),
        'weather_years': weather.year.value_counts().sort_index().to_dict(),
        'claim_loss_date_range': [claims.loss_date.min(), claims.loss_date.max()],
        'claim_report_date_range': [claims.report_date.min(), claims.report_date.max()],
        'products': {},
        'assumptions': [
            'Repudiated means declined and is excluded.',
            'Recoveries are deducted once from incurred; excess is not deducted again.',
            'Dates, rather than supplied exposure, determine inclusive exposure / 365.',
            'Coverage customer IDs are padded to eight digits, matching unique canonical customer keys.',
            'Weather predictors use year t-1, not realized weather in year t.',
            'Customer and address characteristics are used as supplied; historical versions are unavailable.',
            'Exact competition scorer code was not supplied: unweighted row-total Tweedie D2 at power 1.5 is assumed.',
        ],
    }
    # Keep a conservative subset of shared customer attributes. No names or emails.
    customer_cols = ['customer_id', 'occupation', 'marital_status',
                     'household_size', 'credit_band', 'customer_since_year']
    customer_features = customers[customer_cols]
    weather_cols = [c for c in weather if c not in ['address_id', 'year']]
    lag_weather = weather.rename(columns={c: 'prior_weather_' + c for c in weather_cols})
    lag_weather['year'] += 1
    all_ids, all_test_ids, linked_frames, yearly_summaries = [], [], [], []
    for product in PRODUCTS:
        print('Assembling', product, flush=True)
        train = dates(pd.read_parquet(source / f'policies_{product}.parquet'))
        test = dates(pd.read_parquet(source / f'test_{product}.parquet'))
        ntrain = len(train)
        all_test_ids.append(test.row_id)
        coverage = pd.concat([train.assign(is_test=False), test.assign(is_test=True)], ignore_index=True)
        raw_customer_missing = int((~coverage.customer_id.isin(customers.customer_id)).sum())
        coverage['customer_id'] = coverage.customer_id.astype(str).str.zfill(8)
        assert coverage.row_id.is_unique
        all_ids.append(coverage.row_id)
        assert coverage[['row_id', 'policy_id', 'customer_id', 'address_id', 'period_start', 'period_end']].notna().all().all()
        coverage['line'] = product
        coverage['year'] = coverage.period_start.dt.year
        coverage['covered_days'] = (coverage.period_end - coverage.period_start).dt.days + 1
        assert (coverage.covered_days > 0).all()
        assert (coverage.period_start.dt.year == coverage.period_end.dt.year).all()
        assert set(coverage.loc[~coverage.is_test, 'year']) == {2021, 2022, 2023, 2024}
        assert set(coverage.loc[coverage.is_test, 'year']) == {2025}
        coverage = coverage.rename(columns={'exposure': 'source_exposure'})
        coverage['exposure'] = coverage.covered_days / 365.
        difference = (coverage.source_exposure - coverage.exposure).abs()
        ordered = coverage.sort_values(['policy_id', 'period_start', 'period_end'])
        running_end = ordered.groupby('policy_id').period_end.cummax()
        previous_end = running_end.groupby(ordered.policy_id).shift()
        overlap = ordered.period_start <= previous_end
        assert not overlap.any(), 'Coverage overlaps need resolution before claim assignment.'
        cp = claims.loc[claims.line.eq(product)].copy()
        candidates = cp.merge(train[['policy_id', 'row_id', 'period_start', 'period_end']],
                              on='policy_id', how='left', validate='many_to_many')
        matched = candidates.loc[candidates.loss_date.between(candidates.period_start, candidates.period_end)].copy()
        assert matched.claim_id.is_unique, 'A claim matched multiple coverage periods.'
        assignments = cp.merge(matched[['claim_id', 'row_id']], on='claim_id', how='left', validate='one_to_one')
        assignments['linkage_status'] = np.where(assignments.row_id.notna(), 'matched', 'unmatched')
        assignments['unmatched_reason'] = np.where(assignments.row_id.notna(), '',
            np.where(assignments.loss_date < pd.Timestamp('2021-01-01'),
                     'loss_before_training_window', 'no_matching_coverage'))
        linked_frames.append(assignments)
        eligible_matched = matched.loc[matched.eligible]
        totals = eligible_matched.groupby('row_id').agg(
            target_net_incurred=('net_incurred', 'sum'),
            eligible_reported_claim_count=('claim_id', 'size'),
            positive_reported_claim_count=('net_incurred', lambda s: int((s > 0).sum())),
            paid_total=('paid_to_date', 'sum'),
            incurred_total=('incurred_to_date', 'sum'),
            recoveries_total=('recoveries', 'sum'))
        coverage = coverage.merge(totals, on='row_id', how='left', validate='one_to_one')
        for col in totals.columns:
            coverage.loc[~coverage.is_test, col] = coverage.loc[~coverage.is_test, col].fillna(0.)
        assert coverage.loc[coverage.is_test, 'target_net_incurred'].isna().all()
        assert np.isclose(coverage.target_net_incurred.sum(), eligible_matched.net_incurred.sum())
        customer_missing = (~coverage.customer_id.isin(customers.customer_id)).sum()
        address_missing = (~coverage.address_id.isin(addresses.address_id)).sum()
        assert customer_missing == 0 and address_missing == 0
        coverage = coverage.merge(customer_features, on='customer_id', how='left', validate='many_to_one')
        coverage = coverage.merge(addresses, on='address_id', how='left', validate='many_to_one')
        coverage = coverage.merge(lag_weather, on=['address_id', 'year'], how='left', validate='many_to_one')
        assert len(coverage) == ntrain + len(test)
        for col in coverage.select_dtypes('object'):
            coverage[col] = coverage[col].astype('category')
        coverage.to_parquet(output / f'assembled_{product}.parquet', index=False)
        tr = coverage.loc[~coverage.is_test]
        summary = tr.groupby('year').agg(rows=('row_id', 'size'), exposure=('exposure', 'sum'),
            positive_rows=('target_net_incurred', lambda s: int((s > 0).sum())),
            reported_claims=('eligible_reported_claim_count', 'sum'),
            net_incurred=('target_net_incurred', 'sum'))
        summary['net_cost_per_exposure'] = summary.net_incurred / summary.exposure
        summary['product'] = product
        yearly_summaries.extend(summary.reset_index().to_dict('records'))
        audit['products'][product] = {
            'train_rows': ntrain, 'test_rows': len(test),
            'customer_links_repaired_by_zero_padding': raw_customer_missing,
            'source_exposure_material_discrepancy_rows': int((difference > .000051).sum()),
            'source_exposure_greater_than_1_01_rows': int((coverage.source_exposure > 1.01).sum()),
            'source_exposure_nonpositive_rows': int((coverage.source_exposure <= 0).sum()),
            'maximum_source_exposure': float(coverage.source_exposure.max()),
            'calculated_exposure_range': [float(coverage.exposure.min()), float(coverage.exposure.max())],
            'overlapping_rows': int(overlap.sum()),
            'missing_customer_links': int(customer_missing), 'missing_address_links': int(address_missing),
            'matched_claims': len(matched),
            'unmatched_claims_by_reason': assignments.loc[assignments.row_id.isna(), 'unmatched_reason'].value_counts().to_dict(),
            'eligible_matched_claims': len(eligible_matched),
            'eligible_matched_net_incurred': float(eligible_matched.net_incurred.sum()),
            'positive_target_rows': int((tr.target_net_incurred > 0).sum()),
            'zero_target_rows': int((tr.target_net_incurred == 0).sum()),
            'multiple_claim_rows': int((tr.eligible_reported_claim_count > 1).sum()),
            'test_returning_customer_fraction': float(test.customer_id.isin(train.customer_id).mean()),
            'test_returning_policy_fraction': float(test.policy_id.isin(train.policy_id).mean()),
            'missing_values': coverage.isna().sum().loc[lambda s: s > 0].to_dict(),
        }
    assert pd.concat(all_ids).is_unique
    test_ids = pd.concat(all_test_ids)
    sample = pd.read_parquet(source / 'sample_submission.parquet')
    assert sample.row_id.is_unique and set(sample.row_id) == set(test_ids)
    audit['sample_submission_rows'] = len(sample)
    dump(audit, output / 'audit.json')
    dump(yearly_summaries, output / 'yearly_summary.json')
    links = pd.concat(linked_frames, ignore_index=True)
    links.to_parquet(output / 'claim_assignments.parquet', index=False)
    links.loc[links.row_id.isna()].to_parquet(output / 'unmatched_claims.parquet', index=False)
    # Delay diagnostics describe observed reports, not an ultimate delay estimate.
    links['delay_days'] = (links.report_date - links.loss_date).dt.days
    links['loss_year'] = links.loss_date.dt.year
    diagnostics = []
    for (line, year), g in links.loc[links.eligible].groupby(['line', 'loss_year']):
        diagnostics.append({'product': line, 'loss_year': int(year), 'observed_claims': len(g),
            'median_observed_delay_days': float(g.delay_days.median()),
            'p90_observed_delay_days': float(g.delay_days.quantile(.9)),
            'max_observed_delay_days': int(g.delay_days.max()),
            'open_fraction_at_2024_year_end': float(g.status.eq('open').mean())})
    dump(diagnostics, output / 'observed_reporting_diagnostics.json')
    print(json.dumps(audit['products'], indent=2, default=str), flush=True)
    print('Assembly complete; all linkage and target reconciliation checks passed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'data_audit')
    args = parser.parse_args()
    assemble(args.source, args.output)
