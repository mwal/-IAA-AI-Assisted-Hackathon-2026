import pandas as pd
import numpy as np
def enrich_history(d, X, claim_links):
    """History is product-specific; denominators are prior covered exposure.

    Count all reported statuses, avoiding future knowledge of decline decisions.
    Never use historical incurred-cost values or loss-period outcomes as inputs.
    """
    keys = d[['row_id', 'policy_id', 'customer_id', 'address_id']].copy()
    for col in keys:
        keys[col] = keys[col].astype(str)
    events = claim_links.loc[claim_links.line.eq(str(d.line.iloc[0])) & claim_links.row_id.notna()]
    events = events.merge(keys, on='row_id', how='inner', validate='many_to_one')
    events['report_date'] = pd.to_datetime(events.report_date)
    events['loss_year'] = pd.to_datetime(events.loss_date).dt.year
    new_cols = []
    for entity in ['policy_id', 'customer_id', 'address_id']:
        current_key = d[entity].astype(str)
        names = {s: 'r2_' + entity + '_' + s for s in [
            'prior_exposure', 'prior_reported_count', 'smoothed_reported_frequency',
            'last_year_exposure', 'last_year_reported_count', 'last_year_smoothed_frequency',
            'days_since_last_report']}
        for name in names.values():
            X[name] = np.nan
            new_cols.append(name)
        for year in sorted(d.year.unique()):
            mask = d.year.eq(year)
            cutoff = pd.Timestamp(int(year), 1, 1)
            past_coverage = d.loc[d.year < year]
            past_reports = events.loc[events.report_date < cutoff]
            exposure = past_coverage.groupby(entity, observed=True).exposure.sum()
            count = past_reports.groupby(entity, observed=True).size()
            last_exposure = past_coverage.loc[past_coverage.year.eq(year-1)].groupby(entity, observed=True).exposure.sum()
            last_count = past_reports.loc[past_reports.loss_year.eq(year-1)].groupby(entity, observed=True).size()
            denominator = float(past_coverage.exposure.sum())
            population_frequency = len(past_reports)/denominator if denominator > 0 else 0.
            key = current_key.loc[mask]
            ex = key.map(exposure).fillna(0.).astype(float)
            ct = key.map(count).fillna(0.).astype(float)
            le = key.map(last_exposure).fillna(0.).astype(float)
            lc = key.map(last_count).fillna(0.).astype(float)
            X.loc[mask, names['prior_exposure']] = ex.to_numpy()
            X.loc[mask, names['prior_reported_count']] = ct.to_numpy()
            X.loc[mask, names['smoothed_reported_frequency']] = ((ct+2*population_frequency)/(ex+2)).to_numpy()
            X.loc[mask, names['last_year_exposure']] = le.to_numpy()
            X.loc[mask, names['last_year_reported_count']] = lc.to_numpy()
            X.loc[mask, names['last_year_smoothed_frequency']] = ((lc+2*population_frequency)/(le+2)).to_numpy()
            last_report = key.map(past_reports.groupby(entity, observed=True).report_date.max())
            recency = (cutoff - pd.to_datetime(last_report)).dt.days
            assert (recency.dropna() > 0).all()
            X.loc[mask, names['days_since_last_report']] = recency.to_numpy()
    if 'construction_type' in X:
        # A hypothesis feature, not a claim that T and W have confirmed equal semantics.
        # Preserve the separate original categories as well as this union indicator.
        X['r2_construction_t_or_w'] = X.construction_type.isin(['t', 'w']).astype(float)
        new_cols.append('r2_construction_t_or_w')
    return new_cols
