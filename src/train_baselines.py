"""Reproducible temporal baselines for expected coverage-row cost.

2022 selects model/iterations; 2023 is the main held-out assessment; 2024 is
an additional, less mature diagnostic. Outputs are predictions, not known losses.
"""
from pathlib import Path
import argparse
import gc
import json
import time
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import d2_tweedie_score, mean_tweedie_deviance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PRODUCTS = ('motor', 'buildings', 'contents')
EXCLUDE = {
    'row_id', 'policy_id', 'customer_id', 'address_id', 'period_start', 'period_end',
    'source_exposure', 'is_test', 'line', 'year', 'exposure', 'covered_days',
    'target_net_incurred', 'eligible_reported_claim_count', 'positive_reported_claim_count',
    'paid_total', 'incurred_total', 'recoveries_total',
    # Outcome or timing ambiguous; excluded until a data dictionary establishes availability.
    'renewal_decision', 'customer_service_contacts', 'missed_payments_12m',
    'written_premium', 'discount_applied', 'telematics_score',
    'customer_since_year', 'year_built',
    # Irrelevant cosmetic feature; omit to reduce noise.
    'vehicle_colour',
}


class TotalCostDeviance:
    """Exact sum of total-cost deviances for rate targets with sqrt(e) weights.

    CatBoost supplies log-rate predictions. Since d(e*y,e*mu)=sqrt(e)*d(y,mu)
    at p=1.5, this evaluates the unweighted total-cost metric without rescaling
    validation targets inside the modeling library.
    """
    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        mu = np.exp(np.clip(np.asarray(approxes[0]), -40, 40))
        y = np.asarray(target)
        w = np.ones(len(y)) if weight is None else np.asarray(weight)
        dev = 4 * (y / np.sqrt(mu) + np.sqrt(mu) - 2 * np.sqrt(y))
        return float(np.sum(w * dev)), float(len(y))

    def get_final_error(self, error, weight):
        return error / weight


def dump(value, path):
    path.write_text(json.dumps(value, indent=2, default=str))


def score(y, pred):
    pred = np.maximum(np.asarray(pred), 1e-10)
    y = np.asarray(y)
    return {'rows': len(y), 'd2': float(d2_tweedie_score(y, pred, power=1.5)),
            'mean_deviance': float(mean_tweedie_deviance(y, pred, power=1.5)),
            'observed_total': float(y.sum()), 'predicted_total': float(pred.sum()),
            'predicted_to_observed': float(pred.sum() / y.sum())}


def history_events(data_dir):
    keys, observations = [], []
    for product in PRODUCTS:
        d = pd.read_parquet(data_dir / f'assembled_{product}.parquet',
            columns=['row_id', 'customer_id', 'address_id', 'period_start'])
        for c in ['row_id', 'customer_id', 'address_id']:
            d[c] = d[c].astype(str)
        keys.append(d[['row_id', 'customer_id', 'address_id']])
        observations.append(d[['customer_id', 'address_id', 'period_start']])
    key = pd.concat(keys, ignore_index=True)
    links = pd.read_parquet(data_dir / 'claim_assignments.parquet', columns=['row_id', 'report_date'])
    links = links.loc[links.row_id.notna()].merge(key, on='row_id', how='left', validate='many_to_one')
    links['report_year'] = pd.to_datetime(links.report_date).dt.year
    obs = pd.concat(observations, ignore_index=True)
    first = {entity: obs.groupby(entity).period_start.min() for entity in ['customer_id', 'address_id']}
    return links, first


def make_features(d, events, first):
    d = d.copy()
    for entity in ['customer_id', 'address_id']:
        d[entity] = d[entity].astype(str)
        counts = pd.Series(0., index=d.index)
        # Reports before January 1 of the prediction year only. All statuses are
        # counted: a later decline outcome is not used to reconstruct past knowledge.
        for year in sorted(d.year.unique()):
            mask = d.year.eq(year)
            past = events.loc[events.report_year < year].groupby(entity).size()
            counts.loc[mask] = d.loc[mask, entity].map(past).fillna(0.)
        d['history_' + entity + '_reported_count'] = counts
        earliest = pd.to_datetime(d[entity].map(first[entity]))
        cutoff = pd.to_datetime(d.year.astype(str) + '-01-01')
        d['history_' + entity + '_observed_years'] = ((cutoff - earliest).dt.days / 365.).clip(lower=0)
    d['coverage_start_month'] = d.period_start.dt.month
    d['coverage_end_month'] = d.period_end.dt.month
    # Seasonal composition: fractions total one, so exposure is applied only once.
    for season, months in {'winter': (1, 2, 12), 'spring': (3, 4, 5),
                           'summer': (6, 7, 8), 'autumn': (9, 10, 11)}.items():
        days = np.zeros(len(d))
        for month in months:
            month_start = pd.to_datetime({'year': d.year, 'month': month, 'day': 1})
            month_end = month_start + pd.offsets.MonthEnd(0)
            start = d.period_start.where(d.period_start > month_start, month_start)
            end = d.period_end.where(d.period_end < month_end, month_end)
            days += ((end - start).dt.days + 1).clip(lower=0).to_numpy()
        d['coverage_' + season + '_fraction'] = days / d.covered_days.to_numpy()
    d['building_age'] = (d.year - d.year_built).clip(lower=0)
    d['observed_customer_tenure'] = (d.year - d.customer_since_year).clip(lower=0)
    d['total_excess'] = d.voluntary_excess + d.compulsory_excess
    if 'sum_insured' in d:
        d['log_sum_insured'] = np.log1p(d.sum_insured.clip(lower=0))
    cols = [c for c in d if c not in EXCLUDE]
    X = d[cols].copy()
    categoricals = []
    for c in X:
        if isinstance(X[c].dtype, pd.CategoricalDtype) or X[c].dtype == object:
            X[c] = X[c].astype('object').fillna('__MISSING__').astype(str)
            categoricals.append(c)
        else:
            X[c] = pd.to_numeric(X[c], errors='coerce').replace([np.inf, -np.inf], np.nan).astype(float)
    return X, categoricals


def cat_model(iterations, seed=314159):
    return CatBoostRegressor(loss_function='Tweedie:variance_power=1.5',
        eval_metric=TotalCostDeviance(), iterations=iterations,
        depth=5, learning_rate=.06, l2_leaf_reg=12, random_seed=seed,
        thread_count=4, boosting_type='Plain', allow_writing_files=False,
        verbose=False)


def cat_fit(X, cats, rate, w, train, valid=None, iterations=240):
    model = cat_model(iterations)
    train_pool = Pool(X.loc[train], label=rate[train], weight=w[train], cat_features=cats)
    kwargs = {}
    if valid is not None:
        kwargs = {'eval_set': Pool(X.loc[valid], label=rate[valid], weight=w[valid], cat_features=cats),
                  'early_stopping_rounds': 35, 'use_best_model': True}
    model.fit(train_pool, **kwargs)
    return model


def predict_cat(model, X):
    return np.maximum(model.predict(X, prediction_type='Exponent'), 1e-10)


def glm_fit(X, cats, rate, w, train):
    # A deliberately small, interpretable model; richer CatBoost is the challenger.
    candidates = ['driver_age', 'vehicle_age', 'annual_mileage', 'vehicle_value_group',
                  'policyholder_age', 'log_sum_insured', 'total_excess', 'building_age',
                  'cover_type', 'region', 'construction_type', 'occupancy',
                  'security_level', 'tenure', 'flood_zone', 'urban_density']
    cols = [c for c in candidates if c in X]
    numeric = [c for c in cols if c not in cats]
    category = [c for c in cols if c in cats]
    pre = ColumnTransformer([
        ('numeric', Pipeline([('fill', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())]), numeric),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=True), category)])
    model = Pipeline([('features', pre), ('model', TweedieRegressor(
        power=1.5, alpha=.2, link='log', max_iter=150, tol=1e-5))])
    model.fit(X.loc[train, cols], rate[train], model__sample_weight=w[train])
    return model, cols


def run(source, data_dir, output, max_iterations):
    output.mkdir(parents=True, exist_ok=True)
    (output / 'models').mkdir(exist_ok=True)
    events, first = history_events(data_dir)
    results, selections, submission_parts = [], {}, {'mature': [], 'all_years': []}
    validation_frames = {2022: [], 2023: [], 2024: []}
    for product in PRODUCTS:
        started = time.time()
        print('Starting models:', product, flush=True)
        d = pd.read_parquet(data_dir / f'assembled_{product}.parquet')
        X, cats = make_features(d, events, first)
        e = d.exposure.to_numpy()
        y = d.target_net_incurred.to_numpy()
        rate = y / e
        w = np.sqrt(e)
        year = d.year.to_numpy()
        candidate_features = {
            'catboost_base': [c for c in X if not c.startswith('history_')],
            'catboost_history': list(X.columns),
        }
        tr, va = year == 2021, year == 2022
        trained_candidates, selection_scores = {}, {}
        for name, cols in candidate_features.items():
            print(product, '2022 selection:', name, flush=True)
            c = [col for col in cats if col in cols]
            model = cat_fit(X[cols], c, rate, w, tr, va, max_iterations)
            pred = e[va] * predict_cat(model, X.loc[va, cols])
            row = dict(product=product, model=name, validation_year=2022,
                       purpose='selection', iterations=model.tree_count_, **score(y[va], pred))
            results.append(row)
            selection_scores[name] = row['mean_deviance']
            trained_candidates[name] = model
            print(json.dumps(row), flush=True)
        # Select only on 2022; later years do not change model choice or iteration count.
        chosen = min(selection_scores, key=selection_scores.get)
        selected_iterations = trained_candidates[chosen].tree_count_
        chosen_cols = candidate_features[chosen]
        chosen_cats = [c for c in cats if c in chosen_cols]
        selections[product] = {'selected': chosen, 'iterations': selected_iterations,
            'features': chosen_cols, 'categorical_features': chosen_cats,
            'excluded_columns': sorted(EXCLUDE),
            'training_weights': 'sqrt(calculated inclusive-day exposure)',
            'final_primary_training_years': [2021, 2022, 2023],
            'all_years_sensitivity_training_years': [2021, 2022, 2023, 2024]}
        for validation_year in (2022, 2023, 2024):
            tr = year < validation_year
            va = year == validation_year
            if validation_year == 2022:
                model = trained_candidates[chosen]
            else:
                print(product, 'held-out prediction:', validation_year, flush=True)
                model = cat_fit(X[chosen_cols], chosen_cats, rate, w, tr,
                                iterations=selected_iterations)
            pred = e[va] * predict_cat(model, X.loc[va, chosen_cols])
            purpose = 'selection' if validation_year == 2022 else ('main_holdout' if validation_year == 2023 else 'immature_diagnostic')
            # Constant annual rate fitted with the same sqrt(e) rate objective.
            constant_rate = np.average(rate[tr], weights=w[tr])
            constant_pred = e[va] * constant_rate
            glm, glm_cols = glm_fit(X, cats, rate, w, tr)
            glm_pred = e[va] * glm.predict(X.loc[va, glm_cols])
            for name, pr in [('selected_catboost', pred), ('constant_rate', constant_pred), ('simple_tweedie_glm', glm_pred)]:
                row = dict(product=product, model=name, validation_year=validation_year,
                           purpose=purpose, **score(y[va], pr))
                results.append(row)
                print(json.dumps(row), flush=True)
            vf = d.loc[va, ['row_id', 'line', 'exposure', 'target_net_incurred']].copy()
            vf['prediction'] = pred
            vf['constant_rate_prediction'] = constant_pred
            vf['glm_prediction'] = glm_pred
            vf['validation_year'] = validation_year
            # Group diagnostics use canonical IDs and strictly earlier training records.
            vf['returning_customer'] = d.loc[va, 'customer_id'].isin(d.loc[tr, 'customer_id']).to_numpy()
            vf['short_coverage'] = d.loc[va, 'exposure'].to_numpy() < .5
            validation_frames[validation_year].append(vf)
            if validation_year == 2024:
                # This model was fit on 2021-23 only. Reuse it for the mature baseline.
                test = year == 2025
                model.save_model(str(output / 'models' / f'{product}_mature.cbm'))
                test_pred = e[test] * predict_cat(model, X.loc[test, chosen_cols])
                submission_parts['mature'].append(pd.DataFrame({'row_id': d.loc[test, 'row_id'].astype(str), 'prediction': test_pred}))
        print(product, 'fitting all-years sensitivity model', flush=True)
        all_model = cat_fit(X[chosen_cols], chosen_cats, rate, w, year <= 2024,
                            iterations=selected_iterations)
        all_model.save_model(str(output / 'models' / f'{product}_all_years.cbm'))
        test = year == 2025
        test_pred = e[test] * predict_cat(all_model, X.loc[test, chosen_cols])
        submission_parts['all_years'].append(pd.DataFrame({'row_id': d.loc[test, 'row_id'].astype(str), 'prediction': test_pred}))
        imp = pd.Series(all_model.feature_importances_, index=chosen_cols).sort_values(ascending=False)
        dump(imp.to_dict(), output / f'feature_importance_{product}.json')
        dump(results, output / 'metrics.json')
        dump(selections, output / 'model_selection.json')
        print(product, 'completed in', round(time.time() - started, 1), 'seconds', flush=True)
        del X, d, model, all_model, trained_candidates
        gc.collect()
    combined, subgroup = [], []
    for year, frames in validation_frames.items():
        vf = pd.concat(frames, ignore_index=True)
        vf.to_parquet(output / f'validation_{year}.parquet', index=False)
        for col in ['prediction', 'constant_rate_prediction', 'glm_prediction']:
            combined.append(dict(validation_year=year, model=col,
                **score(vf.target_net_incurred, vf[col])))
        for product, g in vf.groupby('line', observed=True):
            for variable in ['returning_customer', 'short_coverage']:
                for value, group in g.groupby(variable):
                    if group.target_net_incurred.sum() > 0:
                        subgroup.append(dict(validation_year=year, product=product,
                            variable=variable, value=bool(value), **score(group.target_net_incurred, group.prediction)))
    sample = pd.read_parquet(source / 'sample_submission.parquet')
    aggregates = {}
    for name, parts in submission_parts.items():
        pred = pd.concat(parts, ignore_index=True)
        assert pred.row_id.is_unique and set(pred.row_id) == set(sample.row_id)
        final = sample[['row_id']].merge(pred, on='row_id', how='left', validate='one_to_one')
        assert list(final.columns) == list(sample.columns)
        assert final.row_id.equals(sample.row_id)
        assert final.prediction.notna().all() and np.isfinite(final.prediction).all()
        assert (final.prediction >= 0).all() and len(final) == 626639
        final.to_parquet(output / f'submission_{name}.parquet', index=False)
        aggregates[name] = {'rows': len(final), 'total': float(final.prediction.sum()),
            'mean': float(final.prediction.mean()),
            'by_product': {p: float(frame.prediction.sum()) for p, frame in zip(PRODUCTS, parts)}}
    dump(results, output / 'metrics.json')
    dump(selections, output / 'model_selection.json')
    dump(combined, output / 'combined_metrics.json')
    dump(subgroup, output / 'subgroup_metrics.json')
    dump(aggregates, output / 'submission_checks.json')
    print('Training and submission checks completed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path(__file__).parent / 'data_audit')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'baselines')
    parser.add_argument('--iterations', type=int, default=240)
    args = parser.parse_args()
    run(args.source, args.data, args.output, args.iterations)
