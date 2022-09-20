from __future__ import annotations

import numpy as np
import pandas as pd


def levels_corr(path, unit_level_df, agg_level, weight_col='weight'):
    cell_levels_corr = pd.read_csv(path)
    cell_levels_corr.columns = cell_levels_corr.columns.str.lower()
    unit_level = unit_level_df.index.name
    cell_levels_corr = cell_levels_corr.groupby([agg_level, unit_level]).first().join(unit_level_df[weight_col], how='inner')[weight_col].rename('weight').to_frame()
    cell_levels_corr['w_sum'] = cell_levels_corr['weight'].groupby(agg_level).transform('sum')
    cell_levels_corr['w_normed'] = cell_levels_corr['weight'] / cell_levels_corr['w_sum']
    return cell_levels_corr


def get_agg_metrics(metrics_df, cell_levels_corr, metric_cols: str | list[str] | None = None):
    if metric_cols is None:
        metric_cols = metrics_df.columns.tolist()
    elif isinstance(metric_cols, str):
        metric_cols = [metric_cols]

    stacked_metrics_df = metrics_df[metric_cols].stack().rename('value').to_frame()
    stacked_metrics_df.index = stacked_metrics_df.index.set_names('metric', level=1)
    cell_level = cell_levels_corr.index.names[0]
    groupby_cols = [cell_level, 'metric']
    if stacked_metrics_df.index.names[0] == cell_level:
        # when there's actually no aggregation to make
        stacked_metrics_df = stacked_metrics_df.join(
            cell_levels_corr.groupby(cell_level)['w_sum'].first()
        )
        stacked_metrics_df['w_normed'] = 1
    else:
        stacked_metrics_df = stacked_metrics_df.join(cell_levels_corr)

    grouped = stacked_metrics_df.groupby(groupby_cols)
    stacked_metrics_df = (
        stacked_metrics_df.assign(avg=grouped['value'].transform('mean'))
         .eval("wavg_elem = w_normed * value")
         .eval("wvar_elem = w_normed * (value - avg)**2")
    )
    grouped = stacked_metrics_df.groupby(groupby_cols)
    agg_metrics = grouped.agg(
        wavg=('wavg_elem', 'sum'),
        wvar=('wvar_elem', 'mean'),
        avg=('avg', 'first'),
        var=('value', 'var'),
        min=('value', 'min'),
        max=('value', 'max'),
        weight=('w_sum', 'first'),
        nr_units=('value', 'size')
    )
    agg_metrics['wstd'] = np.sqrt(agg_metrics['wvar'])
    return agg_metrics


def extract_cells_metric(agg_metrics, metric_name):
    return agg_metrics.loc[(slice(None), metric_name), :].droplevel(1)
