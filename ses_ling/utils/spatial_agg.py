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


def get_agg_metrics(metrics_df, cell_levels_corr, metric_col='score'):
    agg_metrics = metrics_df.join(cell_levels_corr)
    grouper = agg_metrics.groupby(cell_levels_corr.index.names[0])
    agg_metrics['wavg_elem'] = agg_metrics[metric_col] * agg_metrics['w_normed']
    agg_metrics['avg'] = grouper[metric_col].transform('mean')
    agg_metrics['wvar_elem'] = (
        (agg_metrics[metric_col] - agg_metrics['avg'])**2
        * agg_metrics['w_normed']
    )
    agg_metrics = grouper.agg(
        wavg=('wavg_elem', 'sum'),
        wvar=('wvar_elem', 'mean'),
        avg=('avg', 'first'),
        var=(metric_col, 'var'),
        min=(metric_col, 'min'),
        max=(metric_col, 'max'),
        weight=('w_sum', 'first'),
        nr_units=(metric_col, 'size')
    )
    agg_metrics['wstd'] = np.sqrt(agg_metrics['wvar'])
    return agg_metrics
