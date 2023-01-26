from __future__ import annotations

import re

import numpy as np
import geopandas as geopd
import pandas as pd


def normalize_col_names(n):
    if hasattr(n, 'str'):
        return n.str.lower().str.strip().str.replace(' ', '_')
    else:
        return n.lower().strip().replace(' ', '_')

def read_df(
    source_dir_path, fname, index_col: dict | str, score_cols,
    weight_col=None, read_kwargs=None, **kwargs
):
    if read_kwargs is None:
        read_kwargs = {}
    if weight_col is None:
        weight_col = {}

    path = source_dir_path / fname
    if path.suffix == '.shp':
        ses_df = geopd.read_file(path, **read_kwargs)
    else:
        # other options than csv?
        ses_df = pd.read_csv(path, **read_kwargs)

    if isinstance(index_col, dict):
        ses_df = ses_df.rename(columns=index_col)
        index_col = list(index_col.values())
    ses_df = ses_df.set_index(index_col).rename_axis(index=normalize_col_names)

    # weight_col should be kept as first one for convenience later on.
    cols_to_keep = (
        weight_col.copy()
        if isinstance(weight_col, dict)
        else {weight_col: normalize_col_names(weight_col)}
    )

    for col in score_cols:
        if isinstance(col, dict):
            cols_to_keep.update(col)
        else:
            for n in ses_df.columns:
                if re.match(col, n.lower()):
                    cols_to_keep[n] = normalize_col_names(n)

    # ses_df.columns = normalize_col_names(ses_df.columns)
    return ses_df[list(cols_to_keep.keys())].rename(columns=cols_to_keep)


def apply_cells_mask(df, cells_mask=None):
    if cells_mask is not None:
        if df.index.names == ['cell_id'] or df.index.names == [cells_mask.index.name]:
            res_df = df.loc[cells_mask.loc[cells_mask].index]
        else:
            was_series = isinstance(df, pd.Series)
            if was_series:
                df = df.to_frame()
            res_df = df.join(
                cells_mask.loc[cells_mask].rename_axis('cell_id'), on='cell_id', how='inner'
            )
            if was_series:
                res_df = res_df[res_df.columns[0]]
    else:
        res_df = df.copy()

    return res_df

def get_cells_subset_user_res(user_res_cell, cells_mask=None):
    if cells_mask is not None:
        cells_subset_user_res = user_res_cell[['cell_id']].join(
            cells_mask.loc[cells_mask], on='cell_id', how='inner'
        )
    else:
        cells_subset_user_res = user_res_cell.copy()
    return cells_subset_user_res


def get_cells_subset_ses_metric(cells_ses_metric, cells_mask=None):
    if cells_mask is not None:
        cells_subset_ses_metric = cells_ses_metric.loc[cells_mask.loc[cells_mask].index]
    else:
        cells_subset_ses_metric = cells_ses_metric.copy()
    return cells_subset_ses_metric


def get_cells_subset_user_acty(
    user_cell_acty, subreg_res_cell, cells_mask=None, exclude_res_trips=False
):
    subreg_user_cell_acty = (
        user_cell_acty.drop(columns='res_cell_id', errors='ignore')
         .join(subreg_res_cell['cell_id'].rename('res_cell_id'), how='inner')
    )
    if exclude_res_trips:
        res_trips_mask = (
            subreg_user_cell_acty.index.get_level_values('cell_id')
            != subreg_user_cell_acty['res_cell_id']
        )
        subreg_user_cell_acty = subreg_user_cell_acty.loc[res_trips_mask]

    if cells_mask is not None:
        subreg_user_cell_acty = subreg_user_cell_acty.join(
            cells_mask.loc[cells_mask].rename_axis('cell_id'), how='inner'
        )
    subreg_user_cell_acty['prop_user'] = (
        subreg_user_cell_acty['prop_user']
        / subreg_user_cell_acty.groupby('user_id')['prop_user'].transform('sum')
    )
    return subreg_user_cell_acty


def rewire_user_cell_acty(user_cell_acty, user_res_cell, rng, exclude_res_trips=False):
    user_cell_acty.index = user_cell_acty.index.remove_unused_levels()
    user_id_lvl, cell_id_lvl = user_cell_acty.index.levels
    nr_users = user_id_lvl.size
    nr_cells = cell_id_lvl.size

    users_candidate_cells = np.tile(np.arange(nr_cells), (nr_users, 1))
    if exclude_res_trips:
        res_idx = cell_id_lvl.searchsorted(user_res_cell.loc[user_id_lvl, 'cell_id'].values)
        # np.put_along_axis(users_candidate_cells, res_idx[:, np.newaxis], -1, 1)
        # users_candidate_cells = np.sort(users_candidate_cells, axis=1)[:, 1:]
        mask = np.ones_like(users_candidate_cells, dtype=bool)
        mask[np.arange(nr_users), res_idx] = False
        users_candidate_cells = users_candidate_cells[mask].reshape((nr_users, nr_cells - 1))

    draw = rng.integers(0, cell_id_lvl.size - 1, user_cell_acty.shape[0])
    i = user_cell_acty.index.codes[0]
    j = draw
    random_user_acty = user_cell_acty.copy()
    random_user_acty.index = random_user_acty.index.set_codes(users_candidate_cells[i, j], level='cell_id')

    # TOREMOVE
    if exclude_res_trips:
        assert np.all(random_user_acty.index.get_level_values('cell_id') != random_user_acty['res_cell_id'])
    
    return random_user_acty


def attr_user_to_class(user_res_cell, cells_ses_metric, nr_classes):
    user_rank = (
        user_res_cell.join(cells_ses_metric, on='cell_id')[cells_ses_metric.name]
         .rank(axis=0, method='average')
    )
    user_class = np.ceil(nr_classes * user_rank / user_rank.index.size).astype(int)
    nr_users_per_class = user_class.value_counts()
    print(
        f"{nr_users_per_class.mean():.1f} users on average in each of the "
        f"{nr_users_per_class.size} classes."
    )
    return user_class


def attr_cell_to_class(
    cells_ses_metric: pd.Series,
    nr_classes: int,
    nr_residents_per_cell: pd.Series | None = None,
):
    if nr_residents_per_cell is None:
        cell_ranking = cells_ses_metric.rank(axis=0, method='average')
        cells_class = np.ceil(
            nr_classes * cell_ranking / cell_ranking.index.size
        )
    else:
        cell_ranking = (
            nr_residents_per_cell.to_frame()
             .join(cells_ses_metric, how='inner')
             .sort_values(by=cells_ses_metric.name)
        )
        cell_ranking['cumcount'] = cell_ranking[nr_residents_per_cell.name].cumsum()
        cells_class = 1 + (
            (cell_ranking['cumcount'] - cell_ranking['cumcount'].iloc[0])
            / (cell_ranking['cumcount'].iloc[-1] / nr_classes)
        )

    cells_class = cells_class.astype(int).rename('ses_class')
    nr_cells_per_class = cells_class.value_counts()
    print(
        f"{nr_cells_per_class.mean():.1f} cells on average in each of the "
        f"{nr_cells_per_class.size} classes."
    )
    return cells_class


def inter_cell_to_inter_class_od(
    intercell_od: pd.DataFrame, cells_class: pd.Series, weight_col='count'
):
    # assumed in `from, to` order
    from_name = intercell_od.index.names[0]
    to_name = intercell_od.index.names[1]
    interclass_od = (
        intercell_od
         .join(
            cells_class.rename_axis(from_name).rename(f'from_{cells_class.name}'),
            how='inner',
         )
         .join(
             cells_class.rename_axis(to_name).rename(f'to_{cells_class.name}'),
             how='inner',
         )
         .groupby([f'from_{cells_class.name}', f'to_{cells_class.name}'])[weight_col]
         .sum()
    )
    return interclass_od


def interclass_od_to_assort(interclass_od, normalize_incoming=True):
    # assumed in `from, to` order
    from_level = interclass_od.index.names[int(normalize_incoming)]
    assort_df = (
        interclass_od
        / interclass_od.groupby(from_level).transform('sum')
    ).unstack().fillna(0)
    return assort_df


def user_acty_to_assort(
    user_cell_acty,
    user_residence_cell,
    cells_ses_metric,
    nr_classes,
    cells_mask=None,
    exclude_res_trips=False,
    normalize_incoming=False,
):
    od_groupby_col = 'cell_id' if normalize_incoming else 'res_cell_id'
    cells_subset_user_res = get_cells_subset_user_res(
        user_residence_cell, cells_mask=cells_mask
    )
    cells_subset_ses_metric = get_cells_subset_ses_metric(cells_ses_metric, cells_mask)
    user_class = attr_user_to_class(
        cells_subset_user_res, cells_subset_ses_metric, nr_classes
    )
    cells_class = (
        cells_subset_user_res.join(user_class)
         .groupby('cell_id')[cells_ses_metric.name]
         .first()
    )
    cells_subset_user_acty = get_cells_subset_user_acty(
        user_cell_acty, cells_subset_user_res,
        cells_mask=cells_mask, exclude_res_trips=exclude_res_trips
    )

    od_df = (
        cells_subset_user_acty.groupby(['res_cell_id', 'cell_id'])[['prop_user']].sum()
    )
    od_df = od_df / od_df.groupby(od_groupby_col).transform('sum')

    interclass_od = inter_cell_to_inter_class_od(
        od_df,
        cells_class,
        weight_col='prop_user',
    )
    assort = interclass_od_to_assort(
        interclass_od, normalize_incoming=normalize_incoming
    )
    return assort
