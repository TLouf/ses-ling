from __future__ import annotations

import re

import geopandas as geopd
import numpy as np
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


def apply_masks(df, masks_dict=None):
    if masks_dict is None:
        masks_dict = {}
    res_df = df.copy()
    was_series = isinstance(df, pd.Series)
    iterator = [
        (join_on, mask) for join_on, mask in masks_dict.items() if mask is not None
    ]
    for join_on, mask in iterator:
        # or just join anyway?? avoid index size issues
        if df.index.names == [join_on] or df.index.names == [mask.index.name]:
            res_df = res_df.loc[mask.loc[mask].index]
        else:
            if was_series:
                res_df = res_df.to_frame()
            # try: ... except ValueError (no possible join key): pass??
            res_df = res_df.join(
                mask.loc[mask].rename_axis(join_on), on=join_on, how='inner'
            ).drop(columns=mask.name)
            if was_series:
                res_df = res_df[res_df.columns[0]]

    return res_df


def exclude_res_trips_from_acty(raw_user_cell_acty, user_res_cell):
    user_cell_acty = raw_user_cell_acty.copy()
    if 'res_cell_id' not in raw_user_cell_acty.columns:
        user_cell_acty = (
            user_cell_acty.join(user_res_cell['cell_id'].rename('res_cell_id'), how='inner')
        )
    res_trips_mask = (
        user_cell_acty.index.get_level_values('cell_id')
        != user_cell_acty['res_cell_id']
    )
    user_cell_acty = user_cell_acty.loc[res_trips_mask]

    og_nr_users = raw_user_cell_acty.index.levels[0].size
    prop_users_removed = (
        og_nr_users - user_cell_acty.index.get_level_values(0).unique().size
    ) / og_nr_users
    og_nr_trips = raw_user_cell_acty['count'].sum()
    prop_trips_removed = (og_nr_trips - user_cell_acty['count'].sum()) / og_nr_trips
    print(
        f"{prop_users_removed:.1%} of users and {prop_trips_removed:.1%} of trips"
        " removed from exclusion of residence trips"
    )
    return user_cell_acty

def recompute_user_cell_acty(user_cell_acty):
    user_cell_acty['prop_cell'] = (
        user_cell_acty['prop_cell']
        / user_cell_acty.groupby('user_id')['prop_cell'].transform('sum')
    )
    return user_cell_acty

def preprocess_cell_acty(user_cell_acty, user_res_cell, masks_dict, exclude_res_trips=False):
    # only keep trips of residents
    subset_user_acty = (
        user_cell_acty.drop(columns='res_cell_id', errors='ignore')
         .join(user_res_cell['cell_id'].rename('res_cell_id'), how='inner')
    )
    # only keep destination cells within potential subregion defined in masks_dict
    subset_user_acty = apply_masks(subset_user_acty, masks_dict)
    if exclude_res_trips:
        subset_user_acty = exclude_res_trips_from_acty(
            subset_user_acty, user_res_cell
        )
    subset_user_acty = recompute_user_cell_acty(subset_user_acty)
    return subset_user_acty

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
    # classes defined based on users that passed all previous filters TODO: is this
    # correct? or should rather take real world pop of cells, attribute them a class,
    # and then to corresponding users?
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
    return user_class.rename('user_class')


def cell_class_from_residents(user_res, user_class):
    # Assumes that all residents of a cell have same class, true here since only SES
    # data is at cell level. Not optimal but useful when user_class needs to be computed
    # anyway.
    # TODO: is this correct? or should rather take real world pop?
    cells_class = (
        user_res.join(user_class)
         .groupby('cell_id')[user_class.name]
         .first()
         .rename('cell_class')
    )
    return cells_class


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

    cells_class = cells_class.astype(int).rename('cell_class').rename_axis('cell_id')
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
    user_res_cell,
    cells_ses_metric,
    nr_classes,
    cells_pop=None,
    cells_mask=None,
    users_mask=None,
    exclude_res_trips=False,
    normalize_incoming=False,
):
    masks_dict = {'cell_id': cells_mask}
    cells_subset_ses_metric = apply_masks(cells_ses_metric, masks_dict)
    od_df = user_acty_to_od(
        user_cell_acty,
        user_res_cell,
        cells_mask=cells_mask,
        users_mask=users_mask,
        exclude_res_trips=exclude_res_trips,
        normalize_incoming=normalize_incoming,
    )

    masks_dict['user_id'] = users_mask
    cells_subset_user_res = apply_masks(user_res_cell, masks_dict)
    if cells_pop is None:
        user_class = attr_user_to_class(
            cells_subset_user_res, cells_subset_ses_metric, nr_classes
        )
        cells_class = cell_class_from_residents(cells_subset_user_res, user_class)
    else:
        cells_class = attr_cell_to_class(
            cells_subset_ses_metric, nr_classes, cells_pop
        )
        user_class = cells_subset_user_res.join(cells_class, on='cell_id')[cells_class.name]

    assort = inter_cell_od_to_assort(
        od_df, cells_class, normalize_incoming=normalize_incoming
    )
    return assort


def user_acty_to_od(
    user_cell_acty,
    user_res_cell,
    cells_mask=None,
    users_mask=None,
    exclude_res_trips=False,
    normalize_incoming=False,
):
    masks_dict = {'cell_id': cells_mask, 'user_id': users_mask}
    cells_subset_user_res = apply_masks(user_res_cell, masks_dict)
    cells_subset_user_acty = preprocess_cell_acty(
        user_cell_acty, cells_subset_user_res, masks_dict, exclude_res_trips
    )
    od_df = (
        cells_subset_user_acty.groupby(['res_cell_id', 'cell_id'])[['prop_cell']].sum()
    )
    od_groupby_col = 'cell_id' if normalize_incoming else 'res_cell_id'
    od_df = od_df / od_df.groupby(od_groupby_col).transform('sum')
    return od_df


def inter_cell_od_to_assort(od_df, cells_class, normalize_incoming=False):
    interclass_od = inter_cell_to_inter_class_od(
        od_df,
        cells_class,
        weight_col='prop_cell',
    )
    # Here important normalization: if groupby 'cell_id' (ie normalize_incoming), you
    # get proba that a visitor to a dest cell comes from an origin one, else if groupby
    # 'res_cell_id', you get proba that a guy residing in origin cell goes to dest one.
    # TODO: think which makes more sense, Fika does normalize_incoming = False
    assort = interclass_od_to_assort(
        interclass_od, normalize_incoming=normalize_incoming
    )
    return assort

def get_class_user_acty(user_cell_acty):
    class_user_acty = (
        user_cell_acty.groupby(['user_id', 'user_class', 'cell_class'])['prop_cell']
        .sum()
        .rename('prop_cell_class')
    )
    return class_user_acty


def get_user_visited_cells_entropy(class_user_acty, nr_classes):
    prop = class_user_acty#['prop_cell_class']
    user_visited_cells_entropy = (
        -1 / nr_classes
        * (prop * np.log(prop)).groupby('user_id').sum()
    )
    return user_visited_cells_entropy.rename('visited_entropy')


def get_cell_residents_entropy(user_res_cell, user_visited_cells_entropy):
    cell_residents_entropy = (
        user_res_cell.join(user_visited_cells_entropy, how='inner')
         .groupby('cell_id')['visited_entropy']
         .mean()
         .rename('residents_entropy')
    )
    return cell_residents_entropy


def get_cell_inc_class(user_cell_acty):
    cell_inc_class = (
        user_cell_acty.groupby(['cell_id', 'user_class'])['prop_cell']
        .sum()
        .rename('prop_user_class')
    )
    cell_inc_class = cell_inc_class / cell_inc_class.groupby('cell_id').transform('sum')
    return cell_inc_class

def get_cell_visitors_entropy(cell_inc_class, nr_classes):
    prop = cell_inc_class#['prop_user_class']
    cell_visitors_entropy = (
        -1 / nr_classes
        * (prop * np.log(prop)).groupby('cell_id').sum()
    )
    return cell_visitors_entropy.rename('visitors_entropy')


def get_cell_entropies_df(
    user_cell_acty, user_res_cell, cells_ses_metric, nr_classes, cells_pop=None,
    cells_mask=None, users_mask=None, exclude_res_trips=False
):
    masks_dict = {'cell_id': cells_mask}
    cells_subset_ses_metric = apply_masks(cells_ses_metric, masks_dict)
    masks_dict['user_id'] = users_mask
    cells_subset_user_res = apply_masks(user_res_cell, masks_dict)
    if cells_pop is None:
        user_class = attr_user_to_class(
            cells_subset_user_res, cells_subset_ses_metric, nr_classes
        )
        cells_class = cell_class_from_residents(cells_subset_user_res, user_class)
    else:
        cells_class = attr_cell_to_class(
            cells_subset_ses_metric, nr_classes, cells_pop
        )
        user_class = (
            cells_subset_user_res.join(cells_class, on='cell_id')[cells_class.name]
             .rename('user_class')
        )

    cells_subset_user_acty = preprocess_cell_acty(
        user_cell_acty, cells_subset_user_res, masks_dict, exclude_res_trips
    )
    cells_subset_user_acty = cells_subset_user_acty.join(cells_class).join(user_class)
    class_user_acty = get_class_user_acty(cells_subset_user_acty)

    user_visited_cells_entropy = get_user_visited_cells_entropy(
        class_user_acty, nr_classes
    )
    cell_residents_entropy = get_cell_residents_entropy(
        cells_subset_user_res, user_visited_cells_entropy
    )

    cell_inc_class = get_cell_inc_class(cells_subset_user_acty)
    cell_visitors_entropy = get_cell_visitors_entropy(cell_inc_class, nr_classes)

    return pd.concat([cell_residents_entropy, cell_visitors_entropy], axis=1)
