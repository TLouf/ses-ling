from __future__ import annotations

import re

import geopandas as geopd
import pandas as pd


def normalize_col_names(n):
    if hasattr(n, 'str'):
        return n.str.lower().str.strip().str.replace(' ', '_')
    else:
        return n.lower().strip().replace(' ', '_')

def read_df(
    source_dir_path, fname, index_col: dict | str, score_cols,
    weight_col=None, read_kwargs=None
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
