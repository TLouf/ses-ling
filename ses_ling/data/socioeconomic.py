from __future__ import annotations

import re

import geopandas as geopd
import pandas as pd


def read_df(
    source_dir_path, fname, index_col: dict | str, score_cols,
    weight_col=None, weight_from=None, read_kwargs=None
):
    if read_kwargs is None:
        read_kwargs = {}

    path = source_dir_path / fname
    if path.suffix == '.shp':
        ses_df = geopd.read_file(path, **read_kwargs)
    else:
        # other options than csv?
        ses_df = pd.read_csv(path, **read_kwargs)
    ses_df.columns = ses_df.columns.str.lower()
    if isinstance(index_col, dict):
        ses_df = ses_df.rename(columns=index_col)
        index_col = list(index_col.values())[0]
    ses_df = ses_df.set_index(index_col)
    cols_to_keep = {weight_col} if weight_col is not None else set()
    for patt in score_cols:
        cols_to_keep.update(col for col in ses_df.columns if re.match(patt, col))
    return ses_df[list(cols_to_keep)]
