import re

import geopandas as geopd
import pandas as pd


def read_df(source_dir_path, fname, index_col, weight_col, score_cols):
    path = source_dir_path / fname
    if path.suffix == '.shp':
        ses_df = geopd.read_file(path)
    else:
        # other options than csv?
        ses_df = pd.read_csv(path)
    ses_df.columns = ses_df.columns.str.lower()
    ses_df = ses_df.set_index(index_col)
    cols_to_keep = {weight_col}
    for patt in score_cols:
        cols_to_keep.update(col for col in ses_df.columns if re.match(patt, col))
    return ses_df[list(cols_to_keep)]
