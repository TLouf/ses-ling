import pandas as pd


def empy_multiidx(names):
    empty_vals = [[]] * len(names)
    empty_idx = pd.MultiIndex(codes=empty_vals, levels=empty_vals, names=names)
    return empty_idx
