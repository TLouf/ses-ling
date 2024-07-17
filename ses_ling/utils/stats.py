import numpy as np
import pandas as pd


def matrix_pearsonr(mat):
    mat = mat / mat.sum()
    j = np.arange(mat.shape[0])
    wj = j * mat
    wi = (j * mat.T).T
    wjj = (j * wj).sum()
    wii = ((j * wi.T).T).sum()
    wij = (j * wi).sum() # or = ((j * wj.T).T).sum()
    wi = wi.sum()
    wj = wj.sum()
    pearsonr = (wij - wi * wj) / np.sqrt((wii - wi**2) * (wjj - wj**2))
    return pearsonr


def mask_outliers(s: pd.Series, iqr_mult: float = 1.5) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75]).values
    iqr = q3 - q1
    lower = q1 - iqr_mult * iqr
    upper = q3 + iqr_mult * iqr
    mask = (s >= lower) & (s <= upper)
    return mask

def pearson_std(r, N_samples):
    return (1 - r**2) / np.sqrt(N_samples - 3)
