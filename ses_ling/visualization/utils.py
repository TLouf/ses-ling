import numpy as np

def prep_mosaic_from_dict(d, nr_cols):
    mosaic = list(d.keys())
    nr_empty = nr_cols - len(mosaic) % nr_cols
    nr_rows = len(mosaic) // nr_cols + int(nr_empty > 0)
    mosaic.extend(['.'] * nr_empty)
    mosaic = np.vstack([
        np.array(mosaic).reshape(nr_rows, nr_cols),
        # np.array(['cax'] * nr_cols)
    ])[::-1, :]
    return mosaic
