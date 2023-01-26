import numpy as np
import matplotlib.pyplot as plt

import ses_ling.visualization.utils as viz_utils


def assort_mosaic(assort_dict, nr_cols, figsize, log_scale=False):
    mosaic = viz_utils.prep_mosaic_from_dict(assort_dict, nr_cols=2)
    fig, axd = plt.subplot_mosaic(
        mosaic, sharex=True, sharey=True, figsize=figsize,
        constrained_layout=True,
    )
    assort_mins, assort_maxs = zip(*[
        (city_dict['assort'].values.min(), city_dict['assort'].values.max()) 
        for city_dict in assort_dict.values()
    ])
    for i, (city, city_dict) in enumerate(assort_dict.items()):
        ax = axd[city]
        cbar = i == 0
        pearsonr = city_dict['pearsonr']
        assort_plot = city_dict['assort'].T[::-1].copy()
        vmin = min(assort_mins)
        vmax = min(assort_maxs)
        if log_scale:
            assort_plot = np.log10(assort_plot)
            vmax = np.log10(vmax)
            vmin = np.log10(vmin)
        im = ax.imshow(assort_plot, cmap='Blues', vmin=vmin, vmax=vmax)
        ax.set_title(f"{city}, r = {pearsonr:.2f}")

    nr_classes = assort_plot.shape[0]
    ax.set_xticks(np.arange(nr_classes), assort_plot.columns.astype(str))
    ax.set_yticks(np.arange(nr_classes), assort_plot.index.astype(str))
    fig.supxlabel('SES class of residence')
    fig.supylabel('SES class of visited places')
    cbar_label = 'log(probability)' if log_scale else 'probability'
    fig.colorbar(im, ax=list(axd.values()), shrink=0.6, label=cbar_label)
    return fig, axd
