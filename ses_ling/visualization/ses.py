import matplotlib.pyplot as plt
import numpy as np

import ses_ling.visualization.utils as viz_utils


def assort_mosaic(assort_dict, nr_cols, figsize, log_scale=False, **subplot_kwargs):
    mosaic = viz_utils.prep_mosaic_from_dict(assort_dict, nr_cols=nr_cols)
    nr_rows = mosaic.shape[0]
    mosaic = np.column_stack((['supy'] * nr_rows, mosaic, ['cax'] * nr_rows))
    fig, axd = plt.subplot_mosaic(
        mosaic, figsize=figsize,
        constrained_layout=True, **subplot_kwargs
    )
    axd['supy'].set_axis_off()
    assort_mins, assort_maxs = zip(*[
        (city_dict['assort'].values.min(), city_dict['assort'].values.max()) 
        for city_dict in assort_dict.values()
    ])
    for i, (city, city_dict) in enumerate(assort_dict.items()):
        ax = axd[city]
        assort_plot = city_dict['assort'].T[::-1].copy()
        vmin = min(assort_mins)
        vmax = min(assort_maxs)
        if log_scale:
            assort_plot = np.log10(assort_plot)
            vmax = np.log10(vmax)
            vmin = np.log10(vmin)
        im = ax.imshow(assort_plot, cmap='Blues', vmin=vmin, vmax=vmax)
        ax.set_title(f"{city}")#, r = {pearsonr:.2f}")
        spec = ax.get_subplotspec()
        if spec.colspan.start >= 2:
            ax.sharey(axd[mosaic[spec.rowspan.start, 1]])
            ax.tick_params(labelleft=False)
        if spec.rowspan.start < nr_rows - 1:
            ax.sharex(axd[mosaic[-1, spec.colspan.start]])
            ax.tick_params(labelbottom=False)

        nr_classes = assort_plot.shape[0]
        ax.set_xticks(np.arange(nr_classes), assort_plot.columns.astype(str))
        ax.set_yticks(np.arange(nr_classes), assort_plot.index.astype(str))
    fig.supxlabel('SES class of residence')
    fig.supylabel('SES class of visited places', x=0)
    cbar_label = 'log(proportion of trips)' if log_scale else 'proportion of trips'
    fig.colorbar(im, cax=axd['cax'], label=cbar_label)
    return fig, axd
