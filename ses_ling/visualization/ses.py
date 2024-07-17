import re

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import ses_ling.visualization.utils as viz_utils


def assort_mosaic(assort_dict, nr_cols, figsize, log_scale=False, show_pearson=False, single_cbar=True, class_type='SES', **subplot_kwargs):
    cbar_label = 'log(percentage of trips)' if log_scale else 'percentage of trips'
    mosaic = viz_utils.prep_mosaic_from_dict(assort_dict, nr_cols=nr_cols)
    nr_rows = mosaic.shape[0]
    if single_cbar:
        mosaic = [['supy'] * nr_rows, mosaic] + [['cax'] * nr_rows]
    else:
        mosaic = np.insert(mosaic, np.arange(nr_cols + 1)[1:], 'cax', axis=1).tolist()
        for i in range(nr_rows):
            for j in range(nr_cols):
                mosaic[i][2*j+1] = f'{mosaic[i][2*j+1]}_{mosaic[i][2*j]}'
        mosaic = [['left_supy'] * nr_rows, mosaic, ['right_supy'] * nr_rows]
    mosaic = np.column_stack(mosaic)
    fig, axd = plt.subplot_mosaic(
        mosaic, figsize=figsize,
        layout='compressed', **subplot_kwargs
    )
    axd['left_supy'].set_axis_off()
    axd['right_supy'].set_axis_off()
    assort_mins, assort_maxs = zip(*[
        (city_dict['assort'].values.min(), city_dict['assort'].values.max())
        for city_dict in assort_dict.values()
    ])
    for i, (city, city_dict) in enumerate(assort_dict.items()):
        ax = axd[city]
        pearsonr = city_dict['pearsonr']
        # x axis residence, y axis destination
        assort_plot = city_dict['assort'].T[::-1].copy()
        vmin = min(assort_mins) if single_cbar else assort_mins[i]
        vmax = max(assort_maxs) if single_cbar else assort_maxs[i]
        if log_scale:
            assort_plot = np.log10(assort_plot)
            vmax = np.log10(vmax)
            vmin = np.log10(vmin)
        im = ax.imshow(assort_plot, cmap='Blues', vmin=vmin, vmax=vmax)
        short_city_name = re.split(r'\W+', city)[0]
        pearson_str = f"\n$r_M = {pearsonr:.2f}$"
        ax.set_title(
            short_city_name + show_pearson * pearson_str,
            # usetex=True,
            math_fontfamily='cm',
        )
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
        if not single_cbar:
            # ax_divider = make_axes_locatable(ax)
            # # Add an Axes to the right of the main Axes.
            # cax = ax_divider.append_axes("right", size="5%", pad="4%")
            cax = axd[f'cax_{city}']
            fig.colorbar(im, cax=cax)

    fig.supxlabel(f'{class_type} class of residence', fontsize='large')
    # fig.supylabel('SES class of residence', x=0)
    axd['left_supy'].annotate(
        f'{ class_type } class of visited places', (0, 0.47),
        va='center', rotation='vertical', fontsize='large'
    )
    axd['right_supy'].annotate(cbar_label, (0, 0.47), va='center', rotation='vertical')
    if single_cbar:
        fig.colorbar(im, cax=axd['cax'], label=cbar_label)
    return fig, axd
