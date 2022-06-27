import os
from pathlib import Path
import subprocess
import IPython.display
import numpy as np
import plotly.graph_objects as go
import plotly.colors
import plotly.offline
from shapely.geometry import MultiPoint

import ses_ling.utils.geometry as geo_utils


def calc_fit_zoom(gdf):
    max_bound = max(geo_utils.calc_shape_dims(gdf))
    zoom = 11.5 - np.log(max_bound)
    return zoom


def plot_interactive(fig, mapbox_style='stamen-toner', mapbox_zoom=10,
                     plotly_renderer='iframe_connected',
                     save_path=None, show=False):
    '''
    Utility to plot an interactive map, given the plot data and layout given in
    the plotly Figure instance `fig`.
    '''
    fig.update_layout(
        mapbox_style=mapbox_style, mapbox_zoom=mapbox_zoom,
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )
    use_iframe_renderer = plotly_renderer.startswith('iframe')
    if save_path:
        # Path objects are not yet supported by plotly, so first cast to str.
        plotly.offline.plot(fig, filename=str(save_path), auto_open=False)
    if show:
        # With the 'iframe' renderer, a standalone HTML is created in a new
        # folder iframe_figures/, and the files are named according to the cell
        # number. Thus, many files can be created while testing out this
        # function, so to avoid this we simply use the previously saved HTML
        # file (so use_iframe_renderer should imply save_path), which has a
        # name we chose.
        if use_iframe_renderer:
            pwd_path = Path(os.environ['PWD'])
            rel_path = save_path.resolve().relative_to(pwd_path)
            # Print the location of the file to have a clickable hyperlink to
            # open it in a new tab from a notebook.
            print(get_jpt_address() + 'files/' + str(rel_path))
            IPython.display.display(IPython.display.IFrame(
                src=save_path, width=900, height=600))
        else:
            fig.show(renderer=plotly_renderer, width=900, height=600,
                     config={'modeBarButtonsToAdd': ['zoomInMapbox',
                                                     'zoomOutMapbox']})

    return fig


def cells(gdf, metric_col, colorscale='Plasma', latlon_proj='epsg:4326',
          alpha=0.8, **plot_interactive_kwargs):
    '''
    Plots an interactive Choropleth map with Plotly.
    The map layer on top of which this data is shown is provided by mapbox (see
    https://plot.ly/python/mapbox-layers/#base-maps-in-layoutmapboxstyle for
    possible values of 'mapbox_style').
    Plotly proposes different renderers, described at:
    https://plot.ly/python/renderers/#the-builtin-renderers.
    The geometry column of gdf must contain only valid geometries:
    just one null value will prevent the choropleth from being plotted.
    '''
    latlon_gdf = gdf['geometry'].to_crs(latlon_proj)
    plot_interactive_kwargs.setdefault('mapbox_zoom', calc_fit_zoom(latlon_gdf))

    start_point = MultiPoint(gdf.centroid.values).centroid
    layout = go.Layout(
        mapbox_center={"lat": start_point.y, "lon": start_point.x}
    )
    
    # Get a dictionary corresponding to the geojson (because even though the
    # argument is called geojson, it requires a dict type, not a str). The
    # geometry must be in lat, lon.
    geo_dict = latlon_gdf.geometry.__geo_interface__
    choropleth_dict = dict(
        geojson=geo_dict,
        locations=gdf.index.values,
        # hoverinfo='skip',
        colorscale=colorscale,
        marker_opacity=alpha,
        marker_line_width=0.1,
        z=gdf[metric_col],
        visible=True,
    )

    data = [go.Choroplethmapbox(**choropleth_dict)]

    fig = go.Figure(data=data, layout=layout)
    fig = plot_interactive(fig, **plot_interactive_kwargs)
    return fig


def get_jpt_address():
    '''
    Get the currently running Jupyter's server address, or at least the first
    one found, if any.
    '''
    # from jupyter_server import serverapp
    # serverapp.list_running_servers()
    env_path = os.environ.get('VIRTUAL_ENV') or os.environ.get('CONDA_PREFIX')
    serv_list = subprocess.run([env_path + '/bin/jupyter', 'server', 'list'],
                               capture_output=True, check=True)
    all_lines = serv_list.stdout.decode().split('\n')
    for line in all_lines[1:]:
        address, jpt_wd = line.split(' :: ')
        if jpt_wd == os.environ['PWD']:
            return address

    return ''
