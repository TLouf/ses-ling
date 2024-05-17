from __future__ import annotations

import datetime
import logging
from pathlib import Path

import geopandas as geopd
import pandas as pd
import querier as qr

import ses_ling.data.access as data_access
import ses_ling.utils.pandas as pd_utils
import ses_ling.utils.spatial_agg as spatial_agg

LOGGER = logging.getLogger(__name__)


def visited_places_in_year(
    year: int, colls, resident_ids, timezone='UTC', pre_filter: qr.Filter | None = None,
    hour_start_day=8, hour_end_day=18
) -> pd.DataFrame:
    db = f'twitter_{year}'
    f = (
        data_access.base_tweets_filter(year)
        & qr.Filter().not_exists('coordinates.coordinates')
    )
    if pre_filter is not None:
        f = f & pre_filter

    with qr.Connection(db) as con:
        pipeline = [
            {'$match': f},
            {
                "$addFields": {
                    'hour': {
                        '$hour': {
                            'date': '$created_at',
                            'timezone': timezone,
                        }
                    },
                }
            },
            {
                "$addFields": {
                    'is_daytime': {
                        '$and': [
                            {'$gte': ["$hour", hour_start_day]},
                            {'$lt': ["$hour", hour_end_day]},
                        ]
                    },
                }
            },
            {
                "$group": {
                    "_id": {
                        'user_id': "$user.id",
                        'place_id': '$place.id',
                        'is_daytime': '$is_daytime'
                    },
                    "count": {"$sum": 1},
                }
            },
        ]

        res = con[colls].aggregate(pipeline, allowDiskUse=True)
        visited_places = data_access.groupby_res_to_df(res, pipeline)

    residents_series = pd.Series(
        True, index=pd.Index(resident_ids, name='user_id'), name='is_resident'
    )
    visited_places = (
        visited_places.join(residents_series, how='inner')
         .drop(columns=['is_resident'])
    )
    return visited_places


def agg_visited_places(
    year_from, year_to, colls, resident_ids, timezone='UTC', pre_filter=None,
    hour_start_day=8, hour_end_day=18
) -> pd.DataFrame:
    visited_places = pd.DataFrame()
    for year in range(year_from, year_to + 1):
        LOGGER.info(f' Starting on {year}')
        visited_places = visited_places.add(
            visited_places_in_year(
                year, colls, resident_ids, timezone=timezone, pre_filter=pre_filter,
                hour_start_day=hour_start_day, hour_end_day=hour_end_day
            ),
            fill_value=0,
        )
        nr_users, nr_places, _ = visited_places.index.levshape
        LOGGER.info(f'Have info on {nr_users} users in {nr_places} places.')

    return visited_places


def yield_yearly_poi_counts(year_from, year_to, colls, save_path_fmt):
    for year in range(year_from, year_to):
        save_path = str(save_path_fmt).format(year=year)
        if Path(save_path).exists():
            yield pd.read_parquet(save_path)
        else:
            LOGGER.info(f"No POI duplicates counts at {save_path}, computing it...")
            db = f'twitter_{year}'
            f = qr.Filter().not_exists('coordinates.coordinates').equals('place.place_type', 'poi')
            with qr.Connection(db) as con:
                grpby = con[colls].groupby('place.id', pre_filter=f, allowDiskUse=True)
                poi_counts = data_access.mongo_groupby_to_df(
                    grpby, count=qr.NamedAgg('id', 'count')
                )
            poi_counts.to_parquet(save_path)
            yield poi_counts


def get_gps_duplicate_counts(year_from, year_to, colls, pre_filter, save_path):
    if Path(save_path).exists():
        return pd.read_parquet(save_path)

    LOGGER.info(f"No GPS duplicates counts at {save_path}, computing it...")
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)

    pre_filter = data_access.gps_filter(pre_filter)
    # Add post filter with small threshold on duplicate counts to eliminate most GPS
    # coordinates, as most of them are unique of repeated a few times only.
    post_filter = qr.Filter().greater_or_equals('nr_tweets', 5)
    dups_counts = pd.DataFrame()

    for y in range(year_from, year_to + 1):
        db = f'twitter_{y}'
        with qr.Connection(db) as con:
            grpby = con[colls].groupby(
                'coordinates.coordinates', allowDiskUse=True,
                pre_filter=pre_filter, post_filter=post_filter
            )
            res_df = data_access.mongo_groupby_to_df(grpby, nr_tweets=("id", "count")).reset_index()
            coords_col = 'coordinates_coordinates'
            res_df = pd.concat(
                [
                    res_df,
                    pd.DataFrame(res_df[coords_col].tolist(), columns=['lat', 'lon'])
                ],
                axis=1,
            )
            res_df.index = res_df['lat'].astype(str) + res_df['lon'].astype(str)
            res_df, dups_counts = res_df.align(dups_counts)
            dups_counts['nr_tweets'] = dups_counts['nr_tweets'].add(res_df['nr_tweets'], fill_value=0)
            dups_counts[coords_col] = dups_counts[coords_col].combine_first(res_df[coords_col])

    dups_counts.to_parquet(save_path)
    return dups_counts


def filter_gps_duplicates(
    year_from, year_to, colls, dups_counts_path, pre_filter=None, th=100
):
    if pre_filter is None:
        pre_filter = qr.Filter()

    if th < 0:
        # specify -1 to mean no threshold
        return pre_filter

    dups_counts = get_gps_duplicate_counts(
        year_from, year_to, colls, pre_filter, dups_counts_path
    )
    coords_col = 'coordinates_coordinates'
    mask = dups_counts['nr_tweets'] >= th
    gps_above_th = [list(arr) for arr in dups_counts.loc[mask, coords_col].tolist()]
    LOGGER.info(f"{len(gps_above_th)} GPS coordinates have more than {th} duplicates.")
    gps_filter = qr.Filter().none_of('coordinates.coordinates', gps_above_th)
    return gps_filter


def points_to_cells_in_year(
    year, colls, resident_ids, latlon_cells_gdf, pre_filter=None, timezone='UTC',
    hour_start_day=8, hour_end_day=18
) -> pd.DataFrame:
    if pre_filter is None:
        # Here no need for base filter as chunk filters are based on date interval.
        pre_filter = qr.Filter()
    db = f'twitter_{year}'
    start = datetime.datetime(year, 1, 1)
    end = datetime.datetime(year + 1, 1, 1)
    residents_series = pd.Series(True, index=resident_ids, name='is_resident')
    pre_filter = data_access.gps_filter(pre_filter.copy())
    chunk_filters = data_access.dt_chunk_filters_mongo(
        db, colls, pre_filter, start, end, chunksize=20e6
    )
    count_by = ['user_id', latlon_cells_gdf.index.name, 'is_daytime']
    cell_counts = pd.DataFrame(index=pd_utils.empy_multiidx(count_by))

    for i, chunk_filter in enumerate(chunk_filters):
        tweets = data_access.tweets_from_mongo(
            db, chunk_filter, colls,
            cols=['user_id', 'created_at', 'coordinates'],
            coords_as='xy'
        )
        tweets = tweets.join(residents_series, on='user_id', how='inner')
        tweets['created_at'] = tweets['created_at'].dt.tz_localize('UTC').dt.tz_convert(timezone)
        hour = tweets['created_at'].dt.hour
        tweets['is_daytime'] = (hour >= hour_start_day) & (hour < hour_end_day)
        geo = geopd.points_from_xy(tweets['x'], tweets['y'], crs='epsg:4326')
        tweets = geopd.GeoDataFrame(tweets[['user_id', 'is_daytime']], geometry=geo)
        tweets = tweets.sjoin(
            latlon_cells_gdf[['geometry']], how='inner', predicate='intersects'
        )
        tweets = tweets.rename(columns={'index_right': latlon_cells_gdf.index.name})
        cell_counts = cell_counts.add(
            tweets.groupby(count_by).size().rename('count').to_frame(),
            fill_value=0,
        )

    return cell_counts


def agg_points_to_cells(
    year_from, year_to, colls, resident_ids, cells_gdf, pre_filter=None, timezone='UTC',
    hour_start_day=8, hour_end_day=18
) -> pd.DataFrame:
    # frame becaue eg `to_parquet` not implemented for series...
    latlon_cells_gdf = cells_gdf.to_crs('epsg:4326')
    count_by = ['user_id', latlon_cells_gdf.index.name, 'is_daytime']
    cell_counts = pd.DataFrame(index=pd_utils.empy_multiidx(count_by))

    for year in range(year_from, year_to + 1):
        cell_counts = cell_counts.add(
            points_to_cells_in_year(
                year, colls, resident_ids, latlon_cells_gdf,
                pre_filter=pre_filter, timezone=timezone,
                hour_start_day=hour_start_day, hour_end_day=hour_end_day
            ),
            fill_value=0,
        )

    return cell_counts.astype(int)


def places_to_cells(
    places_gdf, cells_gdf, xy_proj='epsg:3857', ntop=2, top_cells_cumoverlap_th=0.9, reg_overlap_th=0.6
):
    '''
    Places which don't have at least a ratio of their area superior to `reg_overlap_th`
    overlapping with the region under consideration are discarded. The `ntop` cells with
    most overlap with a given place must overlap at least a ratio
    `top_cells_cumoverlap_th` of the place. If so, the place is kept with its ratios of
    overlap with the `ntop` cells, otherwise the place is discarded.
    '''
    if 'area' not in cells_gdf:
        cells_gdf['area'] = cells_gdf.area
    cell_id_col = cells_gdf.index.name

    latlon_cells_gdf = cells_gdf.to_crs('epsg:4326')
    point_places_mask = places_gdf.geom_type == 'Point'
    pt_places_to_cells = (
        places_gdf.loc[point_places_mask]
         .sjoin(latlon_cells_gdf, predicate='intersects')[['index_right']]
         .assign(ratio=1)
         .rename(columns={'index_right': cell_id_col})
         .set_index(cell_id_col, append=True)
    )

    xy_poly_places = places_gdf.loc[~point_places_mask].to_crs(xy_proj)
    xy_poly_places['area'] = xy_poly_places.area
    cells_gdf[cell_id_col] = cells_gdf.index
    poly_places_to_cells = xy_poly_places[['geometry', 'id', 'area']].overlay(
        cells_gdf[['geometry', cell_id_col]], how='intersection'
    )
    poly_places_to_cells['area_overlap'] = poly_places_to_cells.area
    poly_places_to_cells['ratio_overlap'] = (
        poly_places_to_cells['area_overlap']
        / poly_places_to_cells['area']
    )
    poly_places_to_cells['ratio_clipped_overlap'] = (
        poly_places_to_cells['area_overlap']
        / poly_places_to_cells.groupby('id')['area_overlap'].transform('sum')
    )
    poly_places_to_cells = (
        poly_places_to_cells.set_index(['id', cell_id_col])
         .sort_index()
         .drop(columns='geometry')
    )
    place_reg_overlap_mask = (
        poly_places_to_cells.groupby('id')['ratio_overlap'].sum()
        < reg_overlap_th
    )
    idc_foreign = poly_places_to_cells.index.levels[0][place_reg_overlap_mask]
    poly_places_to_cells = poly_places_to_cells.drop(index=idc_foreign, level='id')
    # Since groubpy preserves order within each group, the cumsum is computed from the
    # largest to the smallest value in each group. Then we take the first `ntop` (at
    # most, if there are fewer rows they are kept, unlike when using `nth`), in order to
    # take the `ntop`th. For each place, this value then corresponds to the total ratio
    # of overlap of the place summed over the `ntop` cells it overlaps the most with.
    places_top_cells_cumoverlap = (
        poly_places_to_cells.sort_values(by='ratio_overlap', ascending=False)
         .groupby('id').cumsum()
         .groupby('id')['ratio_clipped_overlap'].head(ntop)
         .groupby('id').last()
    )
    places_to_remove = places_top_cells_cumoverlap.index[
        places_top_cells_cumoverlap < top_cells_cumoverlap_th
    ]
    poly_places_to_cells = (
        poly_places_to_cells[['ratio_clipped_overlap']]
         .rename(columns={'ratio_clipped_overlap': 'ratio'})
         .drop(index=places_to_remove, level='id')
    )

    return pd.concat([pt_places_to_cells, poly_places_to_cells]).sort_index()


def get_cell_user_activity(
    user_places_counts, places_to_cells_corr, subcells_user_counts_from_gps, subcells_to_cells
):
    '''
    `subcells_to_cells` Series making the correspondence between the indices of the
    subcells (in its Index) and the ones of the final cells (its values)
    '''
    agg_cells_name = subcells_to_cells.index.names[0]
    user_gps_counts = (
        subcells_user_counts_from_gps.join(subcells_to_cells)
         .groupby(['user_id', agg_cells_name, 'is_daytime'])
         .sum()
    )

    user_acty = (
        user_places_counts.join(
            places_to_cells_corr.reset_index().set_index('id'), on='place_id', how='inner'
        ).groupby(['user_id', agg_cells_name, 'is_daytime']).sum()
    )
    user_acty['count'] = user_acty['count'] * user_acty['ratio']
    user_acty = user_acty.drop(columns='ratio').add(user_gps_counts, fill_value=0)
    return compute_time_fracs(user_acty)


def compute_time_fracs(user_acty):
    user_acty['prop_cell'] = (
        user_acty['count']
        / user_acty.groupby('user_id')['count'].transform('sum')
    )
    user_acty['prop_cell_by_time'] = (
        user_acty['count']
        / user_acty.groupby(['user_id', 'is_daytime'])['count'].transform('sum')
    )
    return user_acty


def assign(user_acty, nighttime_acty_th=0.5, all_acty_th=0.1, count_th=3):
    '''
    Make the user to residence cell final assignation. Take the cells which accounted
    for more than a proportion `nighttime_acty_th` of tweets posted by a given user
    during nighttime (outside of [`hour_start_day`, `hour_end_day` - 1] interval), that
    also accounted for more than a proportion `all_acty_th` of all activity of the user,
    and were the user tweeted from at least `count_th` times.
    '''
    # not optimal, extract nighttime part first?
    nighttime_acty_mask = user_acty['prop_cell_by_time'] >= nighttime_acty_th
    all_acty_mask = user_acty['prop_cell'] >= all_acty_th
    count_th_mask = user_acty['count'] >= count_th
    user_residence_cells = (
        user_acty.loc[nighttime_acty_mask & all_acty_mask & count_th_mask]
         .loc[(slice(None), slice(None), False)]
         .sort_values(by='count')
         .groupby('user_id')
         .tail(1)
         .reset_index().set_index('user_id')
    )
    return user_residence_cells


def user_acty_from_saved_counts(
    reg, gps_attr_level, pois_dups_th=-1, gps_dups_th=-1,
):
    # Load user counts per cell
    subcells_user_counts_from_gps = pd.read_parquet(
        str(reg.paths.user_cells_from_gps).format(
            gps_attr_cell_size=gps_attr_level, gps_dups_th=gps_dups_th
        )
    )

    cell_shp_metadata = reg.cell_shapefiles[reg.res_cell_size]
    cell_col = cell_shp_metadata['index_col']
    subcell_shp_metadata = reg.cell_shapefiles[gps_attr_level]
    subcell_col = subcell_shp_metadata['index_col']
    subcells_to_cells = reg.cell_levels_corr.copy()
    subcells_to_cells = spatial_agg.levels_corr(
        subcells_to_cells, subcell_col, cell_col
    )

    # Load user counts per place
    user_places_counts = pd.read_parquet(reg.paths.user_places)
    poi_counts = pd.DataFrame()
    year_fmt = str(reg.paths.interim_data / reg.cc / "poi_counts_{year}.parquet")
    for year_count in yield_yearly_poi_counts(
        reg.year_from, reg.year_to, reg.mongo_coll, year_fmt
    ):
        poi_counts = poi_counts.add(
            year_count,
            fill_value=0,
        )
    user_places_counts = user_places_counts.join(
        poi_counts.loc[poi_counts['count'] < pois_dups_th, []],
        how='inner',
    )

    # Get places to cells matching
    places_filter = qr.Filter().equals('country_code', reg.cc)
    tweets_filter = qr.Filter().not_exists('coordinates.coordinates')
    places_gdf = data_access.agg_places_from_mongo(
        reg.year_from, reg.year_to, places_filter, tweets_colls=reg.mongo_coll, tweets_filter=tweets_filter
    )

    places_to_cells_corr = places_to_cells(
        places_gdf, reg.cells_geodf, xy_proj=reg.xy_proj, ntop=2,
        top_cells_cumoverlap_th=0.9, reg_overlap_th=0.6
    )

    user_acty = get_cell_user_activity(
        user_places_counts, places_to_cells_corr,
        subcells_user_counts_from_gps, subcells_to_cells
    )
    return user_acty


def whole_cell_attr(
    reg, gps_attr_level, pois_dups_th=-1, gps_dups_th=-1, **assign_kwargs
):
    user_acty = user_acty_from_saved_counts(
        reg, gps_attr_level, pois_dups_th=-1, gps_dups_th=-1,
    )
    user_residence_cell = assign(user_acty, **assign_kwargs)
    return user_residence_cell
