'''
This module is aimed at reading JSON data files, either locally or from a remote
host. The data files are not exactly JSON, they're files in which each line is a
JSON object, thus making up a row of data, and in which each key of the JSON
strings refers to a column. Cannot use Modin or Dask because of gzip
compression, Parquet data files would be ideal.
'''
from __future__ import annotations
from typing import Literal, Iterable
from functools import reduce
from math import ceil
import datetime
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
import geopandas as geopd
import bson
import querier as qr

import ses_ling.utils.geometry as geo_utils


LOGGER = logging.getLogger(__name__)
TWEETS_COLS = {
    'id': {'field': 'id', 'dtype': 'string'},
    'user_id': {'field': 'user.id', 'dtype': 'string'},
    'text': {'field': ['extended_tweet.full_text', 'text'], 'dtype': 'string'},
    'coordinates': {'field': 'coordinates.coordinates', 'dtype': 'object'},
    'place_id': {'field': 'place.id', 'dtype': 'string'},
    'created_at': {'field': 'created_at', 'dtype': 'datetime64[ns]'},
    'source': {'field': 'source', 'dtype': 'string'},
}


def places_from_mongo(
    db: str, filter: qr.Filter, add_fields=None, tweets_filter=None, tweets_colls=None
) -> geopd.GeoDataFrame:
    '''
    Return the GeoDataFrame of places in database `db` matching `filter`. Optionally, if
    there is no "places" collection in the database, will retrieve them from the tweets
    in `tweets_colls` matching `tweets_filter`, via `places_from_mongo_tweets`.
    '''
    if add_fields is None:
        add_fields = []
    default_fields = ['id', 'name', 'place_type', 'bounding_box.coordinates']
    all_fields = default_fields + add_fields

    with qr.Connection(db) as con:

        if 'places' not in con.list_available_collections():
            if tweets_colls is None:
                raise ValueError(
                    f'There is no places collection in {db}, specify tweet'
                    ' collections from which to retrieve them.'
                )
            return places_from_mongo_tweets(
                db, tweets_colls, tweets_filter=tweets_filter, add_fields=add_fields
            )

        places = con['places'].extract(filter, fields=all_fields)

        all_fields.remove('bounding_box.coordinates')
        places_dict = {key: [] for key in all_fields}
        places_dict['geometry'] = []

        for p in places:
            if p.get('bounding_box') is None:
                continue

            for f in all_fields:
                places_dict[f].append(p[f])

            bbox = p['bounding_box']['coordinates'][0]
            geo, _ = geo_utils.geo_from_bbox(bbox)
            places_dict['geometry'].append(geo)

    raw_places_gdf = geopd.GeoDataFrame(places_dict, crs='epsg:4326').set_index('id', drop=False)
    return raw_places_gdf


def places_from_mongo_tweets(
    db: str, colls: str | list, tweets_filter: qr.Filter | None = None, add_fields=None
) -> geopd.GeoDataFrame:
    '''
    Returns the GeoDataFrame of places in database `db` from the tweets in `colls`
    matching `tweets_filter`.
    '''
    if add_fields is None:
        add_fields = []
    if isinstance(colls, str):
        colls = [colls]

    with qr.Connection(db) as con:
        agg_dict = {
            "name": ("place.name", "first"),
            "place_type": ("place.place_type", "first"),
            "nr_tweets": ("place.id", "count"),
            "bbox": ("place.bounding_box.coordinates", "first"),
            **{field: (f"place.{field}", "first") for field in add_fields}
        }
        geodf_dicts = []

        for coll in colls:
            places = con[coll].groupby(
                'place.id', pre_filter=tweets_filter, allowDiskUse=True
            ).agg(**agg_dict)

            for p in tqdm(places):
                p['id'] = p.pop('_id')['place_id']
                bbox = p.pop('bbox')[0]
                p['geometry'], _ = geo_utils.geo_from_bbox(bbox)
                geodf_dicts.append(p)

    raw_places_gdf = (geopd.GeoDataFrame.from_dict(geodf_dicts)
                                        .drop_duplicates(subset='id')
                                        .set_index('id', drop=False)
                                        .set_crs('epsg:4326'))
    return raw_places_gdf


def agg_places_from_mongo(
    year_from, year_to, filter, add_fields=None, tweets_filter=None, tweets_colls=None
):
    '''
    Returns the GeoDataFrame of places contained in the databases corresponding to the
    years between `year_from` and `year_to` included.
    '''
    places_gdf = geopd.GeoDataFrame()
    for year in range(year_to, year_from-1, -1):
        # Read in descending chronological order so as to keep the most recent version
        # of the geometry of every place ID.
        db = f'twitter_{year}'
        places_gdf = places_gdf.combine_first(
            places_from_mongo(
                db, filter, add_fields=add_fields, tweets_filter=tweets_filter,
                tweets_colls=tweets_colls
            )
        )
        LOGGER.info(f'As of year {year}, {places_gdf.shape[0]} places retrieved')

    places_gdf.crs = 'epsg:4326'
    return places_gdf


def tweets_from_mongo(
    db, filter, colls: str | list, cols=None, limit=-1,
    coords_as: Literal['list', 'xy'] = 'xy'
) -> pd.DataFrame:
    '''
    Return the DataFrame of tweets in the collections `colls` of the database `db`
    matching `filter`. add_cols, if specified, should be a dictionary where each element
    is of the form: <output_field>: {'field': <input_field(s)>, 'dtype': <dtype>}
    '''
    cols_dict = get_tweets_df_cols_dict(cols=cols)
    fields_to_extract = []
    for d in cols_dict.values():
        if isinstance(d['field'], str):
            d['field'] = [d['field']]
        fields_to_extract.extend(d['field'])

    with qr.Connection(db) as con:
        tweets = con[colls].extract(filter, fields=fields_to_extract).limit(limit)
        tweets_dict = tweets_res_to_dict(
            tweets, cols_dict=cols_dict, coords_as=coords_as
        )

    # Use `tweets_dict.keys()` and not `cols_dict.keys()` because only some fields may
    # be kept, although they are extracted from the DB.
    dtypes = {
        col: cols_dict.get(col, {}).get('dtype')
        for col in tweets_dict.keys()
    }
    tweets_df = pd.DataFrame(tweets_dict).astype(dtypes)
    return tweets_df


def get_tweets_df_cols_dict(cols: list[str | dict] = None):
    '''
    Returns a dictionary which, for each output column (dict key), give the name of the
    field in the DB, the output dtype and optionally whether it should be kept (via the
    `keep` keyword, only useful for 'id' a priori). It is constructed based on the input
    elements of `cols`: a str will select a key from the default `TWEETS_COLS`, and a
    dictionary of the format of the ones in `TWEETS_COLS` will give custom info, which
    is more flexible.
    '''
    if cols is None:
        cols_dict = TWEETS_COLS.copy()
    else:
        cols_dict = {}
        for c in cols:
            if isinstance(c, str):
                cols_dict[c] = TWEETS_COLS[c].copy()
            elif isinstance(c, dict):
                cols_dict = {**cols_dict, **c}
        # id always needs to be included to detect duplicates, but if not provided,
        # interpret as the user does not want it in the output dataframe.
        if 'id' not in cols_dict:
            cols_dict['id'] = {**TWEETS_COLS['id'], **{'keep': False}}
    return cols_dict


def tweets_res_to_dict(
    res: qr.Result, cols_dict=None, add_cols=None,
    coords_as: Literal['list', 'xy'] = 'xy'
) -> dict:
    '''
    Iterates a query's result `res` to populate a dictionary which gives for each output
    field present in `cols_dict` (default None, all `TWEETS_COLS` fields), a list of the
    values taken for each tweet.
    '''
    if cols_dict is None:
        cols_dict = get_tweets_df_cols_dict(add_cols=add_cols)
    # res iterator over a result
    assign_dict = {
        col: [f.split('.') for f in col_dict['field']]
        for col, col_dict in cols_dict.items()
        if col_dict.get('keep', True)
    }
    tweets_dict = {key: [] for key in assign_dict.keys()}

    last_id = ''
    for t in res:
        if t['id'] != last_id:
            last_id = t['id']
            # In the result, nested fields will actually be contained in a nested
            # dict.
            for col, fields in assign_dict.items():
                for f in fields:
                    value = t.get(f[0])

                    if len(f) > 1 and value is not None:
                        for part in f[1:]:
                            value = value.get(part)

                    if value is not None:
                        break

                tweets_dict[col].append(value)

    # first check to check if res is not empty
    if last_id != '' and 'coordinates' in tweets_dict and coords_as == 'xy':
        coords_arr = np.array(tweets_dict.pop('coordinates'))
        tweets_dict['x'] = coords_arr[:, 0]
        tweets_dict['y'] = coords_arr[:, 1]

    return tweets_dict


def geo_within_bbox_filter(minx, miny, maxx, maxy):
    '''
    From bounds defining a bounding box, return a Mongo filter of geometries
    within this bounding box.
    '''
    return geo_within_filter([
        [minx, miny],
        [minx, maxy],
        [maxx, maxy],
        [maxx, miny],
        [minx, miny]
    ])

def geo_within_filter(polygon_coords):
    '''
    From a list of coordinates defining a polygon `polygon_coords`, return a
    Mongo filter of geometries within this polygon.
    '''
    # Close the polygon if not done
    if polygon_coords[0] != polygon_coords[-1]:
        polygon_coords.append(polygon_coords[0])
    return {
        '$geoWithin': {
            '$geometry': {
                'coordinates': [polygon_coords],
                'type': 'Polygon'
            }
        }
    }


def get_all_types_mongo_id(id: int | str | bson.ObjectId):
    # only conversion that's not clear how to perform is str, objectid -> int, but very
    # few int "_id" so should be ok to discard them.
    types = [id]
    if isinstance(id, bson.ObjectId):
        types.append(str(id))
    elif isinstance(id, str):
        types.append(bson.ObjectId(id.zfill(24)))
    elif isinstance(id, int):
        types.append(str(id).zfill(24))
        types.append(bson.ObjectId(types[-1]))
    return types


def chunk_filters_mongo(db, colls: str | list, filter, chunksize=1e6):
    '''
    Returns a list of filters of ID ranges that split the elements of collection
    `coll` of MongoDB database `db` matching `filter` into chunks of size
    `chunksize`. From (as I'm writing, not-yet-released) dask-mongo.
    '''
    with qr.Connection(db) as con:
        nrows = con[colls].count_entries(filter)
        npartitions = int(ceil(nrows / chunksize))
        LOGGER.info(
            f'{nrows:.3e} tweets matching the filter in collection {colls} of '
            f'DB {db}, dividing in {npartitions} chunks...'
        )
        # TODO: same but with "groupBy": "$created_at"?? may be slower though?
        partitions_ids = list(
            con[colls].aggregate(
                [
                    {"$match": filter},
                    {"$bucketAuto": {"groupBy": "$_id", "buckets": npartitions}},
                ],
                allowDiskUse=True,
            )
        )

    chunk_filters = []
    greater_fun = lambda id: qr.Filter().greater_or_equals("_id", id)
    less_fun = lambda id: qr.Filter().less_than("_id", id)

    for i, partition in enumerate(partitions_ids):
        # As the type of the "_id" field is not consistent throughout the database, for
        # each bound we have to allow comparison with all possible types. So what we do
        # below is to get chunk_filter = (_id > lower bound for one type at least) and
        # (_id <(=) upper bound for one type at least).
        or_operator = lambda f1, f2: f1 | f2

        min_id = partition["_id"]["min"]
        all_types_min_id = get_all_types_mongo_id(min_id)
        min_id_f = reduce(or_operator, map(greater_fun, all_types_min_id))

        max_id = partition["_id"]["max"]
        all_types_max_id = get_all_types_mongo_id(max_id)
        if i == npartitions - 1:
            less_fun = lambda id: qr.Filter().less_or_equals("_id", id)
        max_id_f = reduce(or_operator, map(less_fun, all_types_max_id))

        chunk_filters.append(min_id_f & max_id_f)

    return chunk_filters


def dt_chunk_filters_mongo(db, colls: str | list, filter, start, end, chunksize=1e6):
    '''
    Returns a list of filters of datetime ranges that split the elements of collection
    `coll` of MongoDB database `db` matching `filter` into chunks of equal time delta.
    Has the advantage of being faster than `chunk_filters_mongo`, to the cost of having
    chunks of unequal sizes.
    '''
    count_filter = filter.copy()
    count_filter.greater_or_equals('created_at', start)
    count_filter.less_than('created_at', end)

    with qr.Connection(db) as con:
        nrows = con[colls].count_entries(count_filter)
        npartitions = int(ceil(nrows / chunksize))
        LOGGER.info(
            f'{nrows:.3e} tweets matching the filter in collection {colls} of '
            f'DB {db}, dividing in {npartitions} chunks...'
        )

    dt_step = (end - start) / npartitions
    dt_bounds = [start + i*dt_step for i in range(npartitions+1)]

    chunk_filters = []
    for i in range(npartitions):
        f = filter.copy()
        f.greater_or_equals('created_at', dt_bounds[i])
        f.less_than('created_at', dt_bounds[i+1])
        chunk_filters.append(f)

    return chunk_filters


def groupby_res_to_df(res: qr.Result | Iterable, pipeline) -> pd.DataFrame:
    '''
    From the result `res` of a Mongo aggregation defined in the pipeline `pipeline`
    ending with a groupby stage, returns a DataFrame containing the result, with a
    potential `MultiIndex` if there is a groupby on more than one field.
    '''
    # Get the last groupby stage, and copy to preserve pipeline.
    cols = [p['$group'] for p in pipeline if '$group' in p][-1].copy()
    id_part = cols.pop('_id')

    # If `id_part` is a dict, it means each element in `res` has a depth of 2: the `_id`
    # is given by a dict, associating a value to each groupby key.
    if isinstance(id_part, dict):
        id_keys = list(id_part.keys())
        vals_dict = {k: [] for k in cols}
        ids_dict = {k: [] for k in id_keys}

        for r in res:
            for idk in id_keys:
                ids_dict[idk].append(r['_id'][idk])
            for c in cols:
                vals_dict[c].append(r[c])

        if len(id_keys) > 1:
            index = pd.MultiIndex.from_arrays(list(ids_dict.values()), names=id_keys)
        else:
            index = pd.Index(list(ids_dict.values())[0], name=id_keys[0])

        df = pd.DataFrame(vals_dict, index=index)

    # Else the `_id` field just gives the index value (1D).
    else:
        # We normalize the index name to be more pythonic.
        idx_name = id_part.replace('$', '').replace('.', '_')
        df = pd.DataFrame(list(res)).set_index('_id').rename_axis(idx_name)
    
    return df


def mongo_groupby_to_df(groupby: qr.MongoGroupBy, **agg_kwargs):
    res = groupby.agg(**agg_kwargs)
    return groupby_res_to_df(res, groupby.pipeline)


def gps_filter(f: qr.Filter):
    return f.copy().exists('coordinates.coordinates')


def base_tweets_filter(year: int, filter_sources=False):
    year_start = datetime.datetime(year, 1, 1)
    f = qr.Filter().greater_or_equals('created_at', year_start)
    if filter_sources:
        # Selected after looking at counts by source for 2015-2021 in the US, retains
        # most of the tweets while excluding obvious bot accounts.
        source_patt = '>Twitter |>Instagram|>Tweetbot|>Foursquare'
        f = f.regex('source', source_patt)
    return f
