import os
import sys
import datetime
import re
import json
from pathlib import Path
import logging
import logging.config
import ray
import numpy as np
import pandas as pd
import geopandas as geopd
import language_tool_python
import querier as qr
from dotenv import load_dotenv
load_dotenv()

import ses_ling.data.access as data_access
import ses_ling.utils.paths as path_utils
import ses_ling.utils.text_process as text_process

LOGGER = logging.getLogger(__name__)
LOG_CONFIG = json.load(open('logging.json', 'rt'))
logging.config.dictConfig(LOG_CONFIG)

qr.init_logger("querier", "querier.log", level=logging.INFO)


def assign_user_chunk(user_chunk_df, max_tweets_in_chunk=5e6):
    rng = np.random.default_rng(seed=0)
    nr_chunks = user_chunk_df['nr_tweets'].sum() // max_tweets_in_chunk
    std = max_tweets_in_chunk
    while std > 0.1 * max_tweets_in_chunk:
        nr_users = user_chunk_df.shape[0]
        user_chunk_df['chunk'] = (rng.random(nr_users) * nr_chunks).astype(int)
        std = user_chunk_df.groupby('chunk')['nr_tweets'].sum().std()
        LOGGER.info(f'STD of {std:.0f} in number of tweets between chunks.')
    return user_chunk_df


def get_lang_grp(user_langs_counts, lang_relevant_prop=0.1,
                 lang_relevant_count=5
):
    '''
    Out of the counts of tweets by language of each user in `user_langs_counts`,
    determines which one are relevant for each user, before assigning them their
    spoken languages in `user_langs_agg`.
    '''
    total_per_user = (user_langs_counts.groupby('uid')['count']
                                       .sum()
                                       .rename('user_count'))
    user_langs_agg = user_langs_counts.join(total_per_user).assign(
        prop_lang=lambda df: df['count'] / df['user_count'])
    prop_relevant = user_langs_agg['prop_lang'] > lang_relevant_prop
    count_relevant = user_langs_agg['count'] > lang_relevant_count
    user_langs_agg = user_langs_agg.loc[prop_relevant | count_relevant]
    uid_with_lang = user_langs_agg.index.levels[0].values
    print(f'We were able to attribute at least one language to '
          f'{len(uid_with_lang)} users')
    return user_langs_agg


def process_year(year, f, colls):
    db = f'twitter_{year}'
    tweet_cols = ['text', 'source', 'user_id']
    chunk_tweets = data_access.tweets_from_mongo(db, f, colls, tweet_cols)
    LOGGER.info(f'- Retrieved tweets from {year}')
    chunk_tweets = text_process.clean(chunk_tweets).drop(columns=['text', 'filtered_text'])
    user_langs = chunk_tweets.groupby([ 'user_id', 'cld_lang' ]).count()
    LOGGER.info(f'- Cleaned tweets from {year}')
    return user_langs


@ray.remote
def process_chunk(
    paths, i_chunk, chunk_ids, pre_filter, colls, year_from, year_to
):
    logging.config.dictConfig(LOG_CONFIG)
    LOGGER.info(f'** starting on chunk {i_chunk}')
    f = pre_filter.copy().any_of('user.id', chunk_ids)

    for year in range(year_from, year_to+1):
        LOGGER.info(f'* starting on year {year}')
        if year == year_from:
            chunk_langs = process_year(
                year, f, colls
            )
        else:
            year_langs = process_year(
                year, f, colls
            )
            chunk_langs = chunk_langs.add(year_langs, fill_value=0)

    return chunk_langs


def main(paths, cc, colls, year_from, year_to, max_tweets_in_chunk=5e6):
    # splt by count, to balance out each user chunk's corpora, up to x tweets in total in one chunk
    user_nr_tweets = pd.read_parquet(paths.user_cells_from_gps).groupby('user_id').sum()['count']
    user_nr_tweets = (
        user_nr_tweets.add(
            pd.read_parquet(paths.user_places).groupby('user_id').sum()['count'],
            fill_value=0
         )
         .rename('nr_tweets')
    )
    user_residence = pd.read_parquet(paths.user_residence_cell).join(user_nr_tweets)
    # somehow join most active with least active users...
    user_residence = assign_user_chunk(user_residence, max_tweets_in_chunk=max_tweets_in_chunk)
    list_user_chunks = [list(idc) for idc in user_residence.groupby('chunk').groups.values()]
    LOGGER.info(f'{len(list_user_chunks)} chunks of users.')

    pre_filter = qr.Filter().equals('place.country_code', cc)
    list_chunk_langs = [
            process_chunk.remote(
                paths, i_chunk, chunk_ids, pre_filter, colls, year_from, year_to
            )
            for i_chunk, chunk_ids in enumerate(list_user_chunks) # [(i, list_user_chunks[i]) for i in (7, 10)] #enumerate(list_user_chunks[7:10], start=7) # TODO change
    ]

    user_langs = pd.concat(ray.get(list_chunk_langs))
    user_langs.to_parquet(paths.user_langs, index=True)
    return user_langs

if __name__ == '__main__':
    lc = sys.argv[1]
    cc = sys.argv[2]
    year_from = int(sys.argv[3])
    year_to = int(sys.argv[4])

    paths = path_utils.ProjectPaths()
    with open(paths.countries_dict) as f:
        countries_dict = json.load(f)
    cc_dict = countries_dict[cc]
    cc_dict['cc'] = cc
    cc_dict['lc'] = lc
    cc_dict['year_from'] = year_from
    cc_dict['year_to'] = year_to
    cc_dict['res_cell_size'] = 'MSOA_BGC'

    assign_kwargs = dict(
        gps_attr_cell_size='LSOA_BGC',
        nighttime_acty_th = 0.5,
        all_acty_th = 0.1,
        count_th = 3,
        gps_dups_th=500,
        pois_dups_th=500,
    )

    paths.partial_format(**cc_dict, **assign_kwargs)
    print(paths.user_langs)

    colls = cc_dict['mongo_coll']
    ray.init(num_cpus=8) # TODO
    user_langs = main(paths, cc, colls, year_from, year_to, max_tweets_in_chunk=5e6)
    ray.shutdown()
