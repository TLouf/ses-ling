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


def get_user_corpora(chunk_tweets):
    user_corpora = (
        chunk_tweets.groupby('user_id')
         .agg(
             text=('filtered_text', '\n'.join),
             nr_tweets=('filtered_text', 'size'),
             nr_words=('nr_words', 'sum'),
                # unique cannot be parallelised. store list of words with integer identifier separately? a global one for all users, getting enriched. global_counts basically. then in here just store number sequence? cant with max because there might be holes. or array: first elem is max until hole, then rest of sequence. int arrays to store for ~10k users should be ok
                # vocab_size=('filtered_text', text_process.measure_vocab_size),
            )
    )
    return user_corpora


def process_year(year, f, colls, lc, words_index, user_corpora=None):
    db = f'twitter_{year}'
    tweet_cols = ['text', 'source', 'user_id']
    chunk_tweets = data_access.tweets_from_mongo(db, f, colls, tweet_cols)
    LOGGER.info(f'- Retrieved tweets from {year}')
    chunk_tweets = text_process.clean(chunk_tweets, lang=lc)
    LOGGER.info(f'- Cleaned tweets from {year}')
    if user_corpora is None:
        user_corpora = get_user_corpora(chunk_tweets)
    else:
        count_cols = ['nr_tweets', 'nr_words']
        user_corpora = user_corpora.join(
            get_user_corpora(chunk_tweets), how='outer', rsuffix='new'
        )
        new_count_cols = [f'{c}new' for c in count_cols]
        # for those users in previous year but not in this one (no "textnew" column
        # because we drop the text at the end of this function)
        user_corpora['text'] = user_corpora['text'].fillna('')
        # handle both users not in previous year but in this one and vice versa.
        rename_dict = dict(zip(new_count_cols, count_cols))
        user_corpora[count_cols] = user_corpora[count_cols].add(
            user_corpora[new_count_cols].rename(columns=rename_dict), fill_value=0
        )
        user_corpora = user_corpora.drop(columns=new_count_cols)

    # free up memory
    del chunk_tweets
    lt_config = {'cacheSize': 1000, 'pipelineCaching': True}
    with language_tool_python.LanguageTool(lc, config=lt_config) as language_tool:
        LOGGER.info(f"Starting LT for {year} with PID: {os.getpid()!r}")
        language_tool._TIMEOUT = 8 * 3600
        user_mistakes = text_process.count_mistakes(user_corpora, language_tool)
    LOGGER.info(f'- Processed mistakes from {year}')
    user_corpora = text_process.yearly_vocab(user_corpora, words_index).drop(columns='text')
    return user_mistakes, user_corpora


@ray.remote(num_returns=2)
def process_chunk(
    paths, i_chunk, chunk_ids, pre_filter, lc, colls, year_from, year_to, words_index
):
    logging.config.dictConfig(LOG_CONFIG)
    LOGGER.info(f'** starting on chunk {i_chunk}')
    f = pre_filter.copy().any_of('user.id', chunk_ids)
    # count_cols = ['nr_tweets', 'nr_words']
    
    for year in range(year_from, year_to+1):
        LOGGER.info(f'* starting on year {year}')
        # user_mistakes = user_mistakes.join(user_corpora)
        # don't forget users who didn't make a single mistake!
        # so save both?
        if year == year_from:
            chunk_mistakes, chunk_corpora = process_year(
                year, f, colls, lc, words_index
            )
        else:
            year_mistakes, chunk_corpora = process_year(
                year, f, colls, lc, words_index,
                user_corpora=chunk_corpora,
            )
            chunk_mistakes = chunk_mistakes.add(year_mistakes, fill_value=0)
            # chunk_corpora[count_cols] = chunk_corpora[count_cols].add(
            #     year_corpora[count_cols], fill_value=0
            # )

    chunk_corpora['nr_unique_words'] = chunk_corpora['words_until'].fillna(0).astype(int)
    has_words = chunk_corpora['rank_arr'].notna()
    chunk_corpora.loc[has_words, 'nr_unique_words'] = (
        chunk_corpora.loc[has_words, 'nr_unique_words']
        + chunk_corpora.loc[has_words, 'rank_arr'].apply(len)
    )
    chunk_corpora = chunk_corpora.drop(columns=['rank_arr', 'words_until'])
    chunk_corpora.join(
        chunk_mistakes.droplevel('category').unstack().droplevel(0, axis=1)
    ).to_parquet(str(paths.chunk_user_mistakes).format(i_chunk=i_chunk))
    LOGGER.info(f'** done with chunk {i_chunk} **')
    return chunk_mistakes, chunk_corpora


def main(paths, lc, cc, colls, year_from, year_to, max_tweets_in_chunk=5e6):
    # splt by count, to balance out each user chunk's corpora, up to x tweets in total in one chunk
    # save_path = str(paths.user_cells_from_gps).format(cell_kind=cc_dict['cell_kind'])
    user_nr_tweets = pd.read_parquet(paths.user_cells_from_gps).groupby('user_id').sum()['count']
    # save_path = str(paths.user_places)
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
    words_index = pd.read_parquet(
        paths.counts_files.parent
        / f'region_counts_lang={lc}_cc={cc}_years={year_from}-{year_to}.parquet'
    )
    # can only keep thse with count > 1, if ntot present in there then you know you can just put max_index + 1
    words_index = words_index.loc[words_index['count'] > 1].index
    words_index = pd.Series(range(len(words_index)), index=words_index, name='rank')
    list_chunk_mistakes, list_chunk_corpora = (
        list(z) for z in zip(*[
            process_chunk.remote(
                paths, i_chunk, chunk_ids, pre_filter, lc, colls, year_from, year_to, words_index
            )
            for i_chunk, chunk_ids in [(i, list_user_chunks[i]) for i in (7, 10)] #enumerate(list_user_chunks[7:10], start=7) # TODO change
        ])
    )

    user_mistakes = pd.concat(ray.get(list_chunk_mistakes))
    user_corpora = pd.concat(ray.get(list_chunk_corpora))
    rule_categories = user_mistakes.index.to_frame(index=False)[['category', 'ruleId']].groupby('ruleId')['category'].first()
    user_mistakes = user_corpora.join(
        user_mistakes.droplevel('category').unstack().droplevel(0, axis=1)
    )
    rule_categories.to_csv(paths.rule_category)
    user_mistakes.to_parquet(paths.user_mistakes, index=True)


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
    cc_dict['cell_kind'] = 'MSOA_BGC'

    assign_kwargs = dict(
        nighttime_acty_th = 0.5,
        all_acty_th = 0.1,
        count_th = 3,
    )

    paths.partial_format(**cc_dict, **assign_kwargs)

    colls = cc_dict['mongo_coll']
    ray.init(num_cpus=2) # TODO
    main(paths, lc, cc, colls, year_from, year_to, max_tweets_in_chunk=5e6)
    ray.shutdown()
