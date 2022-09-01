import datetime
import numpy as np
import pandas as pd
import querier as qr

import ses_ling.data.access as data_access


def get_consec_months_in_year(year, colls, pre_filter=None):
    db = f'twitter_{year}'
    f = data_access.base_tweets_filter(year)
    if pre_filter is not None:
        f = f & pre_filter

    with qr.Connection(db) as con:
            pipeline = [
                {'$match': f},
                {
                    "$group": {
                        '_id': {
                            'user_id': "$user.id",
                            # since year is constant in one db, don't bother with
                            # commented stuff
                            'month_nr': {
                                # "$add": [
                                    # {
                                        '$month': {'date' : '$created_at'}
                                    # },
                                #     {
                                #         '$multiply': [
                                #             {'$year': {'date' : '$created_at'}},
                                #             12,
                                #         ]
                                #     }
                                # ]
                            }
                        },
                        "count": {"$sum": 1 },
                    }
                },
            ]
            res = list(con[colls].aggregate(pipeline, allowDiskUse=True))

    user_months_df = data_access.mongo_groupby_to_df(res, pipeline)
    user_months_df['year'] = year
    user_months_df = user_months_df.set_index('year', append=True)
    # put 'year' level before 'month_nr'
    user_months_df = user_months_df.swaplevel(-2, -1)
    return user_months_df


def agg_consec_months(year_from, year_to, colls, pre_filter=None):
    user_months_df = pd.DataFrame()
    for year in range(year_from, year_to + 1):
        user_months_df = pd.concat([
            user_months_df,
            get_consec_months_in_year(year, colls, pre_filter=pre_filter),
        ])
    return user_months_df


def get_resident_ids(user_activity_by_month, nr_consec_months=3) -> list:
    '''
    From 'user_activity_by_month', a DataFrame of all the months in which the
    users have tweeted, returns an
    Index of all the IDs of users considered to be locals. Is considered local
    a user who has tweeted within at least `nr_consec_months` consecutive months.
    '''
    user_activity_by_month = user_activity_by_month.sort_index(
        level=['user_id', 'year', 'month_nr']
    )
    year_values = user_activity_by_month.index.get_level_values('year')
    nr_years = 1 + year_values.max() - year_values.min()
    user_activity_by_month['month_nr_by_user'] = (
        12 * year_values
        + user_activity_by_month.index.get_level_values('month_nr')
    # add the following term to distinguish two users, so that 'month_nr_by_user' may
    # never have consecutive values for different users. It's like faking we had rows
    # for every possible month for every user.
        + 12 * nr_years * user_activity_by_month.index.codes[0]
    )
    # Because index is sorted, only way for month_nr_{i+2} - month_nr_{i} to be equal to
    # `nr_consec_months` is to have elements at i, i+1 and i+2 consecutive.
    nr_gaps = nr_consec_months - 1
    mask_col = f'is_{nr_consec_months}_consec_months'
    month_nr_by_user = user_activity_by_month['month_nr_by_user'].values
    user_activity_by_month[mask_col] = np.concatenate([
        np.zeros(nr_gaps, dtype=bool),
        month_nr_by_user[nr_gaps:] - month_nr_by_user[:-nr_gaps] == nr_gaps
    ])
    consec_months_mask = user_activity_by_month[mask_col].groupby('user_id').max()
    # Return as list as needed in Mongo filters
    residents = consec_months_mask.index[consec_months_mask].values.tolist()
    return residents


def get_user_activity_by_source(year, colls, pre_filter=None):
    db = f'twitter_{year}'
    f = data_access.base_tweets_filter(year)
    if pre_filter is not None:
        f = f & pre_filter

    with qr.Connection(db) as con:
        pipeline = [
            {'$match': f},
            {
                "$group": {
                    "_id": {"user_id": "$user.id", "source": "$source"},
                    "count": {"$sum": 1},
                    "lastTime": {"$last": "$created_at"},
                    "firstTime": {"$first": "$created_at"},
                }
            },
        ]
        res = list(con[colls].aggregate(pipeline, allowDiskUse=True))

    user_activity_df = data_access.mongo_groupby_to_df(res, pipeline)
    return user_activity_df


def agg_user_activity_by_source(year_from, year_to, colls, pre_filter=None):
    t0 = datetime.datetime.fromtimestamp(0)
    
    for year in range(year_from, year_to + 1):
        if year == year_from:
            user_acty_df = get_user_activity_by_source(year, colls, pre_filter=pre_filter)
        else:
            user_acty_df = user_acty_df.join(
                get_user_activity_by_source(year, colls, pre_filter=pre_filter),
                how='outer',
                rsuffix='next',
            )
            user_acty_df['count'] = user_acty_df['count'].add(
                user_acty_df['countnext'], fill_value=0
            )
            user_acty_df['lastTime'] = np.maximum(
                user_acty_df['lastTime'].fillna(t0),
                user_acty_df['lastTimenext'].fillna(t0)
            )
            not_in_prev_year = user_acty_df['firstTime'].isnull()
            user_acty_df.loc[not_in_prev_year, 'firstTime'] = (
                user_acty_df.loc[not_in_prev_year, 'firstTimenext']
            )
            user_acty_df = user_acty_df.loc[:, ~user_acty_df.columns.str.endswith('next')]
    return user_acty_df


def get_persons(user_activity_df, max_hourly_rate=10) -> list:
    # done on all years
    sources = user_activity_df.index.levels[1]
    usource_mask = ~sources.str.contains(r'Twitter |Instagram|Tweetbot|Foursquare')
    source_mask = np.isin(user_activity_df.index.codes[1], np.nonzero(usource_mask)[0])
    nonpersons_ids = set(
        user_activity_df.loc[(slice(None), source_mask), :]
         .groupby('user_id')
         .indices
         .keys()
    )
    user_agg = user_activity_df.groupby('user_id').agg(
        count=('count', 'sum'),
        last_time=('lastTime', 'max'),
        first_time=('firstTime', 'min')
    )
    user_agg['dt'] = (user_agg['last_time'] - user_agg['first_time']).dt.seconds
    user_agg['rate'] = 0
    user_agg.loc[user_agg['dt'] > 0, 'rate'] = user_agg['count'] / user_agg['dt']
    spamming_mask = (
        (user_agg['rate'] > 10 / (60*60))
        & (user_agg['dt'] > 16 * 60 * 60)
    )
    spamming_ids = set(user_agg.index[spamming_mask])
    nonpersons_ids.update(spamming_ids)
    persons_ids = user_activity_df.index.levels[0].difference(nonpersons_ids)
    # Return as list as needed in Mongo filters
    return list(persons_ids)


def get_bot_ids(year, colls, pre_filter=None, max_hourly_rate=10):
    db = f'twitter_{year}'
    f = data_access.base_tweets_filter(year)
    if pre_filter is not None:
        f = f & pre_filter

    with qr.Connection(db) as con:
        pipeline = [
            {'$match': f},
            {
                "$group": {
                    "_id": "$user.id",
                    "count": {"$sum": 1 },
                    "lastTime": {"$last": "$created_at"},
                    "firstTime": {"$first": "$created_at"},
                }
            },
            {
                "$project": {
                    "count": 1,
                    "dt": {"$subtract": ["$lastTime", "$firstTime"]},
                }
            },
            {
                '$project': {
                    'dt': 1,
                    "rate": {
                        '$cond': {
                            'if': {'$eq': ["$dt", 0]},
                            'then': 0,
                            'else': {"$divide": ["$count", '$dt']}
                        }
                    }
                }
            },
            {
                '$match': {
                    'rate': {'$gt': max_hourly_rate / (60*60*1000)},
                    # Someone is awake ~16h per day, so let's consider this rate should
                    # have been sustained for longer than that to be considered a bot.
                    'dt': {'$gt': 16 * 60 * 60 * 1000}
            }}
        ]
        res = con[colls].aggregate(pipeline, allowDiskUse=True)
        bot_ids = [r['_id'] for r in res]

    return bot_ids
