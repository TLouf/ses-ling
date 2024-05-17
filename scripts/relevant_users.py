import datetime
import json
import logging
import logging.config
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import querier as qr
import ray
from dotenv import load_dotenv

load_dotenv()

import ses_ling.data.access as data_access
import ses_ling.data.filter as data_filter
import ses_ling.utils.paths as path_utils

LOGGER = logging.getLogger(__name__)
LOG_CONFIG = json.load(open('logging.json', 'rt'))


@ray.remote
def get_uids_to_include(year_from, year_to, colls, max_hourly_rate=10):
    # Since Python logger module creates a singleton logger per process, loggers
    # should be configured on per task/actor basis.
    logging.config.dictConfig(LOG_CONFIG)
    LOGGER.info('** starting to find which users are persons **')
    user_acty_df = data_filter.agg_user_activity_by_source(
        year_from, year_to, colls, pre_filter=None
    )
    nr_users = len(user_acty_df.index.levels[0])
    uids_to_include = data_filter.get_persons(user_acty_df, max_hourly_rate=max_hourly_rate)
    nr_persons = len(uids_to_include)
    # LOGGER.info(f'{len(uids_to_exclude)} user IDs to exclude')
    LOGGER.info(f'{nr_persons} over {nr_users} user IDs to include.')
    return uids_to_include


@ray.remote
def get_residents(year_from, year_to, colls, pre_filter=None, nr_consec_months=3):
    logging.config.dictConfig(LOG_CONFIG)
    LOGGER.info('** starting to get residents **')
    user_months_df = data_filter.agg_consec_months(
        year_from, year_to, colls, pre_filter=pre_filter
    )
    resident_ids = data_filter.get_resident_ids(user_months_df, nr_consec_months=nr_consec_months)
    LOGGER.info(f'{len(resident_ids)} residents left')
    return resident_ids


def main(year_from, year_to, colls):
    # TODO paralellize by year?
    ray.init(num_cpus=2)
    uids_to_include = get_uids_to_include.remote(year_from, year_to, colls, max_hourly_rate=10)
    resident_ids = get_residents.remote(year_from, year_to, colls, nr_consec_months=3)
    resident_ids = set(ray.get(resident_ids)) & set(ray.get(uids_to_include))
    ray.shutdown()
    return resident_ids


if __name__ == '__main__':
    cc = sys.argv[1]
    year_from = int(sys.argv[2])
    year_to = int(sys.argv[3])

    paths = path_utils.ProjectPaths()
    with open(paths.countries_dict) as f:
        countries_dict = json.load(f)
    cc_dict = countries_dict[cc]
    paths.partial_format(cc=cc, year_from=year_from, year_to=year_to, **cc_dict)
    colls = cc_dict['mongo_coll']

    resident_ids = main(year_from, year_to, colls)

    save_path = paths.resident_ids
    save_path.parent.mkdir(exist_ok=True, parents=True)
    with open(save_path, 'w') as f:
        f.write('\n'.join(resident_ids))
