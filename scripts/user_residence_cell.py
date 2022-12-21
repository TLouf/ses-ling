import sys
import json
import logging
import logging.config
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import querier as qr

import ses_ling.data.access as data_access
import ses_ling.utils.paths as path_utils
import ses_ling.utils.spatial_agg as spatial_agg
import ses_ling.data.user_residence as user_residence
from ses_ling.language import Region

LOGGER = logging.getLogger(__name__)
LOG_CONFIG = json.load(open('logging.json', 'rt'))
logging.config.dictConfig(LOG_CONFIG)


def main(reg: Region, gps_attr_level, pois_dups_th=-1, gps_dups_th=-1, **assign_kwargs):
    return user_residence.whole_cell_attr(
        reg, gps_attr_level, pois_dups_th=pois_dups_th, gps_dups_th=gps_dups_th,
        **assign_kwargs
    )

if __name__ == '__main__':
    cc = sys.argv[1]
    year_from = int(sys.argv[2])
    year_to = int(sys.argv[3])

    res_cell_size = 'MSOA_BGC'
    gps_attr_level = 'LSOA_BGC'
    gps_dups_th = 500
    pois_dups_th = 500

    paths = path_utils.ProjectPaths()
    with open(paths.countries_dict) as f:
        countries_dict = json.load(f)
    cc_dict = countries_dict[cc]
    res_attr_kwargs = dict(
        nighttime_acty_th=0.5,
        all_acty_th=0.1,
        count_th=3,
        gps_dups_th=gps_dups_th,
        pois_dups_th=pois_dups_th,
    )

    reg = Region(
        cc, 'en', cc_dict, res_cell_size, cell_size=res_cell_size,
        year_from=year_from, year_to=year_to,
        res_attr_kwargs=res_attr_kwargs,
    )
    LOGGER.info(f"Will save to {reg.paths.user_residence_cell}")
    user_residence_cell = main(reg, gps_attr_level, **res_attr_kwargs)
    user_residence_cell.to_parquet(reg.paths.user_residence_cell
    )
