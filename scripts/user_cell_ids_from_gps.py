import sys
import json
import pickle
import logging
import logging.config
import geopandas as geopd
from dotenv import load_dotenv
load_dotenv()

import querier as qr
import ses_ling.utils.paths as path_utils
import ses_ling.data.user_residence as user_residence

LOGGER = logging.getLogger(__name__)
LOG_CONFIG = json.load(open('logging.json', 'rt'))
logging.config.dictConfig(LOG_CONFIG)

if __name__ == '__main__':
    cc = sys.argv[1]
    year_from = int(sys.argv[2])
    year_to = int(sys.argv[3])

    gps_dups_th = 100

    paths = path_utils.ProjectPaths()
    with open(paths.countries_dict) as f:
        countries_dict = json.load(f)
    cc_dict = countries_dict[cc]
    colls = cc_dict['mongo_coll']

    paths.partial_format(
        cc=cc, year_from=year_from, year_to=year_to,
        cell_kind='LSOA_BGC', gps_dups_th=gps_dups_th,
        **cc_dict
    )
    LOGGER.info(f"Will save to {paths.user_cells_from_gps}")

    save_path = paths.resident_ids
    with open(str(save_path).format(cc=cc, year_from=year_from, year_to=year_to), 'r') as f:
        resident_ids = f.read().splitlines()
    
    base_name = 'Lower_Layer_Super_Output_Areas_(December_2011)_Boundaries_Generalised_Clipped_(BGC)_EW_V3'
    simplified_lsoas = geopd.read_file(paths.ext_data / base_name / f'{base_name}.shp').set_index('LSOA11CD')

    gps_dups_counts_path = paths.interim_data / cc / 'gps_dups_counts.parquet'
    gps_filter = user_residence.filter_gps_duplicates(
        year_from, year_to, colls, gps_dups_counts_path, th=gps_dups_th
    )

    cell_counts = user_residence.agg_points_to_cells(
        year_from, year_to, colls, resident_ids, simplified_lsoas, timezone=cc_dict['timezone'],
        hour_start_day=8, hour_end_day=18, pre_filter=gps_filter,
    )

    save_path = paths.user_cells_from_gps
    save_path.parent.mkdir(exist_ok=True, parents=True)
    cell_counts.to_parquet(save_path, index=True)
