import sys
import json
import logging
import logging.config
import geopandas as geopd
from dotenv import load_dotenv
load_dotenv()

import ses_ling.utils.paths as path_utils
import ses_ling.data.user_residence as user_residence

LOGGER = logging.getLogger(__name__)
LOG_CONFIG = json.load(open('logging.json', 'rt'))
logging.config.dictConfig(LOG_CONFIG)

if __name__ == '__main__':
    cc = sys.argv[1]
    year_from = int(sys.argv[2])
    year_to = int(sys.argv[3])

    paths = path_utils.ProjectPaths()
    with open(paths.ext_data / 'countries.json') as f:
        countries_dict = json.load(f)
    cc_dict = countries_dict[cc]
    paths.partial_format(cc=cc, year_from=year_from, year_to=year_to, cell_kind='LSOA_BGC', **cc_dict)
    colls = cc_dict['mongo_coll']

    save_path = paths.resident_ids
    with open(str(save_path).format(cc=cc, year_from=year_from, year_to=year_to), 'r') as f:
        resident_ids = f.read().splitlines()
    
    base_name = 'Lower_Layer_Super_Output_Areas_(December_2011)_Boundaries_Generalised_Clipped_(BGC)_EW_V3'
    simplified_lsoas = geopd.read_file(paths.ext_data / base_name / f'{base_name}.shp').set_index('LSOA11CD').rename_axis('lsoa_id')

    cell_counts = user_residence.agg_points_to_cells(
        year_from, year_to, colls, resident_ids, simplified_lsoas, timezone=cc_dict['timezone']
    )

    save_path = paths.user_cells_from_gps
    save_path.parent.mkdir(exist_ok=True, parents=True)
    cell_counts.to_parquet(save_path, index=True)
