import sys
import json
import logging
import logging.config
import querier as qr
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
    with open(paths.countries_dict) as f:
        countries_dict = json.load(f)
    cc_dict = countries_dict[cc]
    paths.partial_format(cc=cc, year_from=year_from, year_to=year_to, **cc_dict)
    colls = cc_dict['mongo_coll']

    save_path = paths.resident_ids
    with open(str(save_path).format(cc=cc, year_from=year_from, year_to=year_to), 'r') as f:
        resident_ids = f.read().splitlines()

    pre_filter = qr.Filter()
    visited_places = user_residence.agg_visited_places(
        year_from, year_to, colls, resident_ids, timezone=cc_dict['timezone']
    )

    save_path = paths.user_places
    save_path.parent.mkdir(exist_ok=True, parents=True)
    visited_places.to_parquet(save_path, index=True)
