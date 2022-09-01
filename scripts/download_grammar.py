import sys
import json

import ses_ling.utils.text_process as text_process
from ses_ling.utils.paths import ProjectPaths


if __name__ == '__main__':
    lc = sys.argv[1]
    lt_cats_dict = text_process.parse_online_grammar(lc)
    paths = ProjectPaths()
    paths.partial_format(lc=lc)
    with open(paths.language_tool_categories, 'w') as f:
        json.dump(lt_cats_dict, f)
