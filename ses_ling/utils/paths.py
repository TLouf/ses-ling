import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from string import Formatter
from dotenv import load_dotenv
load_dotenv()


def yield_kw_from_fmt_str(fmt_str):
    '''
    Yield all keywords from a format string.
    '''
    for _, fn, _, _ in Formatter().parse(fmt_str):
        if fn is not None:
            yield fn


def yield_paramed_matches(file_path_format, params_dict):
    '''
    Generator of named matches of the parameterised format `file_path_format`
    for every parameter not fixed in `params_dict`.
    '''
    fname = file_path_format.name
    # Create a dictionary with keys corresponding to each parameter of the file
    # format, and the values being either the one found in `params_dict`, or a
    # regex named capture group of the parameter.
    pformat = {fn: params_dict.get(fn, f'(?P<{fn}>.+)')
               for fn in yield_kw_from_fmt_str(fname)}
    file_pattern = re.compile(fname.replace('.', r'\.').format(**pformat))
    for f in file_path_format.parent.iterdir():
        match = re.search(file_pattern, f.name)
        if match is not None:
            yield match


def partial_format(fmt_str, *args, **kwargs):
    all_kw = list(yield_kw_from_fmt_str(fmt_str))
    fmt_dict = {**{kw: f'{{{kw}}}' for kw in all_kw}, **kwargs}
    return fmt_str.format(*args, **fmt_dict)


def partial_path_format(path_fmt, *args, **kwargs):
    return Path(partial_format(str(path_fmt), *args, **kwargs))


def format_path(path_fmt, *args, **kwargs):
    '''
    Utility to apply string formatting to a Path.
    '''
    return Path(str(path_fmt).format(*args, **kwargs))


@dataclass
class ProjectPaths:
    '''
    Dataclass containing all the paths used throughout the project. Defining
    this class allows us to define these only once, ensuring consistency.
    '''
    proj: Path = Path(os.environ['PROJ_DIR'])
    countries_shapefile_name: str = 'CNTR_RG_01M_2016_4326'
    user_residence_cell_fname_fmt: str = (
        'user_residence_cell_{year_from}-{year_to}_'
        'nighttime_acty_th={nighttime_acty_th}_all_acty_th={all_acty_th}_'
        'count_th={count_th}_cell_size={res_attrib_level}.parquet'
    )
    user_mistakes_fname_fmt: str = (
        'user_mistakes_{year_from}-{year_to}_nighttime_acty_th={nighttime_acty_th}_'
        'all_acty_th={all_acty_th}_count_th={count_th}.parquet'
    )
    user_corpora_fname_fmt: str = (
        'user_corpora_{year_from}-{year_to}_nighttime_acty_th={nighttime_acty_th}_'
        'all_acty_th={all_acty_th}_count_th={count_th}.parquet'
    )
    counts_fname_fmt: str = (
        '{kind}_lang={lc}_cc={cc}_years={year_from}-{year_to}_'
        'cell_size={cell_size}.parquet'
    )
    map_fig_fname_fmt: str = (
        '{metric}_cell_sizes={cell_sizes}.pdf'
    )

    def __post_init__(self):
        self.proj_data = self.proj / 'data'
        self.ext_data = self.proj_data / 'external'
        self.raw_data = self.proj_data / 'raw'
        self.interim_data = self.proj_data / 'interim'
        self.processed_data = self.proj_data / 'processed'
        self.shp_file_fmt = self.ext_data / '{0}' / '{0}.shp'
        self.countries_shapefile = format_path(self.shp_file_fmt, self.countries_shapefile_name)
        self.countries_dict = self.ext_data / 'countries.json'
        self.figs = self.proj / 'reports' / 'figures'
        self.case_figs = self.figs / '{lc}' / '{str_cc}' / '{year_from}-{year_to}'
        self.resident_ids = (
            self.interim_data / '{cc}' / 'resident_ids_{year_from}-{year_to}.txt'
        )
        self.user_cells_from_gps = (
            self.interim_data
            / '{cc}'
            / 'user_cells_from_gps_{cell_kind}_{year_from}-{year_to}.parquet'
        )
        self.user_places = (
            self.interim_data / '{cc}' / 'user_places_{year_from}-{year_to}.parquet'
        )
        self.user_residence_cell = (
            self.interim_data / '{cc}' / self.user_residence_cell_fname_fmt
        )
        self.user_mistakes = self.interim_data / '{cc}' / self.user_mistakes_fname_fmt
        self.user_corpora = self.interim_data / '{cc}' / self.user_corpora_fname_fmt
        self.chunk_user_mistakes = (
            self.user_mistakes.parent
            / (self.user_mistakes.stem + '_chunk={i_chunk}' + self.user_mistakes.suffix)
        )
        self.rule_category = self.interim_data / '{cc}' / 'rule_category.csv'
        self.counts_files = (
            self.proj.parent / 'words-use' / 'data' / 'raw' / '{lc}' / '{cc}'
            / self.counts_fname_fmt
        )
        self.language_tool_categories = self.ext_data / 'language_tool_categories_{lc}.json'


    def partial_format(self, **kwargs):
        for attr in (
            'case_figs', 'resident_ids', 'user_cells_from_gps', 'user_places',
            'user_residence_cell', 'user_mistakes', 'user_corpora', 
            'chunk_user_mistakes', 'rule_category', 'language_tool_categories',
            'counts_files', 
        ):
            setattr(self, attr, partial_path_format(getattr(self, attr), **kwargs))
