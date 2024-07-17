import re
from dataclasses import InitVar, dataclass
from pathlib import Path
from string import Formatter

from dotenv import load_dotenv

import ses_ling

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


def get_params_fmt_str(*param_names):
    return '_'.join("{0}={{{0}}}".format(p) for p in param_names)


def get_params_str(**kwargs):
    return '_'.join(f"{p}={v}" for p, v in kwargs.items())


@dataclass
class ProjectPaths:
    '''
    Dataclass containing all the paths used throughout the project. Defining
    this class allows us to define these only once, ensuring consistency.
    '''
    proj: Path = Path(ses_ling.__file__).parent.parent
    countries_shapefile_name: str = 'CNTR_RG_01M_2016_4326'

    user_cells_from_gps_params: InitVar[list] = ['gps_attr_cell_size', 'gps_dups_th']
    user_cell_acty_params: InitVar[list] = ['res_cell_size', 'gps_dups_th']
    user_filter_params: InitVar[list] = ['user_nr_words_th']
    cell_assign_params: InitVar[list] = ['nighttime_acty_th', 'all_acty_th', 'count_th']
    cell_filter_params: InitVar[list] = ['cells_nr_users_th']
    generic_year_range_data_fname_fmt: str = '{name}_{year_from}-{year_to}_{params_fmt}.parquet'
    user_cells_from_gps_fname_fmt: str = partial_format(
        generic_year_range_data_fname_fmt,
        name='user_cells_from_gps',
        params_fmt=get_params_fmt_str(*user_cells_from_gps_params),
    )
    user_cell_acty_fname_fmt: str = partial_format(
        generic_year_range_data_fname_fmt,
        name='user_cell_acty',
        params_fmt=get_params_fmt_str(*user_cell_acty_params),
    )
    user_residence_cell_fname_fmt: str = partial_format(
        generic_year_range_data_fname_fmt,
        name='user_residence_cell',
        params_fmt=get_params_fmt_str(
            *user_cell_acty_params, 'pois_dups_th', *cell_assign_params
        ),
    )
    user_langs_fname_fmt: str = partial_format(
        generic_year_range_data_fname_fmt,
        name='user_langs',
        params_fmt=get_params_fmt_str(*cell_assign_params),
    )
    user_mistakes_fname_fmt: str = partial_format(
        generic_year_range_data_fname_fmt,
        name='user_mistakes',
        params_fmt=get_params_fmt_str(*cell_assign_params),
    )
    user_corpora_fname_fmt: str = partial_format(
        generic_year_range_data_fname_fmt,
        name='user_corpora',
        params_fmt=get_params_fmt_str(*cell_assign_params),
    )
    cells_mistakes_fname_fmt: str = partial_format(
        generic_year_range_data_fname_fmt,
        name='cells_mistakes',
        params_fmt=get_params_fmt_str(*user_filter_params, *cell_filter_params),
    )
    inter_cell_od_fname_fmt: str = partial_format(
        generic_year_range_data_fname_fmt,
        name='inter_cell_od_{subreg}',
        params_fmt=get_params_fmt_str(*user_filter_params, *cell_filter_params),
    )

    sim_data_fname_fmt: str = '{name}_{params_fmt}.parquet'
    sim_init_file_fmt: str = partial_format(
        sim_data_fname_fmt,
        name='{region}_{name}',
        params_fmt=get_params_fmt_str('cells_nr_users_th', 'nr_classes'),
    )
    sim_state_file_fmt: str = partial_format(
        sim_data_fname_fmt,
        name='{region}_{name}',
        params_fmt=get_params_fmt_str(
            'cells_nr_users_th', 'nr_classes', 's', 'q1', 'q2', 'step'
        ) + "{sim_id}",
    )

    counts_fname_fmt: str = (
        '{kind}_lang={lc}_cc={cc}_years={year_from}-{year_to}_'
        'cell_size={cell_size}.parquet'
    )
    map_fig_fname_fmt: str = (
        '{metric}_cell_sizes={cell_sizes}.pdf'
    )

    def __post_init__(
        self, user_cells_from_gps_params, user_cell_acty_params, cell_assign_params,
        user_filter_params, cell_filter_params
    ):
        self.proj_data = self.proj / 'data'
        self.ext_data = self.proj_data / 'external'
        self.raw_data = self.proj_data / 'raw'
        self.interim_data = self.proj_data / 'interim'
        self.processed_data = self.proj_data / 'processed'
        self.shp_file_fmt = self.ext_data / '{0}' / '{0}.shp'
        self.countries_shapefile = format_path(self.shp_file_fmt, self.countries_shapefile_name)
        self.countries_dict = self.ext_data / 'countries.json'
        self.figs = self.proj / 'reports' / 'figures'
        self.case_figs = self.figs / '{lc}_{str_cc}' / '{year_from}-{year_to}'
        self.resident_ids = (
            self.interim_data / '{cc}' / 'resident_ids_{year_from}-{year_to}.txt'
        )
        self.user_places = (
            self.interim_data / '{cc}' / 'user_places_{year_from}-{year_to}.parquet'
        )
        self.user_cells_from_gps = (
            self.interim_data / '{cc}'/ self.user_cells_from_gps_fname_fmt
        )
        self.user_cell_acty = self.interim_data / '{cc}' / self.user_cell_acty_fname_fmt
        self.user_residence_cell = (
            self.interim_data / '{cc}' / self.user_residence_cell_fname_fmt
        )
        self.user_langs = self.interim_data / '{cc}' / self.user_langs_fname_fmt
        self.user_mistakes = self.interim_data / '{cc}' / self.user_mistakes_fname_fmt
        self.user_corpora = self.interim_data / '{cc}' / self.user_corpora_fname_fmt
        self.chunk_user_mistakes = (
            self.user_mistakes.parent
            / (self.user_mistakes.stem + '_chunk={i_chunk}' + self.user_mistakes.suffix)
        )
        self.cells_mistakes = self.interim_data / '{str_cc}' / self.cells_mistakes_fname_fmt
        self.inter_cell_od = self.interim_data / '{str_cc}' / self.inter_cell_od_fname_fmt
        self.rule_category = self.interim_data / '{cc}' / 'rule_category.csv'
        self.counts_files = (
            self.proj.parent / 'words-use' / 'data' / 'raw' / '{lc}' / '{cc}'
            / self.counts_fname_fmt
        )
        self.language_tool_categories = self.ext_data / 'language_tool_categories_{lc}.json'
        self.sim_init_fmt = self.proj_data / 'simulations' / self.sim_init_file_fmt
        self.sim_state_fmt = self.proj_data / 'simulations' / self.sim_state_file_fmt

    def partial_format(self, **kwargs):
        for attr in (
            'case_figs', 'resident_ids', 'user_cells_from_gps', 'user_places',
            'user_cell_acty', 'user_residence_cell', 'user_langs', 'user_mistakes', 'user_corpora',
            'chunk_user_mistakes', 'rule_category', 'language_tool_categories',
            'counts_files', 'cells_mistakes', 'inter_cell_od'
        ):
            setattr(self, attr, partial_path_format(getattr(self, attr), **kwargs))
