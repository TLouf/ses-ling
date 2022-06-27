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


def partial_format(fmt_str, **kwargs):
    all_kw = list(yield_kw_from_fmt_str(fmt_str))
    fmt_dict = {**{kw: f'{{{kw}}}' for kw in all_kw}, **kwargs}
    return fmt_str.format(**fmt_dict)


def format_path(path_fmt, **kwargs):
    '''
    Utility to apply string formatting to a Path.
    '''
    return Path(str(path_fmt).format(**kwargs))


@dataclass
class ProjectPaths:
    '''
    Dataclass containing all the paths used throughout the project. Defining
    this class allows us to define these only once, ensuring consistency.
    '''
    proj: Path = Path(os.environ['PROJ_DIR'])
    # source_fname_fmt: str = '{kind}_{from}_{to}_{cc}.json.gz'
    proj_data: Path = field(init=False)
    ext_data: Path = field(init=False)
    raw_data: Path = field(init=False)
    interim_data: Path = field(init=False)
    processed_data: Path = field(init=False)
    counts_files_fmt: Path = field(init=False)
    shp_file_fmt: Path = field(init=False)
    figs: Path = field(init=False)
    cluster_fig_fmt: Path = field(init=False)

    def __post_init__(self):
        # self.source_fmt = self.source_data / self.source_fname_fmt
        self.proj_data = self.proj / 'data'
        self.ext_data = self.proj_data / 'external'
        self.raw_data = self.proj_data / 'raw'
        self.interim_data = self.proj_data / 'interim'
        self.processed_data = self.proj_data / 'processed'
        self.shp_file_fmt = self.ext_data / '{0}.shp' / '{0}.shp'
        self.figs = self.proj / 'reports' / 'figures'
        self.case_figs = self.figs / '{lc}' / '{cc}' / '{year_from}-{year_to}'
        self.resident_ids = self.interim_data / '{cc}' / 'resident_ids_{year_from}-{year_to}.txt'


    def partial_format(self, **kwargs):
        self.case_figs = Path(partial_format(str(self.case_figs), **kwargs))
        self.resident_ids = Path(partial_format(str(self.resident_ids), **kwargs))
