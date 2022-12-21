from __future__ import annotations

import json
import inspect
from dataclasses import dataclass, field, InitVar, asdict, _FIELD

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as geopd

import ses_ling.data.socioeconomic as ses_data
import ses_ling.utils.paths as paths_utils
import ses_ling.utils.geometry as geo_utils
import ses_ling.utils.spatial_agg as spatial_agg
import ses_ling.utils.text_process as text_process
import ses_ling.visualization.maps as map_viz

@dataclass
class Region:
    '''
    Heavy data loading done in this class, and raw data kept here. Will be assembled and
    processed, thus potentially modified in Language as per the needs of the analysis.
    '''
    cc: str
    lc: str
    # Keep the whole init_dict from countries.json in case after init we want to select
    # other options.
    # TODO: make init_dict a dataclass? since some params are needed
    init_dict: dict
    res_cell_size: str
    all_cntr_shapes: InitVar[geopd.GeoDataFrame | None] = None
    # not a problem to default with mutable because this initvar is never touched
    extract_shape_kwargs: InitVar[dict] = {'simplify_tol': 100}
    ses_idx: str | None = None
    year_from: int = 2015
    year_to: int = 2021
    xy_proj: str = 'epsg:3857'
    max_place_area: float = 5e9
    cell_size: InitVar[int | float | str] = 50e3
    cell_kind: str = 'census'
    # Params
    # TODO: make res_attr_kwargs a dataclass?
    res_attr_kwargs: dict = field(default_factory=dict)
    shape_geodf: geopd.GeoDataFrame | None = None
    cell_levels_corr: pd.DataFrame | None = None
    weighted_cell_levels_corr: pd.DataFrame | None = None
    ses_df: pd.DataFrame | None = None
    lt_rules: pd.DataFrame | None = None
    _shape_bbox: list | None = None
    _cells_geodf: geopd.GeoDataFrame | None = None
    _paths: paths_utils.ProjectPaths | None = None
    _user_residence_cell: pd.DataFrame | None = None
    _user_mistakes: pd.DataFrame | None = None
    _user_corpora: pd.DataFrame | None = None

    def __post_init__(
        self, all_cntr_shapes, extract_shape_kwargs, cell_size
    ):
        self._cell_size = cell_size
        self.cell_kind = self.cell_shapefiles[cell_size]['kind'] or self.cell_kind
        if self.shape_geodf is None:
            if all_cntr_shapes is None:
                all_cntr_shapes = geopd.read_file(self.paths.countries_shapefile)
            shapefile_val = self.init_dict.get("shapefile_val", self.cc)
            shapefile_col = self.init_dict.get("shapefile_col", "FID")
            mask = all_cntr_shapes[shapefile_col].str.startswith(shapefile_val)
            self.shape_geodf = geo_utils.extract_shape(
                all_cntr_shapes.loc[mask], self.cc, xy_proj=self.xy_proj,
                bbox=self.init_dict.get("shape_bbox"), **extract_shape_kwargs
            )
        print(f'shape_geodf loaded for {self.cc}')

        if self.ses_idx is None:
            self.ses_idx = list(self.ses_data_options.keys())[0]

        if self.ses_df is None:
            self.ses_df = self.load_ses_df()
        print(f'ses_df loaded for {self.cc}')

        if self.cell_levels_corr is None:
            self.cell_levels_corr = self.load_cell_levels_corr()
            self.weighted_cell_levels_corr = self.get_weighted_cell_levels_corr()
        print(f'cell_levels_corr loaded for {self.cc}')

        if self.lt_rules is None:
            lt_rules_path = self.paths.rule_category
            if lt_rules_path.exists():
                self.lt_rules = (
                    pd.read_csv(lt_rules_path)
                     .rename(columns={'ruleId': 'rule_id', 'category': 'cat_id'})
                     .groupby('rule_id')
                     .first()
                )
        print(f'lt_rules loaded for {self.cc}')

    def __repr__(self):
        field_dict = self.__dataclass_fields__
        persistent_field_keys = [
            key
            for key, value in field_dict.items()
            if value._field_type == _FIELD
        ]
        attr_str_components = []
        for key in persistent_field_keys:
            field = getattr(self, key)
            field_repr = repr(field)
            if len(field_repr) < 200:
                attr_str_components.append(f'{key}={field_repr}')

        attr_str = ', '.join(attr_str_components)
        return f'{self.__class__.__name__}({attr_str})'


    @classmethod
    def from_dict(cls, cc, lc, init_dict, res_cell_size, **kwargs):
        all_kwargs = {**init_dict, **kwargs}
        matching_kwargs = {
            k: v for k, v in all_kwargs.items()
            if k in inspect.signature(cls).parameters
        }
        return cls(cc, lc, init_dict, res_cell_size, **matching_kwargs)


    def update_from_dict(self, d):
        # useful when changes are made to countries.json
        for key, value in d.items():
            if self.hasattr(key):
                setattr(self, key, value)


    def to_dict(self):
        # custom to_dict to keep only parameters that can be in save path
        list_attr = [
            'lc', 'readable', 'cc', 'year_from', 'year_to', 'cell_size', 'res_cell_size',
            'max_place_area', 'xy_proj',
        ]
        return {
            **{attr: getattr(self, attr) for attr in list_attr}, **self.res_attr_kwargs
        }

    @property
    def bbox(self):
        if self._shape_bbox is None:
            if "shape_bbox" in self.init_dict:
                self._shape_bbox = self.init_dict["shape_bbox"]
            else:
                self._shape_bbox = (
                    self.shape_geodf.geometry.to_crs('epsg:4326').total_bounds
                )
        return self._shape_bbox

    @property
    def paths(self):
        self._paths = paths_utils.ProjectPaths()
        self._paths.partial_format(**self.to_dict())
        return self._paths

    @paths.setter
    def paths(self, p):
        self._paths = p

    @paths.deleter
    def paths(self):
        self._paths = None

    @property
    def ses_data_options(self):
        return self.init_dict['ses_data_options']

    @property
    def cell_shapefiles(self):
        return self.init_dict['cell_shapefiles']

    @property
    def cell_levels_corr_files(self):
        return self.init_dict['cell_levels_corr_files']

    @property
    def readable(self):
        return self.init_dict['readable']

    @property
    def mongo_coll(self):
        return self.init_dict['mongo_coll']

    @property
    def cell_size(self):
        return self._cell_size

    @cell_size.setter
    def cell_size(self, _cell_size):
        if _cell_size != self.cell_size:
            self._cell_size = _cell_size
            cell_size_spec = self.cell_shapefiles.get(self._cell_size)
            if cell_size_spec is None:
                if self._cells_geodf is None:
                # correspondence with `cell_levels_corr`
                    raise ValueError(
                        f"cell_size of {self.cell_size} can neither be obtained by "
                        "direct reading of a shapefile or correspondence with a "
                        "previous aggregation level: read a saved one first"
                    )
                agg_level = self._cell_size
                unit_level = self.cells_geodf.index.name
                cell_levels_corr = spatial_agg.levels_corr(
                    self.cell_levels_corr, unit_level, agg_level
                )
                agg_level = cell_levels_corr.index.names[0]
                self.cells_geodf = (
                    self.cells_geodf.join(cell_levels_corr)
                        .dissolve(by=agg_level)
                )
            else:
                self.cell_kind = cell_size_spec["kind"]
                # Else load according to cell_size_spec using cells_geodf's setter
                del self.cells_geodf
                self.cells_geodf = self.cells_geodf

            if _cell_size != self.res_cell_size and _cell_size in self.cell_shapefiles.keys():
                del self.user_residence_cell

            self.weighted_cell_levels_corr = self.get_weighted_cell_levels_corr()

    @property
    def cells_geodf(self):
        if self._cells_geodf is None:
            if isinstance(self.cell_size, str):
                cell_size_spec = self.cell_shapefiles.get(self.cell_size)
                if cell_size_spec is None:
                    raise ValueError(
                        f"cell_size of {self.cell_size} can neither be obtained by "
                        "direct reading of a shapefile or correspondence with a "
                        "previous aggregation level: read a saved one first"
                    )
                fname = cell_size_spec['fname']
                index_col = cell_size_spec['index_col']
                self._cells_geodf = geo_utils.load_ext_cells(
                    fname, index_col, xy_proj=self.xy_proj
                )
            else:
                _, self._cells_geodf, _, _ = geo_utils.create_grid(
                    self.shape_geodf, self.cell_size, self.cc,
                    xy_proj=self.xy_proj, intersect=True
                )
        return self._cells_geodf

    @cells_geodf.setter
    def cells_geodf(self, _cells_geodf):
        self._cells_geodf = _cells_geodf

    @cells_geodf.deleter
    def cells_geodf(self):
        self.cells_geodf = None


    def load_ses_df(self, ses_idx=None):
        # Cannot have multiple indices in `ses_df` here because they may be on different
        # cell levels. Better to aggregate them in Language.
        if ses_idx is not None:
            self.ses_idx = ses_idx
        self.ses_df = ses_data.read_df(
            self.paths.ext_data, **self.ses_data_options[self.ses_idx]
        )
        return self.ses_df


    def load_cell_levels_corr(self):
        cell_shapefile_kind = self.cell_shapefiles[self.cell_size]["kind"]
        corr_path = (
            self.paths.ext_data
            / self.cell_levels_corr_files[cell_shapefile_kind]
        )
        self.cell_levels_corr = pd.read_csv(corr_path)
        return self.cell_levels_corr

    def get_weighted_cell_levels_corr(self):
        # for ses_df exclusively, to convert cells_geodf no need for weights
        agg_level = self.cells_geodf.index.name
        unit_level = self.ses_df.index.name
        cell_levels_corr = spatial_agg.levels_corr(
            self.cell_levels_corr, unit_level, agg_level
        )
        # cell_levels_corr = self.load_agg_cell_levels_corr_for(self.ses_df)
        opt = self.ses_data_options[self.ses_idx]
        if 'weight_col' in opt:
            weight_df = self.ses_df
            weight_col = opt['weight_col']
        else:
            weight_opt = self.ses_data_options[opt['weight_from']]
            weight_col = weight_opt['weight_col']
            weight_df = ses_data.read_df(
                self.paths.ext_data, **weight_opt
            )

        # By weight_df construction, weight_col is first column in both cases above.
        weight_col = weight_df.columns[0]

        self.weighted_cell_levels_corr = spatial_agg.weight_levels_corr(
            cell_levels_corr, weight_df, agg_level, weight_col=weight_col
        )
        return self.weighted_cell_levels_corr


    def make_subregions_mask(self, subreg_df):
        cell_shapefile_kind = self.cell_kind
        corr_df = pd.read_csv(
            self.paths.ext_data / self.cell_levels_corr_files[cell_shapefile_kind]
        )
        corr_df.columns = corr_df.columns.str.lower()
        isin_area = pd.Series(False, name='isin_area', index=self.cells_geodf.index)
        for t in subreg_df.itertuples():
            r = t.Index
            # TODO: when col not in corr_df, read cell_levels_corr_files[col]
            idc_region = pd.Index(corr_df.loc[corr_df[t.agg_col] == r, self.cells_geodf.index.name]).unique()
            isin_area.loc[idc_region] = True
            print(f"{idc_region.size} cells in {r}.")
        return isin_area


    @property
    def user_residence_cell(self):
        if self._user_residence_cell is None:
            p = self.paths.user_residence_cell
            if p.exists():
                self._user_residence_cell = pd.read_parquet(p)
            else:
                raise ValueError(
                    f'Please run the script to generate {p}'
                )

        if self._user_residence_cell.columns[0].lower() != self.cells_geodf.index.name.lower():
            unit_level = (
                self.cell_shapefiles.get(self.res_cell_size, {}).get("index_col")
                or self.res_cell_size
            )
            agg_level = (
                self.cell_shapefiles.get(self.cell_size, {}).get("index_col")
                or self.cell_size
            )
            cell_levels_corr = spatial_agg.levels_corr(
                self.cell_levels_corr, unit_level, agg_level
            )
            cell_levels = cell_levels_corr.index.names
            self._user_residence_cell = (
                self._user_residence_cell.set_index(cell_levels[1], append=True)
                 .join(cell_levels_corr)
                 .droplevel(cell_levels[1])
                 .reset_index(cell_levels[0])
            )
        return self._user_residence_cell

    @user_residence_cell.setter
    def user_residence_cell(self, _user_residence_cell):
        self._user_residence_cell = _user_residence_cell
        
    @user_residence_cell.deleter
    def user_residence_cell(self):
        self._user_residence_cell = None


    @property
    def user_mistakes(self):
        if self._user_mistakes is None:
            self.read_corpora_analysis_res()
        return self._user_mistakes

    @property
    def user_corpora(self):
        if self._user_corpora is None:
            self.read_corpora_analysis_res()
        return self._user_corpora

    def read_corpora_analysis_res(self):
        if self.paths.user_mistakes.exists() and self.paths.user_corpora.exists():
            user_mistakes = pd.read_parquet(self.paths.user_mistakes)
            user_corpora = pd.read_parquet(self.paths.user_corpora)
        else:
            chunk_path_fmt = self.paths.chunk_user_mistakes
            corpora_cols = ['nr_tweets', 'nr_words', 'nr_unique_words']
            user_mistakes = pd.DataFrame()
            user_corpora = pd.DataFrame()
            p_matches = chunk_path_fmt.parent.glob(
                str(chunk_path_fmt.stem).format(i_chunk='*')
            )
            for p in tqdm(p_matches):
                chunk_user_mistakes = pd.read_parquet(p)
                rules_cols = chunk_user_mistakes.columns[~chunk_user_mistakes.columns.isin(corpora_cols)]
                chunk_user_corpora = chunk_user_mistakes[corpora_cols]
                chunk_user_mistakes = (
                    chunk_user_mistakes[rules_cols].stack()
                     .rename_axis(['user_id', 'rule_id'])
                     .rename('count')
                     .astype(int)
                     .to_frame()
                )
                user_corpora = pd.concat([user_corpora, chunk_user_corpora])
                user_mistakes = pd.concat([user_mistakes, chunk_user_mistakes])

            if user_mistakes.empty:
                raise ValueError(
                    'Please run the script to generate `user_mistakes`'
                )
            user_mistakes.to_parquet(self.paths.user_mistakes, index=True)
            user_corpora.to_parquet(self.paths.user_corpora, index=True)

        self._user_corpora = user_corpora.astype(int)
        self._user_mistakes = user_mistakes


@dataclass
class Language:
    lc: str
    readable: str
    # Arguable solution: could also just edit countries_dict in script: pb lose default
    # for subsequent runs... This is more heavy but more robust.
    _cc_init_params: InitVar[dict]
    countries_dict: InitVar[dict]
    all_cntr_shapes: InitVar[geopd.GeoDataFrame | None] = None
    regions: list[Region] = field(default_factory=list)
    year_from: int = 2015
    year_to: int = 2021
    month_from: int = 1
    month_to: int = 12
    latlon_proj: str = 'epsg:4326'
    res_attr_kwargs: InitVar[dict] = field(default_factory=dict)
    # Parameters for word and cell filters:
    user_nr_words_th: InitVar[int] = 0
    cells_nr_users_th: InitVar[int] = 0
    # Data containers (frames, arrays)
    lt_rules: pd.DataFrame = field(init=False)
    lt_categories: pd.DataFrame = field(init=False)
    _cells_geodf: geopd.GeoDataFrame | None = None
    _user_residence_cell: pd.DataFrame | None = None
    _user_corpora: pd.DataFrame | None = None
    _user_df: pd.DataFrame | None = None
    _user_mistakes: pd.DataFrame | None = None
    _cells_ses_df: pd.DataFrame | None = None
    _cells_users_df: pd.DataFrame | None = None
    _cells_mistakes: pd.DataFrame | None = None
    _width_ratios: np.ndarray | None = None

    def __post_init__(
        self, _cc_init_params, countries_dict, all_cntr_shapes, res_attr_kwargs, user_nr_words_th, cells_nr_users_th
    ):
        self._res_attr_kwargs = res_attr_kwargs
        self._user_nr_words_th = user_nr_words_th
        self._cells_nr_users_th = cells_nr_users_th
        if len(self.regions) < len(_cc_init_params):
            missing_ccs = set(_cc_init_params.keys()).difference({reg.cc for reg in self.regions})

            if all_cntr_shapes is None:
                all_cntr_shapes = geopd.read_file(self.paths.countries_shapefile)

            for cc in missing_ccs:
                self.regions.append(
                    Region.from_dict(
                        cc=cc, lc=self.lc, all_cntr_shapes=all_cntr_shapes,
                        year_from=self.year_from, year_to=self.year_to,
                        res_attr_kwargs=self.res_attr_kwargs,
                        init_dict=countries_dict[cc],
                        **{**countries_dict[cc], **_cc_init_params[cc]}
                    )
                )
        print('regions done')
        # sort a posteriori
        _, self.regions = (
            list(x)
            for x in zip(*sorted(zip(self.list_cc, self.regions), key=lambda t: t[0]))
        )

        if self.paths.language_tool_categories.exists():
            with open(self.paths.language_tool_categories) as f:
                lt_cats_dict = json.load(f)
        else:
            lt_cats_dict = text_process.parse_online_grammar(self.lc)
            with open(self.paths.language_tool_categories, 'w') as f:
                json.dump(lt_cats_dict, f)

        self.lt_categories = text_process.get_lt_categories(lt_cats_dict)
        self.lt_rules = self.get_lt_rules(lt_cats_dict)
        print('done')

    def __repr__(self):
        field_dict = self.__dataclass_fields__
        persistent_field_keys = [
            key
            for key, value in field_dict.items()
            if value._field_type == _FIELD
        ]
        attr_str_components = []
        for key in persistent_field_keys:
            field = getattr(self, key)
            field_repr = repr(field)
            if len(field_repr) < 500:
                attr_str_components.append(f'{key}={field_repr}')

        attr_str = ', '.join(attr_str_components)
        return f'{self.__class__.__name__}({attr_str})'


    @classmethod
    def from_other(cls, other: Language):
        field_dict = other.__dataclass_fields__
        init_field_keys = [
            key
            for key, value in field_dict.items()
            if value.init and not isinstance(value.type, InitVar)
        ]
        other_attrs = {
            key: getattr(other, key, field_dict[key].default)
            for key in init_field_keys
        }
        return cls(**other_attrs)


    @classmethod
    def from_countries_dict(cls, lc, readable, list_cc_, countries_dict,
                            all_cntr_shapes=None, year_from=2015,
                            year_to=2021, **kwargs):
        # regions = [
        #     Region.from_dict(cc, lc, countries_dict[cc], year_from=year_from,
        #                      year_to=year_to, month_from=month_from, month_to=month_to)
        #     for cc in list_cc
        # ]
        return cls(
            lc, readable, list_cc_, countries_dict, all_cntr_shapes=all_cntr_shapes,
            year_from=year_from, year_to=year_to, **kwargs
        )

    def update_regions_from_dict(self, d):
        # useful when changes are made to countries.json
        for r in self.regions:
            r.update_from_dict(d.get(r.cc, {}))


    def to_dict(self):
        # custom to_dict to keep only parameters that can be in save path
        list_attr = [
            'lc', 'readable', 'str_cc', 'year_from', 'year_to',
            'user_nr_words_th', 'cells_nr_users_th'
        ]
        return {
            **{attr: getattr(self, attr) for attr in list_attr}, **self.res_attr_kwargs
        }


    @property
    def list_cc(self):
        return [reg.cc for reg in self.regions]

    @property
    def str_cc(self):
        return '-'.join(self.list_cc)

    @property
    def regions_dict(self):
        return {reg.cc: reg for reg in self.regions}

    @property
    def figs_path(self):
        d = self.to_dict()
        for k in ('lc', 'readable', 'str_cc', 'year_from', 'year_to'):
            d.pop(k)
        params_str = "_".join(f"{k}={v}" for k, v in d.items())
        figs_path = (
            self.paths.figs
            / f'{self.lc}_{self.str_cc}'
            / f'{self.year_from}-{self.year_to}'
            / params_str
        )
        figs_path.mkdir(exist_ok=True, parents=True)
        return figs_path

    @property
    def paths(self):
        self._paths = paths_utils.ProjectPaths()
        self._paths.partial_format(**self.to_dict())
        return self._paths

    @paths.setter
    def paths(self, p):
        self._paths = p


    def reload_countries_dict(self):
        with open(self.paths.countries_dict) as f:
            countries_dict = json.load(f)
        for r in self.regions:
            r.init_dict = countries_dict.get(r.cc, {})


    def change_cell_sizes(self, **cc_cell_size_dict):
        for r in self.regions:
            if r.cc in cc_cell_size_dict.keys():
                r.cell_size = cc_cell_size_dict[r.cc]
                del self.cells_geodf

    @property
    def cells_geodf(self):
        if self._cells_geodf is None:
            self._cells_geodf = pd.concat([
                reg.cells_geodf.to_crs(self.latlon_proj)
                for reg in self.regions
            ]).sort_index()
        return self._cells_geodf

    @cells_geodf.setter
    def cells_geodf(self, _cells_geodf):
        self._cells_geodf = _cells_geodf
        del self.user_residence_cell

    @cells_geodf.deleter
    def cells_geodf(self):
        self.cells_geodf = None


    @property
    def res_attr_kwargs(self):
        return self._res_attr_kwargs

    @res_attr_kwargs.setter
    def res_attr_kwargs(self, d):
        self._res_attr_kwargs = d
        for r in self.regions:
            r.res_attr_kwargs = d
        del self.user_residence_cell


    @property
    def user_residence_cell(self):
        # TODO: when several regions, handle duplicate user_id (should be rare though
        # given residence requirement)
        if self._user_residence_cell is None:
            # Rename with r.cells_geodf.index.name to raise if mismatch with
            # r.user_residence_cell.columns[0], which should not happen.
            self._user_residence_cell = pd.concat([
                r.user_residence_cell.rename(columns={r.cells_geodf.index.name: 'cell_id'})
                for r in self.regions
            ])
        return self._user_residence_cell

    @user_residence_cell.setter
    def user_residence_cell(self, _user_residence_cell):
        self._user_residence_cell = _user_residence_cell
        del self.user_df

    @user_residence_cell.deleter
    def user_residence_cell(self):
        self.user_residence_cell = None
        for r in self.regions:
            del r.user_residence_cell
  
        
    @property
    def user_corpora(self):
        if self._user_corpora is None:
            self._user_corpora = pd.concat([
                r.user_corpora for r in self.regions
            ])
        return self._user_corpora


    @property
    def user_df(self):
        if self._user_df is None:
            self._user_df = self.user_corpora.join(self.user_residence_cell['cell_id'])
        return self._user_df

    @user_df.setter
    def user_df(self, df):
        self._user_df = df
        del self.cells_users_df
        del self.cells_ses_df

    @user_df.deleter
    def user_df(self):
        self.user_df = None

    @property
    def user_nr_words_th(self):
        return self._user_nr_words_th

    @user_nr_words_th.setter
    def user_nr_words_th(self, th):
        del self.user_mask
        self._user_nr_words_th = th

    @property
    def user_mask(self):
        if 'is_relevant' not in self.user_df.columns:
            nr_words_mask = self.user_df['nr_words'] >= self.user_nr_words_th
            user_mask = nr_words_mask & self.user_df['cell_id'].notnull()
            self.user_df['is_relevant'] = user_mask
            print(f'Keeping {user_mask.sum()} users out of {self.user_df.shape[0]}')
        return self.user_df['is_relevant']

    @user_mask.deleter
    def user_mask(self):
        if self._user_df is not None:
            self.user_df = self.user_df.drop(columns='is_relevant', errors='ignore')


    @property
    def user_mistakes(self):
        if self._user_mistakes is None:
            user_mistakes = (
                pd.concat([r.user_mistakes for r in self.regions])
                 .join(self.lt_rules.set_index('cat_id', append=True)) # add cat_id index level
            )
            if -1 in user_mistakes.index.codes[2]:
                # if nans in cat_id:
                user_mistakes = user_mistakes.reset_index(level='cat_id')
                # For rules with no category (not present in lt_rules, like java rules),
                # assign them to their own category.
                cat_is_null = user_mistakes['cat_id'].isnull()
                user_mistakes.loc[cat_is_null, 'cat_id'] = (
                    user_mistakes.index.get_level_values('rule_id')[cat_is_null.values]
                )
                user_mistakes = user_mistakes.set_index('cat_id', append=True)

            self._user_mistakes = (
                user_mistakes.swaplevel(0, 1)
                 .swaplevel(1, 2)
                 .join(self.user_corpora)
                 .eval("freq_per_word = count / nr_words")
                 .eval("freq_per_tweet = count / nr_tweets")
                 .sort_index()
            )
            # user_mistakes = user_mistakes[np.setdiff1d(user_mistakes.columns, self.user_corpora.columns)]
        return self._user_mistakes

    @user_mistakes.setter
    def user_mistakes(self, _user_mistakes):
        self._user_mistakes = _user_mistakes
        del cells_mistakes

    @user_mistakes.deleter
    def user_mistakes(self):
        self.user_mistakes = None


    @property
    def cells_ses_df(self):
        if self._cells_ses_df is None:
            self._cells_ses_df = pd.DataFrame()

            for r in self.regions:
                # can actually happen we do this over more than one region: take eg
                # average income (normalized for cost of living eg) for regions speaking
                # same language
                agg_metrics = spatial_agg.get_agg_metrics(
                    r.ses_df, r.weighted_cell_levels_corr
                )
                self._cells_ses_df = pd.concat([
                    self._cells_ses_df, agg_metrics
                ]).sort_index()

        return self._cells_ses_df

    @cells_ses_df.setter
    def cells_ses_df(self, _cells_ses_df):
        if (
            _cells_ses_df is None
            or (
                self.cells_ses_df.index.levels[0].size
                != _cells_ses_df.index.levels[0].size
            )
        ):
            del self.cells_mask
        self._cells_ses_df = _cells_ses_df

    @cells_ses_df.deleter
    def cells_ses_df(self):
        self.cells_ses_df = None

    def add_ses_idx(self, ses_idx):
        ses_idx = {ses_idx} if isinstance(ses_idx, str) else set(ses_idx)
        found_idx = False
        for r in self.regions:
            if ses_idx == {'all'}:
                idx_intersect = set(r.ses_data_options.keys())
            else:
                idx_intersect = set(r.ses_data_options.keys()).intersection(ses_idx)
            for idx in idx_intersect:
                found_idx = True
                r.ses_df = r.load_ses_df(idx)
                agg_metrics = spatial_agg.get_agg_metrics(
                    r.ses_df, r.weighted_cell_levels_corr
                )
                # groupby index and take first to remove duplicate SES
                self.cells_ses_df = (
                    pd.concat([self.cells_ses_df, agg_metrics])
                     .groupby(self.cells_ses_df.index.names)
                     .first()
                )
                # If on right cell size level, optionally load additional stats from
                # files documented in ses_data_options: quartiles, etc
                for col in r.ses_df.columns:
                    add_score_stats = r.ses_data_options[idx].get('score_stats_cols', {}).get(col)
                    if add_score_stats is not None and agg_metrics.shape[0] == r.ses_df.shape[0]:
                        score_stats_options = r.ses_data_options[idx].copy()
                        score_stats_options['score_cols'] = add_score_stats
                        score_stats = (
                            ses_data.read_df(r.paths.ext_data, **score_stats_options)
                             .assign(metric=col)
                             .set_index('metric', append=True)
                        )
                        self.cells_ses_df = self.cells_ses_df.join(score_stats)
                    
        if not found_idx:
            print(f"{ses_idx} was not found anywhere.")

    @property
    def cells_users_df(self):
        if self._cells_users_df is None:
            self._cells_users_df = (
                self.user_df.loc[self.user_mask]
                 .groupby('cell_id')
                 .agg(
                    nr_users=('nr_words', 'size'),
                    nr_tweets=('nr_tweets', 'sum'),
                    nr_words=('nr_words', 'sum'),
                    avg_nr_unique_words=('nr_unique_words', 'mean'),
                )
            )
        return self._cells_users_df

    @cells_users_df.setter
    def cells_users_df(self, _cells_users_df):
        self._cells_users_df = _cells_users_df
        del self.cells_mask

    @cells_users_df.deleter
    def cells_users_df(self):
        self.cells_users_df = None

    @property
    def cells_nr_users_th(self):
        return self._cells_nr_users_th

    @cells_nr_users_th.setter
    def cells_nr_users_th(self, th):
        self._cells_nr_users_th = th
        del self.cells_mask

    @property
    def cells_mask(self):
        if 'is_relevant' not in self.cells_geodf.columns:
            self.cells_geodf['is_relevant'] = False
            nr_users_mask = self.cells_users_df['nr_users'] >= self.cells_nr_users_th
            intersect = self.cells_users_df.loc[nr_users_mask].index.intersection(
                self.cells_ses_df.index.levels[0]
            )
            self.cells_geodf.loc[intersect, 'is_relevant'] = True
            print(
                f'Keeping {intersect.size} cells out of {self.cells_geodf.shape[0]},'
                f' {(~nr_users_mask).sum()} discarded from `cells_nr_users_th`'
            )
        return self.cells_geodf['is_relevant']

    @cells_mask.setter
    def cells_mask(self, mask):
        self.cells_geodf['is_relevant'] = mask
        del self.cells_mistakes

    @cells_mask.deleter
    def cells_mask(self):
        # private here to avoid RecursionError
        if self._cells_geodf is not None:
            self._cells_geodf = self.cells_geodf.drop(columns='is_relevant', errors='ignore')
        del self.cells_mistakes

    @property
    def relevant_cells(self):
        return self.cells_mask.loc[self.cells_mask].index


    @property
    def cells_mistakes(self):
        if self._cells_mistakes is None:
            cells_mask = self.user_df['cell_id'].isin(self.relevant_cells)
            udf = self.user_mistakes.join(
                self.user_df.loc[self.user_mask & cells_mask, 'cell_id']
            )
            self._cells_mistakes = (
                udf.groupby(['cell_id', 'cat_id', 'rule_id'])
                 .agg(
                    count=('count', 'sum'),
                    usum_freq_per_word=('freq_per_word', 'sum'),
                    usum_freq_per_tweet=('freq_per_tweet', 'sum'),
                )
                 .join(self.cells_users_df['nr_users'])
                 .eval("uavg_freq_per_word = usum_freq_per_word / nr_users")
                 .eval("uavg_freq_per_tweet = usum_freq_per_tweet / nr_users")
                 .loc[:, ['count', 'uavg_freq_per_word', 'uavg_freq_per_tweet']]
            )
        return self._cells_mistakes

    @cells_mistakes.setter
    def cells_mistakes(self, _cells_mistakes):
        self._cells_mistakes = _cells_mistakes

    @cells_mistakes.deleter
    def cells_mistakes(self):
        self.cells_mistakes = None


    def select_mistakes(self, metric='uavg_freq_per_word', cat_id=None, rule_id=None):
        series_name = f"{rule_id or cat_id or 'all'} mistakes".lower()
        if cat_id is None:
            cat_id = slice(None)
        if rule_id is None:
            rule_id = slice(None)
        idx_sel = (slice(None), cat_id, rule_id)
        return (
            self.cells_mistakes.loc[idx_sel, metric]
             .groupby('cell_id')
             .sum()
             .rename(series_name)
        )


    def select_ses_metric(self, name: str | list[str], metric='wavg'):
        if name not in self.cells_ses_df.index.levels[1]:
            self.add_ses_idx(name)
        return self.cells_ses_df.loc[(slice(None), name), metric].unstack()[name]


    def make_subregions_mask(self, subreg_df, set_cells_mask=False):
        # set_cells_mask defaults to False because setting cells_mask deletes
        # cells_mistakes
        regions_dict = self.regions_dict
        mask = pd.Series(dtype=bool)
        for cc in subreg_df.index.levels[0]:
            r = regions_dict[cc]
            mask = pd.concat([mask, r.make_subregions_mask(subreg_df.loc[cc])])
        mask = self.cells_mask & mask.reindex(self.cells_geodf.index).fillna(False)
        mask = mask.rename('isin_area')
        if set_cells_mask:
            self.cells_mask = mask
        return mask


    def get_lt_rules(self, lt_cats_dict):
        self.lt_rules = (
            pd.concat(
                [r.lt_rules for r in self.regions if r.lt_rules is not None]
            )
             .groupby('rule_id')
             .first()
             .combine_first(text_process.get_lt_rules(lt_cats_dict))
        )
        return self.lt_rules

    @property
    def width_ratios(self):
        if self._width_ratios is None:
            width_ratios = np.ones(len(self.regions))
            for i, reg in enumerate(self.regions):
                min_lon, min_lat, max_lon, max_lat = reg.bbox
                width, height = geo_utils.calc_bbox_dims(
                    min_lon, min_lat, max_lon, max_lat
                )
                width_ratios[i] = width / height
            self._width_ratios = width_ratios
        return self._width_ratios

    def get_custom_width_ratios(self, cells_idc):
        width_ratios = np.ones(len(self.regions))
        for i, reg in enumerate(self.regions):
            reg_cells = reg.cells_geodf.join(pd.DataFrame(index=cells_idc), how='inner')
            if reg_cells.shape[0] > 0.5 * reg.cells_geodf.shape[0]:
                # we'll plot the shape so use width_ratio saved as property
                width_ratios[i] = self.width_ratios[i]
            else:
                min_x, min_y, max_x, max_y = reg_cells.total_bounds
                width = max_x - min_x
                height = max_y - min_y
                width_ratios[i] = width / height
        return width_ratios

    def get_maps_pos(
        self, total_width, total_height=None, ratio_lgd=None, width_ratios=None
    ):
        if width_ratios is None:
            width_ratios = self.width_ratios.copy()
        if ratio_lgd is not None:
            width_ratios = np.append(width_ratios, ratio_lgd)
        return map_viz.position_axes(
            width_ratios, total_width, total_height=total_height, ratio_lgd=ratio_lgd
        )


    def pre_process_z_plot(self, z_plot) -> pd.Series:
        if not isinstance(z_plot, pd.Series):
            if len(z_plot) == len(self.relevant_cells):
                index = self.relevant_cells
            elif len(z_plot) == len(self.cells_geodf.index):
                index = self.cells_geodf.index
            else:
                raise ValueError(
                    "Input z values' length does not match either the number of "
                    "relevant_cells or the total number of cells"
                )
            z_plot = pd.Series(z_plot, index=index, name='z_plot')
        return z_plot


    def map_continuous_choro(
        self, z_plot, normed_bboxes: bool | np.ndarray = True,
        total_width=178, total_height=None, axes=None, cax=None,
        cbar_kwargs=None, save=False, **choro_kwargs
    ):
        z_plot = self.pre_process_z_plot(z_plot)
        
        if normed_bboxes is True:
            if z_plot.shape[0] < 0.9 * self.cells_geodf.shape[0]:
                width_ratios = self.get_custom_width_ratios(z_plot.index)
            else:
                width_ratios = None
            # calculate optimal position
            normed_bboxes, (total_width, total_height) = self.get_maps_pos(
                total_width, total_height=total_height, ratio_lgd=1/10,
                width_ratios=width_ratios
            )

        if axes is None:
            if normed_bboxes is False:
                total_height = total_width / self.width_ratios[0]
                figsize = (total_width/10/2.54, total_height/10/2.54)
                _, axes = plt.subplots(len(self.regions), figsize=figsize)
            else:
                figsize = (total_width/10/2.54, total_height/10/2.54)
                _, axes = plt.subplots(len(self.regions) + 1, figsize=figsize)
                cax = axes[-1]
                axes = axes[:-1]

        if save and not choro_kwargs.get('save_path'):
            # deprecated
            cell_sizes = '-'.join(r.cell_size for r in self.regions)
            choro_kwargs['save_path'] = (
                self.paths.case_figs
                / self.paths.map_fig_fname_fmt.format(
                    metric=z_plot.name, cell_sizes=cell_sizes
                )
            )

        # If normed_bboxes set to False, don't position the axes
        if not normed_bboxes is False:
            choro_kwargs['normed_bboxes'] = normed_bboxes

        fig, axes = map_viz.choropleth(
            z_plot, self.regions, axes=axes, cax=cax,
            cbar_kwargs=cbar_kwargs, **choro_kwargs
        )

        return fig, axes


    def map_mistake(
        self, cat_id=None, rule_id=None, metric='uavg_freq_per_word', vmin=None, vmax=None,
        cmap='plasma', cbar_kwargs=None, **plot_kwargs
    ):
        cbar_label = f'{metric} of {rule_id or cat_id}'
        z_plot = self.select_mistakes(metric=metric, cat_id=cat_id, rule_id=rule_id)

        fig, axes = self.map_continuous_choro(
            z_plot, cmap=cmap, vmin=vmin, vmax=vmax,
            cbar_label=cbar_label, cbar_kwargs=cbar_kwargs, **plot_kwargs
        )
        return fig, axes


    def map_interactive(
        self, z_plot, tooltip_df=None, save=False, save_path=None,
        **explore_kwargs
    ):
        z_plot = self.pre_process_z_plot(z_plot)

        plot_gdf = self.cells_geodf[['geometry']].join(z_plot, how='inner')
        if tooltip_df is not None:
            plot_gdf = plot_gdf.join(tooltip_df)
        m = plot_gdf.explore(z_plot.name, **explore_kwargs)

        if save or save_path:
            cell_sizes = '-'.join(r.cell_size for r in self.regions)
            if save_path is None:
                save_path = self.paths.case_figs / self.paths.map_fig_fname_fmt.format(
                    metric=z_plot.name, cell_sizes=cell_sizes
                )
            m.save(save_path.with_suffix('.html'))
        return m


    def explore_mistake(
        self, cat_id=None, rule_id=None, metric='uavg_freq_per_word',
        **map_interactive_kw
    ):
        cbar_label = f'{metric} of {rule_id or cat_id}'
        z_plot = self.select_mistakes(metric=metric, cat_id=cat_id, rule_id=rule_id)
        return self.map_interactive(z_plot, **map_interactive_kw)
