from __future__ import annotations

import inspect
from dataclasses import _FIELD, InitVar, dataclass, field

import geopandas as geopd
import pandas as pd
from tqdm.auto import tqdm

import ses_ling.data.socioeconomic as ses_data
import ses_ling.utils.geometry as geo_utils
import ses_ling.utils.paths as paths_utils
import ses_ling.utils.spatial_agg as spatial_agg


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
    _user_cell_acty: pd.DataFrame | None = None
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
        cell_level = self.cells_geodf.index.name
        idc_col = subreg_df.columns[-1]
        isin_area = pd.Series(False, name='isin_area', index=self.cells_geodf.index)
        for (r, agg_col), idc  in subreg_df.groupby(subreg_df.index.names):
            if idc.shape[0] > 1:
                # List of indices is provided in the values of subreg_df
                mask = corr_df[idc_col.lower()].isin(idc[idc_col].values)
            else:
                # Only one item at agg_col level to match
                # TODO: when col not in corr_df, read cell_levels_corr_files[col]
                mask = corr_df[agg_col] == r
            idc_region = pd.Index(corr_df.loc[mask, cell_level]).unique()
            isin_area.loc[idc_region] = True
            print(f"{idc_region.size} cells in {r}.")
        return isin_area


    def match_cell_level(self, df, cell_id_name='cell_id', isin_col=False):
        # by default, cell_id info should be in index
        if cell_id_name.lower() != self.cells_geodf.index.name.lower():
            if isin_col:
                df = df.set_index(cell_id_name, append=True)

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
            df = df.join(cell_levels_corr).droplevel(cell_levels[1])

            if isin_col:
                df = df.reset_index(cell_levels[0])

        return df


    @property
    def user_cell_acty(self):
        if self._user_cell_acty is None:
            p = self.paths.user_cell_acty
            if p.exists():
                self._user_cell_acty = pd.read_parquet(p)
            else:
                raise ValueError(
                    f'Please run the script to generate {p}'
                )

        # convention set in `user_residence.get_cell_user_activity`
        self._user_cell_acty = self.match_cell_level(
            self._user_cell_acty, self._user_cell_acty.index.names[1]
        )
        return self._user_cell_acty


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

        self._user_residence_cell = self.match_cell_level(
            self._user_residence_cell, self._user_residence_cell.columns[0], True
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
