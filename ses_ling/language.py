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
    all_cntr_shapes: InitVar[geopd.GeoDataFrame]
    # If many more such dicts (3 atm), switch to init_dict solution
    ses_data_options: dict
    shapefile_col: InitVar[str] = 'FID'
    shapefile_val: InitVar[str | None] = None
    # not a problem to default with mutable because this initvar is never touched
    extract_shape_kwargs: InitVar[dict] = {'simplify_tol': 100}
    cell_shapefiles: dict = field(default_factory=dict)
    cell_levels_corr_files: dict = field(default_factory=dict)
    ses_idx: str | None = None
    mongo_coll: str = ''
    year_from: int = 2015
    year_to: int = 2021
    readable: str = ''
    xy_proj: str = 'epsg:3857'
    max_place_area: float = 5e9
    cell_size: int | float | str = 50e3
    # Params
    nighttime_acty_th: float = 0.5
    all_acty_th: float = 0.1
    count_th: int = 3
    shape_geodf: geopd.GeoDataFrame | None = None
    cells_geodf: geopd.GeoDataFrame | None = None
    cell_lvls_corr: pd.DataFrame | None = None
    ses_df: pd.DataFrame | None = None
    lt_rules: pd.DataFrame | None = None
    _paths: paths_utils.ProjectPaths | None = None
    _user_residence_cell: pd.DataFrame | None = None
    _user_mistakes: pd.DataFrame | None = None
    _user_corpora: pd.DataFrame | None = None

    def __post_init__(
        self, all_cntr_shapes, shapefile_col, shapefile_val, extract_shape_kwargs,
    ):
        if self.shape_geodf is None:
            shapefile_val = shapefile_val or self.cc
            mask = all_cntr_shapes[shapefile_col].str.startswith(shapefile_val)
            self.shape_geodf = geo_utils.extract_shape(
                all_cntr_shapes.loc[mask], self.cc, xy_proj=self.xy_proj, **extract_shape_kwargs
            )
        if self.cells_geodf is None:
            self.load_cells_geodf()

        if self.ses_idx is None:
            self.ses_idx = list(self.ses_data_options.keys())[0]

        if self.ses_df is None:
            self.ses_df = self.load_ses_df()

        if self.cell_lvls_corr is None:
            # may not be used but does not cost much to load (just a small csv)
            cell_shapefile_kind = self.cell_shapefiles[self.cell_size]["kind"]
            corr_path = (
                self.paths.ext_data
                / self.cell_levels_corr_files[cell_shapefile_kind]
            )
            agg_level = self.cells_geodf.index.name
            weight_col = self.ses_data_options[self.ses_idx]['weight_col']
            self.cell_levels_corr = spatial_agg.levels_corr(
                corr_path, self.ses_df, agg_level, weight_col=weight_col
            )

        if self.lt_rules is None:
            lt_rules_path = self.paths.rule_category
            if lt_rules_path.exists():
                self.lt_rules = (
                    pd.read_csv(lt_rules_path)
                     .rename(columns={'ruleId': 'rule_id', 'category': 'cat_id'})
                     .groupby(['cat_id', 'rule_id'])
                     .first()
                )

    def __repr__(self):
        field_dict = self.__dataclass_fields__
        attr_str_components = []
        for key in field_dict.keys():
            field = getattr(self, key)
            field_repr = repr(field)
            if len(field_repr) < 200:
                attr_str_components.append(f'{key}={field_repr}')

        attr_str = ', '.join(attr_str_components)
        return f'{self.__class__.__name__}({attr_str})'


    @classmethod
    def from_dict(cls, cc, lc, all_cntr_shapes, **kwargs):
        matching_kwargs = {
            k: v for k, v in kwargs.items()
            if k in inspect.signature(cls).parameters
        }
        return cls(cc, lc, all_cntr_shapes, **matching_kwargs)


    @staticmethod
    def from_global_json(file_path, cc, lc, all_cntr_shapes, **kwargs):
        with open(file_path) as f:
            countries_dict = json.load(f)
        d = {**countries_dict[cc], **kwargs}
        return Region(cc, lc, all_cntr_shapes, **d)


    def to_dict(self):
        # custom to_dict to keep only parameters that can be in save path
        list_attr = [
            'lc', 'readable', 'cc', 'year_from', 'year_to', 'cell_size',
            'max_place_area', 'xy_proj', 'nighttime_acty_th', 'all_acty_th', 'count_th'
        ]
        return {attr: getattr(self, attr) for attr in list_attr}


    @property
    def shape_bbox(self):
        if self.shape_bbox is None:
            self.shape_bbox = self.shape_geodf.geometry.to_crs('epsg:4326').total_bounds
        return self.shape_bbox

    @property
    def paths(self):
        # # Here use this convention so that Region can inherit a Language's paths
        # if self._paths is None:
        self._paths = paths_utils.ProjectPaths()
        self._paths.partial_format(**self.to_dict())
        return self._paths

    @paths.setter
    def paths(self, p):
        self._paths = p

    @paths.deleter
    def paths(self):
        self._paths = None


    def load_cells_geodf(self, cell_size=None):
        # other way if to make cell_size a property, and its setter calls this function
        # without this initial if, and without the kwarg
        if cell_size is not None:
            self.cell_size = cell_size

        if isinstance(self.cell_size, str):
            fname = self.cell_shapefiles[self.cell_size]['fname']
            index_col = self.cell_shapefiles[self.cell_size]['index_col']
            self.cells_geodf = geo_utils.load_ext_cells(
                fname, index_col, xy_proj=self.xy_proj
            )
        else:
            _, self.cells_geodf, _, _ = geo_utils.create_grid(
                self.shape_geodf, self.cell_size, self.cc,
                xy_proj=self.xy_proj, intersect=True
            )

    def load_ses_df(self, ses_idx=None):
        if ses_idx is not None:
            self.ses_idx = ses_idx

        return ses_data.read_df(
            self.paths.ext_data, **self.ses_data_options[self.ses_idx]
        )

    @property
    def user_residence_cell(self):
        if self._user_residence_cell is None:
            p = self.paths.user_residence_cell
            if p.exists():
                self._user_residence_cell = pd.read_parquet(p)
            else:
                raise ValueError(
                    'Please run the script to generate `user_residence_cell`'
                )
        return self._user_residence_cell

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

        self._user_corpora = user_corpora
        self._user_mistakes = user_mistakes


@dataclass
class Language:
    lc: str
    readable: str
    # Arguable solution: could also just edit countries_dict in script: pb lose default
    # for subsequent runs... This is more heavy but more robust.
    _cc_init_params: InitVar[dict]
    all_cntr_shapes: InitVar[geopd.GeoDataFrame]
    countries_dict: InitVar[dict]
    regions: list[Region] = field(default_factory=list)
    year_from: int = 2015
    year_to: int = 2021
    month_from: int = 1
    month_to: int = 12
    latlon_proj: str = 'epsg:4326'
    # Parameters for word and cell filters:
    nighttime_acty_th: float = 0.5
    all_acty_th: float = 0.1
    count_th: int = 3
    # Data containers (frames, arrays)
    cells_geodf: geopd.GeoDataFrame = field(init=False)
    lt_rules: pd.DataFrame = field(init=False)
    lt_categories: pd.DataFrame = field(init=False)
    # TODO: combine next two in _user_df?
    _user_residence_cell: pd.DataFrame | None = None
    _user_corpora: pd.DataFrame | None = None
    _user_mistakes: pd.DataFrame | None = None
    width_ratios: np.ndarray | None = None

    def __post_init__(self, _cc_init_params, all_cntr_shapes, countries_dict):
        if len(self.regions) < len(_cc_init_params):
            missing_ccs = set(_cc_init_params.keys()).difference({reg.cc for reg in self.regions})
            for cc in missing_ccs:
                self.regions.append(
                    Region.from_dict(
                        cc, self.lc, all_cntr_shapes,
                        year_from=self.year_from, year_to=self.year_to,
                        **{**countries_dict[cc], **_cc_init_params[cc]}
                    )
                )
        # sort a posteriori
        _, self.regions = (
            list(x)
            for x in zip(*sorted(zip(self.list_cc, self.regions), key=lambda t: t[0]))
        )
        self.cells_geodf = pd.concat([
            reg.cells_geodf.to_crs(self.latlon_proj)
            for reg in self.regions
        ]).sort_index()

        if self.paths.language_tool_categories.exists():
            with open(self.paths.language_tool_categories) as f:
                lt_cats_dict = json.load(f)
        else:
            lt_cats_dict = text_process.parse_online_grammar(self.lc)
            with open(self.paths.language_tool_categories, 'w') as f:
                json.dump(lt_cats_dict, f)

        self.lt_categories = text_process.get_lt_categories(lt_cats_dict)
        self.lt_rules = self.get_lt_rules(lt_cats_dict)


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
    def from_countries_dict(cls, lc, readable, list_cc_, all_cntr_shapes,
                            countries_dict, year_from=2015,
                            year_to=2021, **kwargs):
        # regions = [
        #     Region.from_dict(cc, lc, countries_dict[cc], year_from=year_from,
        #                      year_to=year_to, month_from=month_from, month_to=month_to)
        #     for cc in list_cc
        # ]
        return cls(
            lc, readable, list_cc_, all_cntr_shapes, countries_dict,
            year_from=year_from, year_to=year_to, **kwargs
        )


    def to_dict(self):
        # custom to_dict to keep only parameters that can be in save path
        list_attr = [
            'lc', 'readable', 'str_cc', 'year_from', 'year_to',
            'nighttime_acty_th', 'all_acty_th', 'count_th'
        ]
        return {attr: getattr(self, attr) for attr in list_attr}


    @property
    def list_cc(self):
        return [reg.cc for reg in self.regions]

    @property
    def str_cc(self):
        return '-'.join(self.list_cc)

    @property
    def paths(self):
        self._paths = paths_utils.ProjectPaths()
        self._paths.partial_format(**self.to_dict())
        return self._paths

    @paths.setter
    def paths(self, p):
        self._paths = p


    @property
    def user_residence_cell(self):
        if self._user_residence_cell is None:
            self._user_residence_cell = pd.concat([
                r.user_residence_cell for r in self.regions
            ])
        return self._user_residence_cell

    @property
    def user_mistakes(self):
        if self._user_mistakes is None:
            user_mistakes = (
                pd.concat([r.user_mistakes for r in self.regions])
                 .join(self.lt_rules[[]]) # add cat_id index level
                 .reset_index(level='cat_id')
            )
            # For rules with no category (not present in lt_rules, like java rules),
            # assign them to their own category.
            cat_is_null = user_mistakes['cat_id'].isnull()
            user_mistakes.loc[cat_is_null, 'cat_id'] = user_mistakes.index.get_level_values('rule_id')[cat_is_null.values]
            user_mistakes = (
                user_mistakes.set_index('cat_id', append=True)
                 .swaplevel(0, 1)
                 .swaplevel(1, 2)
                 .join(self.user_corpora)
            )
            user_mistakes['freq_per_word'] = user_mistakes['count'] / user_mistakes['nr_words']
            user_mistakes['freq_per_tweet'] = user_mistakes['count'] / user_mistakes['nr_tweets']
            # user_mistakes = user_mistakes[np.setdiff1d(user_mistakes.columns, self.user_corpora.columns)]
            self._user_mistakes = user_mistakes.sort_index()
        return self._user_mistakes

    @user_mistakes.setter
    def user_mistakes(self, _user_mistakes):
        self._user_mistakes = _user_mistakes


    def get_lt_rules(self, lt_cats_dict):
        self.lt_rules = text_process.get_lt_rules(lt_cats_dict)
        self.lt_rules = (
            pd.concat(
                [self.lt_rules]
                + [r.lt_rules for r in self.regions if r.lt_rules is not None]
            )
             .groupby(['cat_id', 'rule_id'])
             .first()
        )
        return self.lt_rules


    def read_metric(self, metric_col):
        for r in self.regions:
            # can actually happen we do this over more than one region: take eg average
            # income (normalized for cost of living eg) for regions speaking same
            # language
            agg_metrics = spatial_agg.get_agg_metrics(
                r.ses_df, r.cell_levels_corr, metric_col=metric_col
            )
            
            self.cells_geodf = (
                self.cells_geodf.drop(columns=agg_metrics.columns, errors='ignore')
                 .join(agg_metrics)
            )


    def get_width_ratios(self, ratio_lgd=None):
        self.width_ratios = np.ones(len(self.list_cc) + 1)
        for i, reg in enumerate(self.regions):
            min_lon, min_lat, max_lon, max_lat = reg.shape_bbox
            # For a given longitude extent, the width is maximum the closer to the
            # equator, so the closer the latitude is to 0.
            eq_not_crossed = int(min_lat * max_lat > 0)
            lat_max_width = min(abs(min_lat), abs(max_lat)) * eq_not_crossed
            width = geo_utils.haversine(min_lon, lat_max_width, max_lon, lat_max_width)
            height = geo_utils.haversine(min_lon, min_lat, min_lon, max_lat)
            self.width_ratios[i] = width / height
        if ratio_lgd:
            self.width_ratios[-1] = ratio_lgd
        else:
            self.width_ratios = self.width_ratios[:-1]
        return self.width_ratios


    def get_maps_pos(self, total_width, total_height=None, ratio_lgd=None):
        width_ratios = self.get_width_ratios(ratio_lgd=ratio_lgd)
        return map_viz.position_axes(
            width_ratios, total_width, total_height=total_height, ratio_lgd=ratio_lgd
        )


    def map_continuous_choro(
        self, z_plot, normed_bboxes: bool | np.ndarray = True,
        total_width=178, total_height=None, axes=None, cax=None,
        cbar_kwargs=None, **choro_kwargs
    ):
        if normed_bboxes is True:
            # calculate optimal position
            normed_bboxes, (total_width, total_height) = self.get_maps_pos(
                total_width, total_height=total_height, ratio_lgd=1/10
            )

        if axes is None:
            figsize = (total_width/10/2.54, total_height/10/2.54)
            _, axes = plt.subplots(len(self.regions) + 1, figsize=figsize)
            cax = axes[-1]
            axes = axes[:-1]

        if isinstance(z_plot, pd.Series):
            plot_series = z_plot
        else:
            plot_series = pd.Series(z_plot, index=self.relevant_cells, name='z')

        fig, axes = map_viz.choropleth(
            plot_series, self.regions, axes=axes, cax=cax,
            cbar_kwargs=cbar_kwargs, **choro_kwargs
        )

        # if normed_bboxes set to False, don't position the axes
        if not normed_bboxes is False:
            for ax, bbox in zip(np.append(axes, cax), normed_bboxes):
                ax.set_position(bbox)

        return fig, axes


    def map_word(self, word, vcenter=0, vmin=None, vmax=None, cmap='bwr',
                 cbar_kwargs=None, **plot_kwargs):
        cbar_label = f"{self.word_vectors.word_vec_var.split('(')[0]} of {word}"
        is_regional = self.global_counts['is_regional']
        word_idx = self.global_counts.loc[is_regional].index.get_loc(word)
        z_plot = self.word_vectors[:, word_idx]

        fig, axes = self.map_continuous_choro(
            z_plot, cmap=cmap, vcenter=vcenter, vmin=vmin, vmax=vmax,
            cbar_label=cbar_label, cbar_kwargs=cbar_kwargs, **plot_kwargs
        )
        return fig, axes
