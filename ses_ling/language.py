from __future__ import annotations

import json
from dataclasses import _FIELD, InitVar, dataclass, field

import geopandas as geopd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ses_ling.data.socioeconomic as ses_data
import ses_ling.utils.geometry as geo_utils
import ses_ling.utils.paths as paths_utils
import ses_ling.utils.spatial_agg as spatial_agg
import ses_ling.utils.text_process as text_process
import ses_ling.visualization.maps as map_viz
from ses_ling.region import Region


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
    _user_cell_acty: pd.DataFrame | None = None
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
    def user_cell_acty(self):
        if self._user_cell_acty is None:
            user_cell_acty = pd.DataFrame()
            for r in self.regions:
                reg_acty = r.user_cell_acty.copy()
                reg_acty.index = reg_acty.index.rename({reg_acty.index.names[-2]: 'cell_id'})
                user_cell_acty = pd.concat([user_cell_acty, reg_acty])

            user_cell_acty = (
                user_cell_acty.groupby(['user_id', 'cell_id'])[['count', 'prop_cell']]
                 .sum()
                 .join(self.user_residence_cell['cell_id'].rename('res_cell_id'), how='inner')
            )
            self._user_cell_acty = user_cell_acty
        return self._user_cell_acty

    @user_cell_acty.setter
    def user_cell_acty(self, _user_cell_acty):
        self._user_cell_acty = _user_cell_acty

    @user_cell_acty.deleter
    def user_cell_acty(self):
        self.user_cell_acty = None

    
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
        del self.user_cell_acty

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
        # means relevant in the sense that there was sufficient text to detect mistakes
        # so should not be used to compute assortativity matrices eg
        if 'is_relevant' not in self.user_df.columns:
            nr_words_mask = self.user_df['nr_words'] >= self.user_nr_words_th
            user_mask = nr_words_mask & self.user_df['cell_id'].notnull()
            self.user_df['is_relevant'] = user_mask
            print(f'Keeping {user_mask.sum()} users out of {self.user_df.shape[0]}')
        return self.user_df['is_relevant']

    @user_mask.setter
    def user_mask(self, _user_mask):
        self.user_df['is_relevant'] = _user_mask
        # to force setters' deletions:
        self.user_df['is_relevant'] = self.user_df['is_relevant'] 

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
        print(
            f'{mask.sum()} cells left after masking cells where there are fewer than',
            f' {self.cells_nr_users_th} residents',
        )
        return mask


    def iter_subregs_mask(self, raw_subreg_df, selected_subregs=None):
        if selected_subregs is None:
            subreg_df = raw_subreg_df.copy()
        else:
            mask = raw_subreg_df.index.isin(selected_subregs, level='subreg')
            subreg_df = raw_subreg_df.loc[mask, :].copy()

        for name, df in subreg_df.groupby('subreg'):
            print(f'** {name} **')
            reg_mask = self.make_subregions_mask(df, set_cells_mask=False)
            yield name, reg_mask

    def iter_subregs(
        self, raw_subreg_df, selected_subregs=None, ses_metric=None, cat_id=None,
        include_pop=False
    ):
        for name, reg_mask in self.iter_subregs_mask(raw_subreg_df, selected_subregs):
            # careful: all these DFs are not necessarily based on same set of users!
            subreg_d = {}
            subreg_d['cells_mask'] = reg_mask
            masks_dict = {'cell_id': reg_mask}
            subreg_d['user_res_cell'] = ses_data.apply_masks(
                self.user_residence_cell, masks_dict
            )
            # subreg_d['user_cell_acty'] = ses_data.apply_masks(
            #     self.user_cell_acty, masks_dict
            # )
            subreg_d['user_cell_acty'] = ses_data.preprocess_cell_acty(
                self.user_cell_acty, subreg_d['user_res_cell'], masks_dict
            )
            if include_pop:
                cells_pop = self.select_ses_metric('total_pop', 'weight')
                subreg_d['cells_pop'] = ses_data.apply_masks(
                    cells_pop, masks_dict
                )
            if ses_metric is not None:
                cells_ses_metric = self.select_ses_metric(ses_metric)
                subreg_d['cells_ses_metric'] = ses_data.apply_masks(
                    cells_ses_metric, masks_dict
                )
            if cat_id is not None:
                cells_mistake = self.select_mistakes(cat_id=cat_id)
                subreg_d['cells_mistake'] = ses_data.apply_masks(
                    cells_mistake, masks_dict
                )
            yield name, subreg_d


    @property
    def subreg_df(self):
        return self._subreg_df

    @subreg_df.setter
    def subreg_df(self, _subreg_df):
        self._subreg_df = _subreg_df
        del self.cells_subreg

    @property
    def cells_subreg(self):
        if 'subreg' not in self.cells_geodf.columns:
            self.cells_geodf['subreg'] = None
            for name, df in self.subreg_df.groupby('subreg'):
                mask = self.make_subregions_mask(df)
                self.cells_geodf.loc[mask, 'subreg'] = name
        return self.cells_geodf['subreg']

    @cells_subreg.setter
    def cells_subreg(self, _cells_subreg):
        self.cells_geodf['subreg'] = _cells_subreg

    @cells_subreg.deleter
    def cells_subreg(self):
        if self._cells_geodf is not None:
            self._cells_geodf = self.cells_geodf.drop(columns='subreg', errors='ignore')


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
