from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import ses_ling.utils.paths as path_utils
import ses_ling.visualization.sim as sim_viz


def det_cell(thresholds_cell_moves, cell_cumul_pop, rng):
    draw = rng.random(cell_cumul_pop[-1])
    list_cell_arrays = []
    for i_cell in range(cell_cumul_pop.shape[0] - 1):
        list_cell_arrays.append(
            np.searchsorted(
                thresholds_cell_moves[i_cell, :],
                draw[cell_cumul_pop[i_cell]: cell_cumul_pop[i_cell+1]],
            )
        )
    return np.concatenate(list_cell_arrays)


def det_variant_interact(agent_df, rng):
    nr_agents = agent_df.shape[0]
    cell_x_variant = agent_df.groupby(["cell", "variant"]).size().unstack()
    cell_x_variant = (cell_x_variant.T / cell_x_variant.sum(axis=1)).T
    thresholds_cell_variants = cell_x_variant.values.cumsum(axis=1)[
        agent_df["cell"].values
    ]
    variant_interact = (
        rng.random(nr_agents)[:, np.newaxis] > thresholds_cell_variants
    ).argmin(axis=1)
    # variant_interact = agent_df['variant'][(rng.random(nr_agents)[:, np.newaxis] > thresholds_cell_variants).argmin(axis=1)].values
    return variant_interact


def get_switch_mask(df_may_switch, l_arr, q_arr, rng):
    # og variant = 0 implies s, else 1-s.
    s_switch = l_arr[df_may_switch["variant"].astype(int)]
    # ses = 0 implies q1, og variant 0 implies (1 - q1), og 1 implies q1
    # ses = 1 implies q2, og variant 0 implies q2, og 1 implies (1 - q2)
    q_ses = q_arr[df_may_switch["ses"].astype(int)]
    q_switch = q_ses - (df_may_switch["variant"] == 0).astype(int) * (2 * q_ses - 1)
    switch_th = q_switch * s_switch
    switch_mask = rng.random(df_may_switch.shape[0]) < switch_th
    return switch_mask


def get_evolution_from_cumul(evol_df, list_changes):
    new_iters = evol_df.values[-1, :] + np.asarray(list_changes).cumsum(axis=0)
    new_iters = pd.DataFrame(
        new_iters, index=evol_df.shape[0] + np.arange(new_iters.shape[0])
    )
    new_evol_df = pd.concat([evol_df, new_iters])
    return new_evol_df


class Simulation:
    def __init__(
        self,
        region: str,
        agent_df: pd.DataFrame,
        mob_matrix: pd.DataFrame,
        lv: float = 0.5,
        q1: float = 0.5,
        q2: float = 0.5,
        rng: int | np.random.Generator = 1,
    ):
        """_summary_

        Parameters
        ----------
        agents : dict | pd.DataFrame
            Must have "ses", "res_cell" and "variant" columns / keys
        mob_matrix : pd.DataFrame
            _description_
        l_arr : np.ndarray
            _description_
        q_arr : np.ndarray
            _description_
        rng : int | np.random.Generator, optional
            _description_, by default 1

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """
        self.agent_df = agent_df
        if not self.agent_df['res_cell'].is_monotonic_increasing:
            raise ValueError(
                "agents must be sorted in contiguous groups of residence cells with" 
                " monotonically increasing IDs."
            )
        if isinstance(rng, int):
            self.rng = np.random.default_rng(rng)
        elif isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            raise ValueError(
                "rng must either be a seed or a numpy random number generator"
            )

        self.nr_classes = self.agent_df["ses"].nunique()
        self.mob_matrix = mob_matrix
        self.thresholds_cell_moves = self.mob_matrix.to_numpy().cumsum(axis=1)
        # It's so we can use the following as indexer in `agent_df` that we require
        # "res_cell" to be monotonically increasing.
        self.cell_cumul_pop = np.concatenate([
            [0],
            self.agent_df['res_cell'].value_counts().sort_index().cumsum().values,
        ])

        self.lv = lv
        # This array has size (nr_variants,)
        self.l_arr = np.array([lv, 1 - lv])
        self.q1 = q1
        self.q2 = q2
        # This array should be modified if more than 2 classes are introduced!
        self.q_arr = np.array([q1, 1 - q2])
        self.region = region

        self.evol_variant0 = (
            self.agent_df.groupby(["variant", "ses"])
            .size()
            .loc[0]
            .rename(0)
            .reindex(range(self.nr_classes), fill_value=0)
            .to_frame()
            .T.rename_axis(index="step")
        )
        # self.iterations_summary = self.agent_df.groupby(['ses', 'variant']).size().rename(0).to_frame()

    @classmethod
    def from_dict(cls, region, agent_dict: dict, *args, **kwargs):
        """Initialize agent_df from dictionary.

        Parameters
        ----------
        agent_dict : dict
            Must have "ses", "res_cell" and "variant" keys
        """
        agent_df = pd.DataFrame(agent_dict)
        return cls(region, agent_df, *args, **kwargs)

    @classmethod
    def from_saved_state(
        cls, region, *args,
        step=0, nr_classes=2, path_fmt=None, lv=0.5, q1=0.5, q2=0.5, **kwargs,
    ):
        if path_fmt is None:
            path_fmt = path_utils.ProjectPaths().sim_state_fmt
        fmt_dict = {
            **kwargs,
            **{'region': region, 'nr_classes': nr_classes, 'lv': lv, 'q1': q1, 'q2': q2}
        }
        mob_matrix_save_path = Path(str(path_fmt).format(kind='mob_matrix', step=0, **fmt_dict))
        mob_matrix = pd.read_parquet(mob_matrix_save_path)

        fmt_dict['step'] = step
        agent_df_save_path = Path(str(path_fmt).format(kind='agent_df', **fmt_dict))
        agent_df = pd.read_parquet(agent_df_save_path).astype(int)

        instance = cls(region, *args, agent_df=agent_df, mob_matrix=mob_matrix, **kwargs)
        if step > 0:
            evol_save_path = Path(str(path_fmt).format(kind='evol_df', **fmt_dict))
            evol_df = pd.read_parquet(evol_save_path)
            cols = evol_df.columns
            evol_df = evol_df.rename(columns=dict(zip(cols, cols.astype(int))))
            instance.evol_variant0 = evol_df
        return instance

    @property
    def nr_agents(self):
        return self.agent_df.shape[0]

    def det_variant_interact(self):
        return det_variant_interact(self.agent_df, self.rng)

    def det_cell(self):
        return det_cell(self.thresholds_cell_moves, self.cell_cumul_pop, self.rng)

    def get_switch_mask(self, df_may_switch):
        return get_switch_mask(df_may_switch, self.l_arr, self.q_arr, self.rng)

    def step(self, tracking_list: list):
        agent_df = self.agent_df
        agent_df["cell"] = self.det_cell()
        agent_df["variant_interact"] = self.det_variant_interact()
        # optionally ses_interact if different influences depending on SES group, but
        # well part of homophily already taken into account in mobility, so not
        # mandatory.
        same_variant = agent_df["variant"] == agent_df["variant_interact"]
        df_may_switch = agent_df.loc[~same_variant]
        switch_mask = self.get_switch_mask(df_may_switch)

        whole_switch_mask = ~same_variant & switch_mask
        agent_df.loc[whole_switch_mask, "variant"] = agent_df.loc[
            whole_switch_mask, "variant_interact"
        ]

        switches_df = (
            df_may_switch.loc[switch_mask]
            .groupby(["variant_interact", "ses"])
            .size()
            .unstack(fill_value=0)
        )
        switches = switches_df.to_dict()
        tracking_list.append(
            [
                switches.get(i, {}).get(0, 0) - switches.get(i, {}).get(1, 0)
                for i in range(self.nr_classes)
            ]
        )
        agent_df = agent_df.drop(columns=['cell', 'variant_interact'])
        # more general but more computationally intensive:
        # iter_summmary = agent_df.groupby(['ses', 'variant']).size().rename(i + iterations_summary.shape[1])
        # tracking_list.append(iter_summmary)
        return switches_df

    def run(self, nr_steps):
        pbar = tqdm(range(nr_steps))
        tracking_list = []
        for i in pbar:
            switches_df = self.step(tracking_list)
            pbar.set_description(f"{switches_df.sum().sum()} switched variant")

        self.evol_variant0 = get_evolution_from_cumul(self.evol_variant0, tracking_list)
        # iterations_summary = pd.concat([iterations_summary] + list_iter_summary, axis=1).fillna(0).astype(int).rename_axis('time', axis=1)

    def save_state(self, path_fmt=None):
        if path_fmt is None:
            path_fmt = path_utils.ProjectPaths().sim_state_fmt
        fmt_dict = {
            'region': self.region,
            'step': self.evol_variant0.shape[0] - 1,
            'nr_classes': self.nr_classes,
            'lv': self.lv,
            'q1': self.q1,
            'q2': self.q2,
        }
        evol_save_path = Path(str(path_fmt).format(kind='evol_df', **fmt_dict))
        cols = self.evol_variant0.columns
        self.evol_variant0.rename(columns=dict(zip(cols, cols.astype(str)))).to_parquet(evol_save_path)

        saved_agent_df = self.agent_df.copy().astype({'variant': bool})
        if self.nr_classes == 2:
            saved_agent_df = saved_agent_df.astype({'ses': bool})
        agent_df_save_path = Path(str(path_fmt).format(kind='agent_df', **fmt_dict))
        saved_agent_df.to_parquet(agent_df_save_path)

        fmt_dict = {**fmt_dict, **{'step': 0}}
        mob_matrix_save_path = Path(str(path_fmt).format(kind='mob_matrix', **fmt_dict))
        self.mob_matrix.to_parquet(mob_matrix_save_path)

    def plot_evol(self, **kwargs):
        nr_agents_per_class = self.agent_df.groupby('ses').size().to_dict()
        ax = sim_viz.variant_evol(self.evol_variant0, nr_agents_per_class, **kwargs)
        return ax
