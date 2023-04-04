from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

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
        agents: dict | pd.DataFrame,
        mob_matrix: np.ndarray,
        l_arr: np.ndarray,
        q_arr: np.ndarray,
        rng: int | np.random.Generator = 1,
    ):
        """_summary_

        Parameters
        ----------
        agents : dict | pd.DataFrame
            Must have "ses", "res_cell" and "variant" columns / keys
        mob_matrix : np.ndarray
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
        if isinstance(agents, dict):
            self.agent_df = pd.DataFrame(agents)
        elif isinstance(agents, pd.DataFrame):
            self.agent_df = agents.copy()
        else:
            raise ValueError("agents must either be a dict or a DataFrame")
        if not self.agent_df['res_cell'].is_monotonic_increasing:
            raise ValueError(
                "agents must be sorted in contiguous groups of residence cells with" 
                " monotonically increasing IDs."
            )

        self.nr_classes = self.agent_df["ses"].nunique()
        self.mob_matrix = mob_matrix
        # It's so we can use the following as indexer in `agent_df` that we require
        # "res_cell" to be monotonically increasing.
        self.cell_cumul_pop = np.concatenate([
            [0],
            self.agent_df['res_cell'].value_counts().sort_index().cumsum().values,
        ])
        self.thresholds_cell_moves = self.mob_matrix.cumsum(axis=1)
        self.l_arr = l_arr
        self.q_arr = q_arr
        if isinstance(rng, int):
            self.rng = np.random.default_rng(rng)
        elif isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            raise ValueError(
                "rng must either be a seed or a numpy random number generator"
            )
        # self.iterations_summary = self.agent_df.groupby(['ses', 'variant']).size().rename(0).to_frame()
        self.evol_variant0 = (
            self.agent_df.groupby(["variant", "ses"])
            .size()
            .loc[0]
            .rename(0)
            .reindex(range(self.nr_classes), fill_value=0)
            .to_frame()
            .T.rename_axis(index="step")
        )

    @classmethod
    def from_dict(cls, agent_dict: dict, *args, **kwargs):
        """Initialize agent_df from dictionary.

        Parameters
        ----------
        agent_dict : dict
            Must have "ses", "res_cell" and "variant" keys
        """
        agent_df = pd.DataFrame(agent_dict)
        return cls(agent_df, *args, **kwargs)

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
        # more general but more computationally intensive:
        # iter_summmary = agent_df.groupby(['ses', 'variant']).size().rename(i + iterations_summary.shape[1])
        # tracking_list.append(iter_summmary)
        return switches_df

    def run(self, nr_iter):
        pbar = tqdm(range(nr_iter))
        tracking_list = []
        for i in pbar:
            switches_df = self.step(tracking_list)
            pbar.set_description(f"{switches_df.sum().sum()} switched variant")

        self.evol_variant0 = get_evolution_from_cumul(self.evol_variant0, tracking_list)
        # iterations_summary = pd.concat([iterations_summary] + list_iter_summary, axis=1).fillna(0).astype(int).rename_axis('time', axis=1)

    def plot_evol(self, **kwargs):
        nr_agents_per_class = self.agent_df.groupby('ses').size().to_dict()
        ax = sim_viz.variant_evol(self.evol_variant0, nr_agents_per_class, **kwargs)
        return ax
