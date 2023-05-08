from __future__ import annotations

import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
import paramiko
from dotenv import load_dotenv
from tqdm import tqdm

import ses_ling.utils.paths as path_utils
import ses_ling.visualization.sim as sim_viz


def nured_run(
    script_fname,
    args,
    allocated_time,
    waiting_time=0,
    executable=None,
    custom_log_file=True,
    ssh_domain="nuredduna2020",
    username_key="IFISC_USERNAME",
    run_path="/usr/local/bin/run",
    script_dir="scripts",
    proj_dir=None,
):
    """Runs a Python script on nuredduna with some arguments.

    Parameters
    ----------
    script_fname : str
        Name of the python file to execute.
    args : str | list[str]
        Arguments to provide to the script. Can contain optional, named arguments of the
        form --arg value, that can be parsed easily with `argparse`
        https://docs.python.org/3/library/argparse.html
    allocated_time : int
        How many minutes your job is allowed to run.
    waiting_time : int, optional
        How many seconds to wait between job submissions, to avoid collapsing the disk
        servers. Default is 0.
    executable : str, optional
        Optionally, select manually the executable, which by default is the virtual
        environment's Python. This default only works if you're launching a job from
        `salmunia` that shares the same conda environments, to launch locally you'd have
        to specify this argument.
    custom_log_file : bool, optional
        Whether to customize run's default log file names, if True, will be in a "logs"
        subfolder and named according to the script name and time of submission.
    ssh_domain : str, optional
        Name of the server to connect to, by default "nuredduna2020"
    username_key : str, optional
        Name of the environment variable set in `.env` that contains your user name on
        the server, by default "IFISC_USERNAME"
    run_path : str
        Path to the `run` executable, "/usr/local/bin/run" by default.
    script_dir : str, optional
        Directory where the script to run is located, relative to your project
        directory, by default "scripts".
    proj_dir : str, optional
        Directory where your project files are located, by default the environment
        variable "PROJ_DIR" will be taken.
    """
    load_dotenv()
    if isinstance(args, str):
        args = [args]

    script_dir = Path(proj_dir or os.environ["PROJ_DIR"]) / script_dir

    if executable is None:
        # This assumes we're on the server.
        venv_path = os.environ.get("CONDA_PREFIX", os.environ.get("VIRTUAL_ENV"))
        executable = Path(venv_path) / "bin" / "python"

    sbatch_dir = "/common/slurm/bin"
    base_cmd = f"export PATH=$PATH:{sbatch_dir};cd {script_dir};"

    py_file_path = Path(script_fname)
    if custom_log_file:
        logs_dir = script_dir / 'logs'
        logs_dir.mkdir(exist_ok=True)
        time_str = datetime.datetime.now().isoformat(timespec='milliseconds')
        log_file_fmt = f"{py_file_path.stem}_{time_str}{{i}}{{type}}.log"
    else:
        log_str = ""

    whole_cmd = base_cmd
    for i, a in enumerate(args):
        if custom_log_file:
            iter_log_file = logs_dir / log_file_fmt.format(type="_out", i=i)
            iter_err_log_file = logs_dir / log_file_fmt.format(type="_err", i=i)
            log_str = f" -o {iter_log_file} -e {iter_err_log_file}"

        whole_cmd += (
            f"{run_path} -t {allocated_time}{log_str}"
            # Because this part below is enclosed in quotes, optional arguments can be
            # used, otherwise they are interpreted as `run`'s optional arguments.
            f' "{executable} {py_file_path} {a}";'
        )

        if waiting_time:
            whole_cmd += f"sleep {waiting_time};"

    print(whole_cmd)

    # Get username fron environment variable in case it was not configured in
    # `~/.ssh/.config`.
    ssh_username = os.environ.get(username_key)
    with paramiko.client.SSHClient() as ssh_client:
        ssh_client.load_system_host_keys()
        ssh_client.connect(ssh_domain, username=ssh_username)
        ssh_stdin, ssh_stdout, ssh_stderr = ssh_client.exec_command(whole_cmd)
        print(ssh_stderr.readlines())
        print(ssh_stdout.readlines())


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
    return variant_interact


def get_switch_mask(df_may_switch, l_arr, q_arr, rng):
    # og variant = 0 implies s, else 1-s.
    s_switch = l_arr[df_may_switch["variant"].astype(int)]
    # ses = k: og variant 0 implies (1 - q_{k,0}]), og 1 implies q_{k,0}
    q_ses = q_arr[df_may_switch["ses"].astype(int)]
    q_switch = q_ses - (2 * q_ses - 1) * (df_may_switch["variant"] == 0).astype(int)
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
        cells_nr_users_th: int = 0,
        rng: None | int | np.random.Generator = 1,
        sim_id=None,
        **kwargs,
    ):
        """_summary_

        Parameters
        ----------
        region : str
            _description_
        agent_df : pd.DataFrame
            _description_
        mob_matrix : pd.DataFrame
            _description_
        lv : float, optional
            Global prestige of variant 2, by default 0.5. Expected to be above 0.5.
        q1 : float, optional
            Preference of lowest class for variant 1, by default 0.5. Expected to be
            superior to lv to have "interesting" simulations, meaning ones that don't
            lead for sure to extinction of one of the two varieties..
        q2 : float, optional
            Preference of highest class for variant 2, by default 0.5. If more than 2
            classes, preference of each class for variant 1 will be taken from a linear
            interpolation between q1 and (1 - q2). q2 is thus expecetd to be superior or
            equal to 0.5.
        cells_nr_users_th : int, optional
            _description_, by default 0: run simulation with all cells, potentially
            measure metrics on subsset of cells though.
        rng : int | np.random.Generator, optional
            _description_, by default 1

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """
        self.sim_id = sim_id
        self.region = region
        self.cells_nr_users_th = cells_nr_users_th
        # agent_df must have "ses", "res_cell" and "variant" columns / keys
        self.agent_df = agent_df
        if not self.agent_df['res_cell'].is_monotonic_increasing:
            raise ValueError(
                "agents must be sorted in contiguous groups of residence cells with" 
                " monotonically increasing IDs."
            )
        if rng is None or isinstance(rng, int):
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
        # to make computation easier, in the following array suppose the given qs are
        # all preference for variant 0 of the SES classes, hence the 1-q2
        self.q_arr = np.linspace(q1, 1-q2, self.nr_classes)

        if 'variant' not in self.agent_df.columns:
            variant_probas = agent_df['ses'] / (self.nr_classes - 1)
            self.agent_df['variant'] = self.rng.random(agent_df.shape[0]) < variant_probas

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
        cls, region, step=0, nr_classes=2, lv=0.5, q1=0.5, q2=0.5,
        cells_nr_users_th=15, path_fmt=None, **kwargs,
    ):
        # Here have to mimick __init__'s signature to make clear some arguments are
        # needed for `sim_init_fmt` and `sim_state_fmt`
        cls_args_dict = {
            'region': region, 'cells_nr_users_th': cells_nr_users_th,
            'lv': lv, 'q1': q1, 'q2': q2
        }
        fmt_dict = {**kwargs, **cls_args_dict, **{'sim_id': kwargs.get('sim_id') or ""}}
        fmt_dict['nr_classes'] = nr_classes
        init_fmt = path_fmt or path_utils.ProjectPaths().sim_init_fmt
        mob_matrix_save_path = Path(str(init_fmt).format(name='mob_matrix', **fmt_dict))
        mob_matrix = pd.read_parquet(mob_matrix_save_path)

        if path_fmt is None:
            if step == 0:
                path_fmt = init_fmt
            else:
                path_fmt = path_utils.ProjectPaths().sim_state_fmt
        fmt_dict['step'] = step
        agent_df_save_path = Path(str(path_fmt).format(name='agent_df', **fmt_dict))
        agent_df = pd.read_parquet(agent_df_save_path).astype(int)

        class_instance = cls(
            agent_df=agent_df, mob_matrix=mob_matrix, **cls_args_dict, **kwargs
        )
        if step > 0:
            evol_save_path = Path(str(path_fmt).format(name='evol_df', **fmt_dict))
            evol_df = pd.read_parquet(evol_save_path)
            cols = evol_df.columns
            evol_df = evol_df.rename(columns=dict(zip(cols, cols.astype(int))))
            class_instance.evol_variant0 = evol_df
        return class_instance

    @property
    def nr_agents(self):
        return self.agent_df.shape[0]

    @property
    def step(self):
        return self.evol_variant0.shape[0] - 1

    @property
    def nr_cells(self):
        return self.mob_matrix.shape[0]

    def to_dict(self):
        list_attr = [
            'region', 'lv', 'q1', 'q2',
            'nr_classes', 'nr_agents', 'nr_cells',
            'cells_nr_users_th', 'step', 'sim_id'
        ]
        return {attr: getattr(self, attr) for attr in list_attr}

    def det_variant_interact(self):
        return det_variant_interact(self.agent_df, self.rng)

    def det_cell(self):
        return det_cell(self.thresholds_cell_moves, self.cell_cumul_pop, self.rng)

    def get_switch_mask(self, df_may_switch):
        return get_switch_mask(df_may_switch, self.l_arr, self.q_arr, self.rng)

    def run_step(self, tracking_list: list):
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
        # To update if more than 2 variants.
        tracking_list.append(
            [
                switches.get(i, {}).get(0, 0) - switches.get(i, {}).get(1, 0)
                for i in range(self.nr_classes)
            ]
        )
        self.agent_df = agent_df.drop(columns=['cell', 'variant_interact'])
        return switches_df

    def run(self, nr_steps, show_progress=False):
        pbar = tqdm(range(nr_steps), disable=not show_progress)
        tracking_list = []
        for i in pbar:
            switches_df = self.run_step(tracking_list)
            pbar.set_description(f"{switches_df.sum().sum()} switched variant")

        self.evol_variant0 = get_evolution_from_cumul(self.evol_variant0, tracking_list)

    def save_state(self, path_fmt=None):
        if path_fmt is None:
            if self.step == 0:
                path_fmt = path_utils.ProjectPaths().sim_init_fmt
            else:
                path_fmt = path_utils.ProjectPaths().sim_state_fmt

        fmt_dict = {**self.to_dict(), **{'sim_id': self.sim_id or ""}}

        if self.step > 0:
            evol_save_path = Path(str(path_fmt).format(name='evol_df', **fmt_dict))
            cols = self.evol_variant0.columns
            self.evol_variant0.rename(columns=dict(zip(cols, cols.astype(str)))).to_parquet(evol_save_path)

        saved_agent_df = self.agent_df.copy().astype({'variant': bool})
        if self.nr_classes == 2:
            saved_agent_df = saved_agent_df.astype({'ses': bool})
        agent_df_save_path = Path(str(path_fmt).format(name='agent_df', **fmt_dict))
        saved_agent_df.to_parquet(agent_df_save_path)

        if self.step == 0:
            fmt_dict = {**fmt_dict, **{'step': 0}}
            mob_matrix_save_path = Path(str(path_fmt).format(name='mob_matrix', **fmt_dict))
            self.mob_matrix.to_parquet(mob_matrix_save_path)

    def plot_evol(self, **kwargs):
        nr_agents_per_class = self.agent_df.groupby('ses').size().to_dict()
        ax = sim_viz.variant_evol(self.evol_variant0, nr_agents_per_class, **kwargs)
        return ax
