import datetime
import os
import subprocess
from pathlib import Path

import pandas as pd

import wandb
from wandb.apis.public import Run

TEST_DIR = Path(__file__).resolve().parent
SAVANNA_DIR = TEST_DIR.parent
SAVANNA_RUN_VAR = "SAVANNA_RUN_ID"
DEFAULT_WANDB_PROJECT = "tests"
DEFAULT_DATA_CONFIG = os.path.join(TEST_DIR, "test_configs/og.yml")


def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d%H%M")


def print_delimiter(ch="-", length=80):
    print()
    print(ch * length)
    print()


def run_program(
    model_config_path,
    data_config_path=DEFAULT_DATA_CONFIG,
    wandb_project=DEFAULT_WANDB_PROJECT,
    wandb_group=None,
    wandb_run_name=None,
    timeout=300,
):
    """Run the program with the specified YAML config."""
    # Generate formatted time to the minute include year month and day
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M")
    if wandb_group is None:
        wandb_group = f"{ts}"
    cmd = [
        "python",
        f"{SAVANNA_DIR}/launch.py",
        f"{SAVANNA_DIR}/train.py",
        str(data_config_path),
        str(model_config_path),
        "--wandb_project",
        wandb_project,
        "--wandb_group",
        wandb_group,
    ]
    print(f"Running command: {' '.join(cmd)}")

    os.environ[SAVANNA_RUN_VAR] = (
        str(wandb_run_name) if wandb_run_name is not None else f"{ts}"
    )

    process = subprocess.run(cmd, check=True, timeout=timeout)


def get_wandb_runs(wandb_project, wandb_entity=None, num_samples=500, summaries_only=False):
    """
    Downloads all runs from the specified project and group

    Args:
        wandb_project: The name of the project
        wandb_group: The name of the group
        summaries_only: Whether to only return a summary dataframe which contains the last logged metrics

    Returns:
        pd.DataFrame:
        A dataframe containing all runs for the `name`, `config`, and `history` columns
            - `name`: The name of the run
            - `config`: The (json) config of the run
            - `history`: The history of the run as an embedded dataframe
        If `summaries_only` is True, the dataframe will contain a flattened version of the `summary` column,
        containing the last logged metrics for each run without the full history
    """
    api = wandb.Api()
    if wandb_entity is None:
        wandb_entity = api.default_entity
    runs_path = f"{wandb_entity}/{wandb_project}"
    runs = api.runs(runs_path)
    print(f"Downloading runs for {runs_path}")
    summaries, configs, names, histories = [], [], [], []
    for run in runs:
        run: Run
        names.append(run.name)

        summaries.append(run.summary._json_dict)
        configs.append({k: v for k, v in run.config.items() if not k.startswith("_")})
        # histories.append(pd.DataFrame(run.scan_history()))
        histories.append(run.history(samples=num_samples))
   
    runs_df = pd.DataFrame(
        {"name": names, "config": configs, "summary": summaries, "history": histories}
    )
    if summaries_only:
        _summary_df = pd.json_normalize(runs_df["summary"])
        summary_df = pd.concat(
            [runs_df.drop(columns=["summary", "history"]), _summary_df], axis=1
        )
        return summary_df

    return runs_df.drop(columns=["summary"])


def get_wandb_run(wandb_project, wandb_run_name, summaries_only=False):
    """Get a wandb run from its id
    Args:
        run_path: The id of the run in the format of `project/group/run_id`

    """
    runs_df = get_wandb_runs(wandb_project=wandb_project, summaries_only=summaries_only)
    return runs_df[runs_df.name == wandb_run_name]