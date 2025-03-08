# %%
from argparse import ArgumentParser
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz
from tqdm import tqdm
from wandb.apis.public.runs import Run

import wandb

PROJECT_URL = "https://wandb.ai/hyena/40b-train"
ENTITY = "hyena"
PROJECT = "40b-train"
RUN = "40b-train-n256-v2"
WANDB_DT_FORMAT="%Y-%m-%dT%H:%M:%S.%fZ"
DT_FORMAT = "%Y-%m-%d %H:%M:%S"
WANDB_TZ = pytz.UTC
TZ = pytz.timezone("America/Los_Angeles")
HISTORY_KEYS = [
    "_step",
    "data/tokens_per_second_per_gpu",
    "runtime/iteration_time",
    "train/lm_loss",
    "validation/lm_loss",
]
DEFAULT_COL_ORDER = [
        "name",
        "start",
        "end",
        "duration",
        "start_step",
        "end_step",
        "avg_throughput",
        "avg_iteration_time",
        "start_train_loss",
        "end_train_loss",
        "start_val_loss",
        "end_val_loss",
    ]

def preprocess_args(args):
    """Normalize arguments by converting hyphens to underscores."""
    return [arg.replace('-', '_') if arg.startswith('--') else arg for arg in args]

def str_to_list(s):
    return s.split(",")

def convert_iso_to_dt(dt) -> datetime:
    if isinstance(dt, str):
        utc_datetime = datetime.strptime(dt, WANDB_DT_FORMAT).replace(tzinfo=WANDB_TZ)
        dt = utc_datetime.astimezone(TZ)
        # dt = pst_datetime.strftime(DT_FORMAT)
    return dt

def download_run_data(runs: list[Run], num_samples=None, keys=HISTORY_KEYS):
    run_dict = {}

    progress_bar = tqdm(runs)
    for i, run in enumerate(progress_bar):
        try:
            metadata = run.metadata

            # Get start and end times
            start = metadata["startedAt"]
            start = convert_iso_to_dt(start)
            start_key = start.strftime(DT_FORMAT)
            key = f"{run.name}_{start}"
            duration = timedelta(seconds=run.summary["_wandb"]["runtime"])
            end = start + duration
            end_key = end.strftime(DT_FORMAT)

            # Get run history
            history = (
                pd.DataFrame(run.scan_history(keys=keys))
                if num_samples is None
                else run.history(keys=keys, samples=num_samples)
            )
            if len(history) == 0:
                continue
                print(f"Skipping {key} due to empty history")

            key = run.url
            run_dict[key] = {
                "start": start_key,
                "end": end_key,
                "name": run.name,
                "rawconfig": run.rawconfig,
                "metadata": metadata,
                "history": history,
                "summary": run.summary
            }

            # Print summary
            avg_tpt, avg_t = history.mean()[
                ["data/tokens_per_second_per_gpu", "runtime/iteration_time"]
            ]
            start_step, end_step = history["_step"].iloc[0], history["_step"].iloc[-1]
            start_train_loss, end_train_loss = (
                history["train/lm_loss"].iloc[0],
                history["train/lm_loss"].iloc[-1],
            )
            start_val_loss, end_val_loss = (
                history["validation/lm_loss"].iloc[0],
                history["validation/lm_loss"].iloc[-1],
            )
            # Update run_dict with summary stats
            run_dict[key].update(
                {
                    "start_step": start_step,
                    "end_step": end_step,
                    "duration": duration.total_seconds() / 3600,
                    "avg_throughput": avg_tpt,
                    "avg_iteration_time": avg_t,
                    "start_train_loss": start_train_loss,
                    "end_train_loss": end_train_loss,
                    "start_val_loss": start_val_loss,
                    "end_val_loss": end_val_loss,
                }
            )

            header = f"{key}:\n Started at: {start_key} Ended at: {end_key} Duration: {duration.total_seconds()/3600:.2f}hrs"
            steps = f"Start Step: {start_step} End Step: {end_step}"
            runtime = f"Avg Throughput: {avg_tpt:.2f} Avg Iteration Time: {avg_t:.2f}"
            losses = f"Start Train Loss: {start_train_loss:.2f} End Train Loss: {end_train_loss:.2f}\n Start Val Loss: {start_val_loss:.2f} End Val Loss: {end_val_loss:.2f}"
            print_str = "\n ".join([header, steps, runtime, losses])
            print(print_str)
        except Exception as e:
            print(f"Error processing run {i} {run.project}/{run.name}/{run.id}: {e}")
            continue
        
    return pd.DataFrame(run_dict).T

def get_runs(entity: str, project: str, run_pats: list[str]) -> list[Run]:
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    runs: list[Run] = [run for run in runs if any(n in run.name for n in run_pats)]
    return runs

def summarize_run_data(df: pd.DataFrame, col_order=DEFAULT_COL_ORDER, exclude_keys: list[str] = None):
    summary_keys = list(set(df.columns) - set(exclude_keys))
    summary_stats = df[summary_keys][col_order]

    return summary_stats

def main(args):
    runs: list[Run] = get_runs(entity=args.entity, project=args.project, run_name=args.run)
    print(f"Runs: {len(runs)}")
    runs = runs[-args.most_recent:]
    df = download_run_data(runs, num_samples=args.num_samples)
    df = df[DEFAULT_COL_ORDER + ["metadata", "history"]]
    summary_df = summarize_run_data(df)

    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H")
    full_df_path = args.output_dir / f"{timestamp}-{args.run}_full.csv"
    summary_df_path = args.output_dir / f"{timestamp}-{args.run}_summary.csv"

    print(f"Saving full data to {full_df_path}")
    df.to_csv(full_df_path)
    print(f"Saving summary data to {summary_df_path}")
    summary_df.to_csv(summary_df_path)

    pd.options.display.max_columns = None
    pd.options.display.max_colwidth = None
    
    print(summary_df)
# %%

if __name__ == "__main__":
    parser = ArgumentParser(description="Export run data to CSV")
    parser.add_argument("--entity", type=str, default=ENTITY, help="Wandb entity")
    parser.add_argument("--project", type=str, default=PROJECT, help="Wandb project")
    parser.add_argument("--run", type=str, default=RUN, help="Run name")
    parser.add_argument("--most_recent", type=int, default=2, help="Number of most recent runs to fetch")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to fetch")
    parser.add_argument("--output_dir", type=Path, default="wandb_runs", help="Directory to save output files")
    args = parser.parse_args()
    main(args)


