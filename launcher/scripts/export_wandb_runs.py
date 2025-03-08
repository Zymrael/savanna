import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from wandb.apis.public.runs import Run
from wandb_lib import DEFAULT_COL_ORDER, download_run_data, get_runs, str_to_list

if __name__ == "__main__":

    # Replaced hyphens with underscores when options are specified in the command line. E.g., --output-dir -> --output_dir

    args = ArgumentParser()
    args.add_argument("--entity", type=str, required=True)
    args.add_argument("--project", type=str, required=True)
    args.add_argument("--run", type=str_to_list, required=True)
    args.add_argument("--num_samples", "--num-samples", type=int, default=None)
    args.add_argument("--most_recent", "--most-recent",  type=int, default=5)
    args.add_argument("--output_dir", "--output-dir", type=Path, default=Path("wandb_runs"))
    args = args.parse_args()

    print(args)

    runs: list[Run] = get_runs(entity=args.entity, project=args.project, run_pats=args.run)
    
    runs = runs[-args.most_recent :]
    print(f"Downloading {args.most_recent} of {len(runs)} runs from {args.project} matching {args.run}")
    df = download_run_data(runs, num_samples=args.num_samples)
    summary_df = df[["name"] + DEFAULT_COL_ORDER]

    output_dir = args.output_dir
    dt = datetime.now().strftime("%Y-%m-%d_%H:%M")
    output_dir = output_dir / args.project / dt
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving to {output_dir}")
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    df.to_csv(output_dir / "full.csv", index=False)

    print("Done!")