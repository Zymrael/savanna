import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd


def get_files_in_step_dir(step_dir, step_number, base_dir):
    """Extracts file information from a single directory."""
    data = []
    for file_name in os.listdir(step_dir):
        file_path = os.path.join(step_dir, file_name)
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path)
            # common_dir = os.path.relpath(step_dir, base_dir)
            data.append(
                {
                    "checkpoint_dir": base_dir,
                    "step": step_number,
                    "filename": file_name,
                    "size": file_size,
                }
            )
    return data

def summarize_df(df, base_dir):
    # Print summary
    print(f"Found {len(df)} files across {len(df['step'].unique())} checkpointed directories")
    print(f"Total size: {df['size'].sum() / 1e9:.2f} GB")
    # chkpted_dirs = df.step.map(lambda s: os.path.join(base_dir, f"global_step{s}")).unique()
    # print(f"Checkpointed dirs: {chkpted_dirs}")

    pd.options.display.max_categories = 100
    pd.options.display.max_columns = 100
    pd.options.display.max_colwidth = 1000
    print(f"Checkpoint dir: {base_dir}")
    summary_stats = df.groupby("step")["size"].describe().reset_index()[["step", "count", "min", "50%", "max"]]
    summary_stats["count"] = summary_stats["count"].astype(int)
    print(summary_stats)
    
def scan_directory(base_dir):
    # Regular expression to match directories starting with "global_step" followed by an integer
    step_pattern = re.compile(r"global_step(\d+)")

    # List to store file information
    data = []
    subdirs = []
    # Traverse the directory structure
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            match = step_pattern.match(dir_name)
            if match:
                # Extract the step number
                step_number = int(match.group(1))

                # Construct full path to the directory
                step_dir = os.path.join(root, dir_name)
                subdirs.append((step_dir, step_number))
    
    # Process directories in parallel
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_files_in_step_dir, step_dir, step_number, base_dir): (step_dir, step_number) for step_dir, step_number in subdirs}

        for future in as_completed(futures):
            data.extend(future.result())
    
    # Create DataFrame from collected data
    df = pd.DataFrame(data)

    return df.sort_values("step")


if __name__ == "__main__":
    DEFAULT_CHKPT_DIR = "/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-v2/40b_train_v2/202410271619"
    parser = argparse.ArgumentParser(description="Scan directory for global_step checkpoints")
    parser.add_argument("--ckpt_dir", type=Path, default=DEFAULT_CHKPT_DIR, help="Base directory to scan")
    parser.add_argument("--log_dir", type=str, default="local_logs", help="Directory to save log file")
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)

    args = parser.parse_args()
    file_prefix = str(args.ckpt_dir).split("/checkpoints/")[-1:][0].replace("/", "_")

    save_dir = os.path.join(args.log_dir, file_prefix)
    os.makedirs(save_dir, exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    output_path = f"{save_dir}/{timestamp}.csv"

    if not os.path.exists(output_path):
        print(f"Scanning {args.ckpt_dir} for checkpoints, file prefix: {file_prefix}")
        start = time.time()
        df = scan_directory(args.ckpt_dir)
        end = time.time()
        print(f"Scan completed in {end - start:.2f} seconds")
        print(f"Saving to {output_path}")
        df.to_csv(output_path, index=False)
    else:
        print(f"Log file {output_path} already exists. Skipping scan.")
        df = pd.read_csv(output_path)    
    
    summarize_df(df, args.ckpt_dir)