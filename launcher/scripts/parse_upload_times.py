import argparse
import os
import re

import pandas as pd


def parse_upload_times(fp):
    lines = open(fp, "r").readlines()
    data = []

    pattern = r"^\s*(\d+):\s+SAVE_CHECKPOINT:\s+Upload took ([\d.]+) seconds"

    # Parse each line
    for line in lines:

        match = re.search(pattern, line.strip())
        if match:
            rank = int(match.group(1))
            time_taken = float(match.group(2))
            data.append({"rank": rank, "time_taken": time_taken})

    # Create a DataFrame
    df = pd.DataFrame(data)
    df["rank"] = df["rank"].astype(int)
    df.time_taken = df.time_taken.astype(float)
    return df

if __name__ == "__main__":
    # Parse the file
    parser = argparse.ArgumentParser(
        description="Parse upload times", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("log_file", type=str, help="Path to the log file")
    parser.add_argument("--output_file", default=None, type=str, help="Path to the output file")
    args = parser.parse_args()
    print(f"Parsing log file {args.log_file}")
    df = parse_upload_times(args.log_file)
    print(df.head())
    stats = df.groupby("rank").describe()
    pd.set_option("display.max_columns", None)
    print(stats)
    if not args.output_file:
        # Get the output file without extension Path("a/b/c.txt").stem -> "c"
        args.output_file = os.path.basename(args.log_file.split("/")[-1])
    df.to_csv(f"s3_logs/{args.output_file}_parsed.csv")
    stats.to_csv(f"s3_logs/{args.output_file}_stats.csv")
    
