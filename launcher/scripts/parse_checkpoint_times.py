import argparse
import re

import pandas as pd

CHECKPOINT_PATTERN = r"SAVE_CHECKPOINT: iteration: (\d+), ckpt saving takes ([\d.]+) seconds"

def parse_checkpoint_times(file_path, pattern=CHECKPOINT_PATTERN):
    iterations = []
    checkpoint_times = []
    
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                iterations.append(int(match.group(1)))
                checkpoint_times.append(float(match.group(2)))
    
    df = pd.DataFrame({
        'iteration': iterations,
        'checkpoint_time': checkpoint_times
    })
    return df

if __name__ == '__main__':
    # Example usage
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', default="local_logs/checkpoint_times.txt", help='Path to the log file to parse')
    args = parser.parse_args()
    df = parse_checkpoint_times(args.log_file)

    print(df.head())
    print(df.tail())
    print(df.describe()['checkpoint_time'])
    df.to_csv('local_logs/checkpoint_times.csv', index=False)