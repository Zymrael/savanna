import argparse
import os
import re
from datetime import datetime

import pandas as pd

LOG_DIR="/lustre/fs01/portfolios/dir/users/jeromek/40b-train-n256-v2/202410281731/logs/40b_train_v2"
CHECKPOINT_PATTERN = r"SAVE_CHECKPOINT: iteration: (\d+), ckpt saving takes ([\d.]+) seconds"
CHECKPOINT_HEADERS=["iteration", "checkpoint_time"]
STARTUP_TIME_PATTERN = r"model and optimizer: ([\d.]+)\s+[|]\s+train/valid/test data iterators:\s+([\d.]+)"
STARTUP_HEADERS=["model_and_optimizer_ms", "data_iterator_time_ms"]

def parse_datetime_from_log_path(log_path):
    # Define the regex pattern to extract date and time
    filename = log_path.split('/')[-1]
    pattern = r'.+_date_(\d{2}-\d{2}-\d{2})_time_(\d{2}-\d{2}-\d{2})\.log'
    
    # Search for the pattern in the log path
    match = re.search(pattern, filename)
    if match:
        # Extract date parts from the match
        dt = match.group(1)   
        t = match.group(2)   
        
        # Parse the components into a datetime object
        date_str = f"{dt} {t.replace('-',':')}"
        parsed_datetime = datetime.strptime(date_str, "%y-%m-%d %H:%M:%S")
        
        return parsed_datetime.strftime("%Y-%m-%d %H:%M:%S")
    else:
        return log_path
    
def search_pattern_in_dir(directory, pattern, headers):
    # Compile the regex pattern for efficiency
    regex = re.compile(pattern)
    data = []
    _type1 = int if headers[0] == 'iteration' else float
    _type2 = float

    # Loop over all files in the directory recursively
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"Searching in {file_path}")
            # Open each file and search for the pattern line by line
            with open(file_path, 'r') as f:
                for line_no, line in enumerate(f, start=1):
                    match = regex.search(line)
                    
                    if match:
                        # Extract the filename, line number, and captured values from regex groups
                        # print(f"Found match {line}")
                        filename = file_path
                        log_date = parse_datetime_from_log_path(file_path)
                        lineno = line_no
                        d1 = _type1(match.group(1))
                        d2 = _type2(match.group(2))

                        # Append extracted data to list as a dictionary
                        data.append({
                            "filename": filename,
                            "lineno": lineno,
                            "log_date": log_date,
                            headers[0]: d1,
                            headers[1]: d2
                        })
    
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)
    df = df.sort_values(by='iteration') if headers[0] == 'iteration' else df.sort_values(by='filename')
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default=LOG_DIR, help='Directory to search for log files')
    parser.add_argument('--pattern', choices=['checkpoint','startup'], default='checkpoint', help='Regex pattern to search for')
    parser.add_argument('--output_dir', default='local_logs', help='Path to save the output CSV file')
    args = parser.parse_args()
    
    print(f"Searching for pattern '{args.pattern}' in directory '{args.directory}'")
    if args.pattern == 'checkpoint':
        pattern = CHECKPOINT_PATTERN
    elif args.pattern == 'startup':
        pattern = STARTUP_TIME_PATTERN
    else:
        raise ValueError(f"Invalid pattern: {args.pattern}")
    
    headers = CHECKPOINT_HEADERS if args.pattern == 'checkpoint' else STARTUP_HEADERS
    df = search_pattern_in_dir(args.directory, pattern, headers)
    print(df.head())
    print(df.tail())
    # print(df.describe()['checkpoint_time'])

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_path = os.path.join(args.output_dir, f"{args.pattern}_times.csv")
    df.to_csv(output_path, index=False)