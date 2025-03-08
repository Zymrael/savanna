import argparse
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_log_to_dataframe(logfile_path, debug=False):
    # Regex patterns to extract data
    timestamp_pattern = r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]"
    bottom_pattern = r"Bottom 32 Ranks with lowest  Etpt\(TF\): ([\d., /]+)"
    top_pattern = r"Top    32 Ranks with highest Etpt\(TF\): ([\d., /]+)"

    data = {
        'timestamp': [],
        'bottom_32_ranks': [],
        'bottom_32_etpt': [],
        'top_32_ranks': [],
        'top_32_etpt': []
    }

    with open(logfile_path, 'r') as file:
        for line in file:
            # Find timestamp and Etpt information
            timestamp_match = re.search(timestamp_pattern, line)
            bottom_match = re.search(bottom_pattern, line)
            top_match = re.search(top_pattern, line)

            if timestamp_match:
                # Extract timestamp
                timestamp = datetime.strptime(timestamp_match.group(1), "%Y-%m-%d %H:%M:%S")
                data['timestamp'].append(timestamp)

            if bottom_match:
                # Extract bottom 32 ranks and Etpt values
                bottom_etpt_data = bottom_match.group(1).split(', ')
                try:
                    bottom_32_etpt = [float(etpt.split('/')[0]) for etpt in bottom_etpt_data if etpt]
                    if len(bottom_32_etpt) < 32:
                        if debug:
                            print(f"Missing entries in bottom 32 etpt")
                            print(f"Line: {line}")
                            print(f"Bottom match: {bottom_etpt_data}")
                        continue
                    bottom_32_ranks = [int(etpt.split('/')[1].replace(',','')) for etpt in bottom_etpt_data if etpt]
                except Exception as e:
                    print(f"Line: {line}")
                    print(f"Bottom match: {bottom_etpt_data}")
                    raise e
                finally:
                    data['bottom_32_etpt'].append(bottom_32_etpt)
                    data['bottom_32_ranks'].append(bottom_32_ranks)

            if top_match:
                # Extract top 32 ranks and Etpt values
                top_etpt_data = top_match.group(1).split(', ')

                top_32_etpt = [float(etpt.split('/')[0]) for etpt in top_etpt_data if etpt]
                if len(top_32_etpt) < 32:
                    if debug:
                        print(f"Missing entries in top 32 etpt")
                        print(f"Line: {line}")
                        print(f"Top match: {top_etpt_data}")
                top_32_ranks = []
                for etpt in top_etpt_data:
                    try:
                        top_32_ranks.append(int(etpt.split('/')[1].replace(',','')))
                    except:
                        top_32_ranks.append(-1)
                data['top_32_etpt'].append(top_32_etpt)
                data['top_32_ranks'].append(top_32_ranks)

    # Create DataFrame
    df = pd.DataFrame(data)
    return df

def summarize_rank_distribution(df):
    bottom_ranks = [rank for ranks in df['bottom_32_ranks'] for rank in ranks]
    top_ranks = [rank for ranks in df['top_32_ranks'] for rank in ranks]
    bottom_rank_counts = Counter(bottom_ranks)
    top_rank_counts = Counter(top_ranks)

    print("Bottom rank counts:")
    print(bottom_rank_counts.most_common(20))

    print("Top rank counts:")
    print(top_rank_counts.most_common(20))
    bottom_count_df = pd.DataFrame(bottom_rank_counts.most_common(), columns=['Rank', 'Count'])
    top_count_df = pd.DataFrame(top_rank_counts.most_common(), columns=['Rank', 'Count'])
   
    print("Bottom rank count distribution")
    print(bottom_count_df.Count.describe())
    print("Top rank count distribution")
    print(top_count_df.Count.describe())
    bottom_count_df.to_csv('bottom_rank_counts.csv')
    top_count_df.to_csv('top_rank_counts.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfile", type=Path, required=True, help="Path to the log file")
    args = parser.parse_args()

    df = parse_log_to_dataframe(args.logfile, debug=False)
    df['bottom_avg_etpt'] = df['bottom_32_etpt'].map(lambda x: sum(x) / len(x))
    df['top_avg_etpt'] = df['top_32_etpt'].map(lambda x: sum(x) / len(x))
    df['avg_etpt_gap'] = df['top_avg_etpt'] - df['bottom_avg_etpt']
    print(" - " * 20)
    summarize_rank_distribution(df)

    print(" - " * 20)
    pd.options.display.max_colwidth = 1000
    pd.options.display.max_columns = 1000
    
    print("Average Etpt gap distribution")
    print(df.avg_etpt_gap.describe())
    out_path = args.logfile.stem + '_parsed.csv'
    df.to_csv(out_path)
    # Get counter of bottom ranks and top ranks


    # rank_counts = df['top_32_ranks'].map(lambda x: len(x)).value_counts()
    # print(rank_counts)