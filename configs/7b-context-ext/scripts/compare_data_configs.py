# I'll load the two YAML files and compare their contents, highlighting the differences.
import argparse
import re
from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "data_configs"
IGNORE_KEYS = {'data-weights', 'checkpoint_validation'}
DATA_KEYS = [f"{t}-data-paths" for t in ['train', 'valid', 'test']]

# Function to compare two dictionaries and find differences
def compare_dicts(d1, d2):
    differences = []

    for key in DATA_KEYS:
        p1, p2 = set(d1[key]), set(d2[key])
        only_in_first = p1 - p2
        only_in_second = p2 - p1
        if len(only_in_first) > 0:
            s = "\n ".join(only_in_first)
            differences.append(f"{key}: Only in first file:\n {s}")
        if len(only_in_second) > 0:
            s = "\n ".join(only_in_second)
            differences.append(f"{key}: Only in second file:\n {s}")
    return differences


def compare_configs(file1_path, file2_path):
    # Load both files
    print(f"Comparing {file1_path.stem} and {file2_path.stem}")
    with open(file1_path, "r") as file1, open(file2_path, "r") as file2:
        data1 = yaml.safe_load(file1)
        data2 = yaml.safe_load(file2)

    # Run comparison
    differences = compare_dicts(data1, data2)
    if len(differences) == 0:
        print("No differences found")
    else:
        for diff in differences:
            print(diff)
            print(" ---- ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two YAML files")
    parser.add_argument("ref_config", type=Path, help="Path to the first file")
    parser.add_argument("test_config", type=Path, help="Path to the second file")
    args = parser.parse_args()

    compare_configs(args.ref_config, args.test_config)
