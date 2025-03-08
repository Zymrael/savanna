# I'll load the two YAML files and compare their contents, highlighting the differences.
import yaml
import argparse
from pathlib import Path


# Function to compare two dictionaries and find differences
def compare_dicts(dict1, dict2, path=""):
    differences = []
    for key in dict1.keys() | dict2.keys():
        full_path = f"{path}/{key}" if path else key
        if key not in dict2:
            differences.append(f"{full_path} only in first file")
        elif key not in dict1:
            differences.append(f"{full_path} only in second file")
        elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            differences.extend(compare_dicts(dict1[key], dict2[key], path=full_path))
        elif dict1[key] != dict2[key]:
            differences.append(f"{full_path} differs: {dict1[key]} != {dict2[key]}")
    return differences


def compare_configs(file1_path, file2_path):
    # Load both files
    print(f"Comparing {file1_path.stem} and {file2_path.stem}")
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        data1 = yaml.safe_load(file1)
        data2 = yaml.safe_load(file2)

    # Function to compare two dictionaries and find differences
    def compare_dicts(dict1, dict2, path=""):
        differences = []
        for key in dict1.keys() | dict2.keys():
            full_path = f"{path}/{key}" if path else key
            if key not in dict2:
                differences.append(f"{full_path} only in first file")
            elif key not in dict1:
                differences.append(f"{full_path} only in second file")
            elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                differences.extend(compare_dicts(dict1[key], dict2[key], path=full_path))
            elif dict1[key] != dict2[key]:
                differences.append(f"{full_path} differs: {dict1[key]} != {dict2[key]}")
        return differences

    # Run comparison
    differences = compare_dicts(data1, data2)
    for diff in differences:
        print(diff)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two YAML files")
    parser.add_argument("file1", type=Path, help="Path to the first file")
    parser.add_argument("file2", type=Path, help="Path to the second file")
    args = parser.parse_args()
    compare_configs(args.file1, args.file2)