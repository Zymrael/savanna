# I'll load the two YAML files and compare their contents, highlighting the differences.
import argparse
import re
from pathlib import Path

import yaml

IGNORE_KEYS = ["cgcg", "iteration", "load", "master_port", "save_retain_interval", "log_memory", "partition-activations", "sequence_parallel", "finetune", "per_ds", "num_workers", "model_parallel"]
CONFIGS_DIR = Path(__file__).resolve().parent.parent / "model_configs"
DEFAULT_REF_DIR = Path(CONFIGS_DIR / "reference")
DEFAULT_TEST_DIR = Path(CONFIGS_DIR / "generated")
ROPE_SCALES = ["hybrid","linear", "log", "evo1", "5x"]
# Function to compare two dictionaries and find differences
def compare_dicts(dict1, dict2, path=""):
    differences = []
    for key in dict1.keys() | dict2.keys():
        full_path = f"{path}/{key}" if path else key
        if any(re.search(k, key) for k in IGNORE_KEYS):
            continue
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two YAML files")
    parser.add_argument("--ref_dir", default=DEFAULT_REF_DIR, type=Path, help="Path to the reference directory")
    parser.add_argument("--test_dir", default=DEFAULT_TEST_DIR, type=Path, help="Path to the directory to compare")
    parser.add_argument("--ref_config", default=None, type=Path, help="Path to the first file")
    parser.add_argument("--test_config", default=None, type=Path, help="Path to the second file")
    args = parser.parse_args()
    if args.ref_config and args.test_config:
        compare_configs(args.ref_config, args.test_config)
    else:
        assert args.ref_dir.is_dir(), f"Reference directory {args.ref_dir} does not exist"
        assert args.test_dir.is_dir(), f"Test directory {args.test_dir} does not exist"
        
        ref_configs = list(args.ref_dir.glob("*.yml"))
        test_configs = list(args.test_dir.glob("*.yml"))

        for scale in ROPE_SCALES:
            ref_configs = sorted(list(args.ref_dir.glob(f"*{scale}*.yml")))
            test_configs = sorted(list(args.test_dir.glob(f"*{scale}*.yml")))
           # print(f"Comparing {scale} configs")
            for ref_config, test_config in zip(ref_configs, test_configs):
                compare_configs(ref_config, test_config)
                print("=" * 80)

        # linear_ref_configs = sorted([config for config in ref_configs if "linear" in config.stem])
        # log_ref_configs = sorted(list(set(ref_configs) - set(linear_ref_configs)))
        # linear_configs = sorted(list(args.test_dir.glob("*linear*.yml")))
        # log_configs = sorted(list(args.test_dir.glob("*log*.yml")))
        # evo1_configs = sorted(list(args.test_dir.glob("*evo1*.yml")))
        # evo2_configs = sorted(list(args.test_dir.glob("*evo2*.yml")))
        ## print delimiter for better readability

        # for ref_config, test_config in zip(linear_ref_configs, linear_configs):
        #     # print header for better readability denoting configs being compared
        #     compare_configs(ref_config, test_config)
        #     print("-" * 80)

        # print("=" * 80)
        # for ref_config, test_config in zip(log_ref_configs, log_configs):
        #     # print header for better readability denoting configs being compared
        #     compare_configs(ref_config, test_config)
        #     print("-" * 80)
