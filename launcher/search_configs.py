import copy
import itertools
import re
from pathlib import Path
from typing import List, Union

from launcher.utils import load_yaml, yaml

MAX_FILENAME_LENGTH = 150
# Abbreviations for the most common parameters
# copilot add additional keys by replacing hyphens with underscores
ABBREVIATIONS = {
    "train_micro_batch_size_per_gpu": "mbs",    
    "sequence_parallel": "sp",
    "model_parallel_size": "mp",
    "gradient_accumulation_steps": "gas",
    "checkpoint_num_layers": "ac",
    "train-micro-batch-size-per-gpu": "mbs",
    "sequence-parallel": "sp",
    "model-parallel-size": "mp",
    "gradient-accumulation-steps": "gas",
    "checkpoint-num-layers": "ac",
    "True": "T",
    "False": "F",
}



# Recursively update nested dictionaries
def recursive_update(d, u):
    for k, v in u.items():
        try:
            if isinstance(v, dict):
                d[k] = recursive_update(d.get(k, {}), v)
            else:
                d[k] = v
        except Exception as e:
            print(f"Error updating {k}: {e}")
    return d


def flatten_dict(d: dict, parent_key: str = "") -> dict:
    """
    Flattens a nested dictionary into a single-level dictionary.
    """

    items = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


# Expand the parameters with multiple values into all combinations
def expand_search_space(search_space):
    flattened = flatten_dict(search_space)

    keys = []
    values = []
    updated_keys = []

    for key, val in flattened.items():
        if isinstance(val, list):
            keys.append(key)
            values.append(val)
            updated_keys.append(key)
        else:
            keys.append(key)
            values.append([val])

    # Generate all combinations of the multi-value parameters
    param_combinations = list(itertools.product(*values))

    expanded_configs = []
    for combo in param_combinations:
        config = {}
        for i, key in enumerate(keys):
            keys_hierarchy = key.split(".")
            d = config
            for part in keys_hierarchy[:-1]:
                d = d.setdefault(part, {})
            d[keys_hierarchy[-1]] = combo[i]
        expanded_configs.append(config)
    return expanded_configs, updated_keys


def abbreviate(s):

    for phrase, abbreviation in ABBREVIATIONS.items():
        s = s.replace(phrase, abbreviation)

    return s


# Generate descriptive filename based on updated parameters
def generate_filename(base_name, updated_params):
    filename = base_name
    for param, value in updated_params.items():
        param_name = param.replace(".", "_")
        if isinstance(value, list):
            value = ",".join([str(v) for v in value])

        value_str = str(value).replace(".", "p")
        filename += f"_{param_name}={value_str}"
        # Replace repeated zero_optimization_ in filename
        filename = filename.replace("zero_optimization_", "")
    
    filename = abbreviate(filename)
    filename = filename[:MAX_FILENAME_LENGTH]
    return filename + ".yml"


# Generate new config files for each combination
def generate_configs(
    template_yaml: Union[Path, dict], search_config_yaml: Path, output_dir: Path, job_name: str = None
) -> List[Path]:
    search_config_dir = output_dir / "search_configs"
    search_config_dir.mkdir(parents=True, exist_ok=True)

    template = load_yaml(template_yaml) if isinstance(template_yaml, Path) else template_yaml
    search_space = load_yaml(search_config_yaml)
    # Step 1: Expand the user-defined search space
    expanded_configs, param_keys = expand_search_space(search_space)

    # Step 2: Merge each expanded config with the template YAML
    config_count = 0
    config_paths = []

    template_save_path = output_dir / "template.yml"
    with open(template_save_path, "w") as template_file:
        yaml.dump(template, template_file)

    base_name = "config"  # search_config_yaml.stem
    for expanded_config in expanded_configs:
        # Merge expanded config with the template
        final_config = copy.deepcopy(template)
        recursive_update(final_config, expanded_config)

        flattened_config = flatten_dict(final_config)

        updated_params = {k: v for k, v in flattened_config.items() if k in param_keys}

        if len(updated_params) == 0:
            config_filename = f"{base_name}_{config_count}_{list(expanded_config.keys())[0]}.yml"
        else:
            config_filename = generate_filename(f"{base_name}_{config_count}", updated_params)

        if job_name is not None:
            config_filename = f"{job_name}_{config_filename}"

        config_filepath = search_config_dir / config_filename

        with open(config_filepath, "w") as config_file:
            yaml.dump(final_config, config_file)

        config_count += 1
        config_paths.append(config_filepath)

    return search_config_dir, config_paths
