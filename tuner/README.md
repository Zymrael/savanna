## Tuner
General purpose utility for generating and running experiments with SLURM.

### Features
- Grid search across param values, e.g.:
    - Search across array of `micro_batch_size_per_gpu` for a given model config
    - Define multiple parallelism configs to search over from a given model template
    - All generated artifacts (configs, logs, scripts) written to common location for reproducibility
- Simplifies `wandb` run organization
    - `group`, and `run_id` names will be generated based on model config and search space
    - Runs will be grouped by template name
    - Each run will be named based on grid search config (one for each unique param combination)
    - Enables easier organization and documentation of runs for reproducibility and collaboration.
- Integrated `nsys` support
    - Pre-configured `nsys` runner with sensible default
    - Fully configurable profiling options through the commandline 
- Standalone SLURM / Python runner
    - Can be used for running individual jobs (no search config, just a template model config)
    - Automatically generates slurm sbatch script
    - Also generates standalone python script which can be directly executed on interactive compute node
- Dynamic config overrides on CLI
    - Can dynamically update config from the CLI
    - Updated config will be written for reproducibility

### Usage

#### Grid Search
`cd` into `tuner` directory from `savanna` root

```
python tuner.py --data-config data_configs/minimal_data_config.yml --template templates/regression_base.yml --search-config search_configs/batchsizes.yml --wandb_project tuner-test
```

where `templates` and `search_configs` are directories within the `tuner` directory, and `batchsizes.yml` is the search space config.  Only difference between a search space config and typical model config is that the former can specify lists in place of scalars to define the search space.  

E.g., `batchsizes.yml`:
```yaml
{
  "train_micro_batch_size_per_gpu": [1, 2, 4],
}
```

Running the command above should output:
```text
Generated 3 config files for template `regression_base` and data config `opengenome`:
    /home/jeromeku/savanna/tuner/generated_configs/regression_base/202409281048/configs/config_0__train_micro_batch_size_per_gpu=1.yml
    /home/jeromeku/savanna/tuner/generated_configs/regression_base/202409281048/configs/config_1__train_micro_batch_size_per_gpu=2.yml
    /home/jeromeku/savanna/tuner/generated_configs/regression_base/202409281048/configs/config_2__train_micro_batch_size_per_gpu=4.yml
Generating scripts...
Found 3 config files.
SLURM job script created: /home/jeromeku/savanna/tuner/generated_configs/regression_base/202409281048/slurm_job.sh
Created standalone script: /home/jeromeku/savanna/tuner/generated_configs/regression_base/202409281048/config_0__train_micro_batch_size_per_gpu=1-standalone.sh
Created standalone script: /home/jeromeku/savanna/tuner/generated_configs/regression_base/202409281048/config_1__train_micro_batch_size_per_gpu=2-standalone.sh
Created standalone script: /home/jeromeku/savanna/tuner/generated_configs/regression_base/202409281048/config_2__train_micro_batch_size_per_gpu=4-standalone.sh
Run all standalone scripts written at /home/jeromeku/savanna/tuner/generated_configs/regression_base/202409281048/run_all_standalones.sh
Done generating scripts for template `regression_base`
```

The generated scripts are as follows:
- `slurm_job.sh` - sbatch script that can be directly submitted with `sbatch slurm_job.sh`
- `config_{x}__train_micro_batch_size_per_gpu={x}-standalone.sh` - these are standalone bash scripts that be directly run on the existing compute node `./config_0__xxx-standalone.sh` individually or `./run_all_standalones.sh` to run all configs (sequentially).
- the `configs` sub-directory contains the generated model configs -- the original template updated with the respective search config values.  E.g., `config_0__train_micro_batch_size_per_gpu=0.yml` is the template with `train_micro_batch_size_per_gpu` set to `0`. 

The above script will only generate the configs, if you wish to submit the SLURM script as well, simply append `--submit-slurm` to the command, which will submit the generated `slurm_job.sh` and output the following:  

```text
 ...
 Submitting SLURM job array with: sbatch /home/jeromeku/savanna/tuner/generated_configs/regression_base/202409281048/slurm_job.sh
 Submitted batch job 32854
 SLURM job array submitted successfully. Logging to /home/jeromeku/savanna/tuner/generated_configs/regression_base/202409281048/logs
 Run can be tracked on wandb at project/group: tuner-test/regression_base
```

Resources can be specified at the commandline:
```text
python tuner.py --template ... --num-nodes 2 --num-gpus 4 --job-time 01:00:00 --submit-slurm
```
Will generate a slurm script with the requested resources (2 nodes, 4 gpus each, run for 1 hour) and will also generate the necessary hostfile for multi-node training.

#### Logging
Running the slurm script above will log to `wandb` (if `use_wandb` is enabled in the template config).
- The project will be as specified on the commandline (i.e., `tuner-test`)
- Runs will be grouped (share the same `wandb_group`) by the template file name; more than 1 template can be provided
- Individual runs will be named based on the config file to easily distinguish different runs by search param within and across groups.

#### Standalone Job Runner
You can also omit search configs and provide only a template, in which case a single config will be generated along with the previously artifacts -- slurm sbatch script, standalone script, etc.  

You can also override any of the keys in the template by providing `key=value` pairs with the `--overrides` flag:
```text
python tuner.py --template templates/regression_base.yml --overrides train-iters=10
```

The slurm job and other scripts can be run per above, only difference is a single job will be submitted for slurm instead of a job array.

#### Profiling
Appending `nsys` to the above command will generate same configs, except the generated SLURM job and standalone python runners are now configured to run with `nsys` with sensible defaults.  

- Even if the provided template model config does not have profiling enabled, it will be updated to enable profiling with `nsys` with `--nsys` flag is specified.

- `nsys` profiling options can be fully configured as well:
    ```
    python tuner.py --help

    usage: tuner.py [-h] [--search-config SEARCH_CONFIG] [--template TEMPLATE [TEMPLATE ...]] [--data-config DATA_CONFIG] [--output_dir OUTPUT_DIR]
                    [--wandb_project WANDB_PROJECT] [--wandb_group WANDB_GROUP] [--overwrite-output-dir] [--submit-slurm] [--standalone-scripts]
                    [--calculate-gas] [--num-nodes NUM_NODES] [--num-gpus NUM_GPUS] [--job-time JOB_TIME] [--nsys]
                    [--gpu-metrics-device GPU_METRICS_DEVICE] [--cuda-memory-usage CUDA_MEMORY_USAGE] [--cudabacktrace CUDABACKTRACE]
                    [--python-sampling PYTHON_SAMPLING] [--capture-range CAPTURE_RANGE] [--stats STATS] [--nic-metrics NIC_METRICS]
                    [--show-output SHOW_OUTPUT] [--trace TRACE] [--sample SAMPLE] [--force-overwrite FORCE_OVERWRITE] [--stop-on-exit STOP_ON_EXIT]
                    [--inherit-environment INHERIT_ENVIRONMENT] [--wait WAIT] [--overrides [OVERRIDES ...]]

    options:
    -h, --help            show this help message and exit
    --search-config SEARCH_CONFIG
    --template TEMPLATE [TEMPLATE ...]
                            Path to template file, can be a list (default: /home/jeromeku/savanna/tuner/templates)
    --data-config DATA_CONFIG
    --output_dir OUTPUT_DIR
    --wandb_project WANDB_PROJECT
    --wandb_group WANDB_GROUP
    --overwrite-output-dir
                            Overwrite output directory (default: False)
    --submit-slurm        Generate configs and submit to SLURM (default: False)
    --standalone-scripts  Generate standalone scripts (independent of SLURM) (default: False)
    --calculate-gas       Calculate gas based on train_batch_size andtrain_micro_batch_size_per_gpu (default: False)
    --num-nodes NUM_NODES
    --num-gpus NUM_GPUS
    --job-time JOB_TIME
    --nsys
    --gpu-metrics-device GPU_METRICS_DEVICE
                            `nsys` flag: GPU metrics device (default: none)
    --cuda-memory-usage CUDA_MEMORY_USAGE
                            `nsys` flag: CUDA memory usage flag (default: true)
    --cudabacktrace CUDABACKTRACE
                            `nsys` flag: CUDA backtrace flag (default: false)
    --python-sampling PYTHON_SAMPLING
                            `nsys` flag: Python sampling flag (default: false)
    --capture-range CAPTURE_RANGE
                            `nsys` flag: Capture range for profiling (default: cudaProfilerApi)
    --stats STATS         `nsys` flag: Stats flag (default: false)
    --nic-metrics NIC_METRICS
                            `nsys` flag: NIC metrics flag (default: true)
    --show-output SHOW_OUTPUT
                            `nsys` flag: Show output flag (default: true)
    --trace TRACE         `nsys` flag: Trace options for nsys (default: cuda,nvtx,osrt,cudnn,cublas-verbose)
    --sample SAMPLE       `nsys` flag: Sampling method (default: process-tree)
    --force-overwrite FORCE_OVERWRITE
                            `nsys` flag: Force overwrite flag (default: true)
    --stop-on-exit STOP_ON_EXIT
                            `nsys` flag: Stop on exit flag (default: false)
    --inherit-environment INHERIT_ENVIRONMENT
                            `nsys` flag: Inherit environment flag (default: true)
    --wait WAIT           `nsys` flag: Wait mode (default: all)
    --overrides [OVERRIDES ...]
                            Overrides in the form of key=value (e.g., learning_rate=0.01, model.hidden_size=1024) (default: None)
    ```                        
- `nsys` options are marked as such with their defaults shown.  These options map directly to what will be passed to `nsys`.

### TODO
- [ ] Add more examples