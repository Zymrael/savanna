## `pdsh`-less Distributed Launcher

### Background
#### Problem
The existing launcher `launch.py` uses `pdsh` to coordinate distributed launch, which is not compatible with NVIDIA's containerized cluster env.

`launch.py`
- Typically, we launch multi-node training by calling `python launch.py train.py data_config model_config`
  - This is performing 2 primary functions under the hood:
    - **Config parsing** - parses the the data and model configs, separates deepspeed-specific args from those specific to the training script, and encodes these args as base64-encoded strings
    - **Distributed launch** - passes these encoded args to `deepspeed.launcher.runner.main`
      - `deepspeed.launcher.runner.main` in turn uses `pdsh` to launch the module script `deepspeed.launcher.launch` with the provided args
      - `deepspeed.launcher.launch` is a wrapper around `torchrun`, which is to say, its responsibility is to spawn local processes at the local node level -- one for each GPU.

#### Solution
`srun` + `torchrun / deepspeed` 
- What I've done is to separate the 2 functions performed by `launch.py` into two separate job steps within a SLURM batch job.
  - **Config parsing** - The responsibility of this step is to parse the configs and generate the parsed args in the expected format for downstream consumption by `train.py`. This is done only on the `MASTER_NODE` so that only a single set of configs is broadcasted to all nodes.  This is done by `config_parser.py`. 
  - **Distributed launch** - In place of `deepspeed.launcher.runner.main`:
    - `srun` is used to launch a single (containerized) process on each node.
    - Each process, in turn, runs either `torchrun` or `deepspeed.launcher.launch` which perform the following:
      - it spawns local processes (each executing the train script `train.py`) per number of local GPUs
      - sets the env vars necessary for distributed coordination (`WORLD_SIZE`, `RANK`, `MASTER_ADDR`, etc.)
    - `deepspeed.launcher.launch` has the additional benefit of routing per-rank `stdout / stderr` to individual files for distributed logging.

_Update_ (10/11/2024)
- Added an `srun` only launcher that does not rely on `torchrun` or `deepspeed` to manage the distributed environment
- The benefit of using `srun` only is that it enables control over the per-rank processes, which is necessary for downstream optimization such as NUMA process binding.
- Compared to `torchrun / deepspeed`, `SLURM / srun` manages `WORLD_SIZE` number of processes - so `SLURM_NTASKS == WORLD_SIZE` whereas with the former, `SLURM_NTASKS == NNODES`.

### Tests
#### Sanity checks
`savanna/launcher/tests/{torch,deepspeed,srun}.launcher.test.sh` 
- This test script runs the pipeline with a "mock" train script `savanna/launcher/tests/check_args.py` that:
  - checks that `GlobalConfig` can be correctly instantiated
  - checks that distributed env vars are set (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`)
  - attempts to initiate torch distributed env `torch.distributed.init_process_group`, which completes only if the distributed env is set up correctly.
- Added debug messages and assertions in `train.py` to check that the distributed env: `RANK`, `LOCAL_RANK`, `WORLD_SIZE` are set correctly when using the distributed launcher.

`savanna/launcher/tests/deepspeed-rank-log.sh`
- This tests that per-rank logging works when using the `deepspeed` launcher
- More specifically, you should see a `rank_logs` folder in the output directory with files for `stdout/stderr` for each rank after running the test.

#### Integration tests
Ran the following e2e tests for the following model configs / parallelisms to ensure that training runs to completion for a small number of iterations (20) using the `EFA` container for both the `torch` and `deepspeed` launchers:
- `regression_test`
  - `ZeRO-1` (DP2) - 2 nodes / 1 gpus per node
  - `ZeRO-1` (DP4) - 2 nodes / 2 gpus per node
- `7b_shc_post_refactor`
  - `MP2 + DP2` - 2 nodes / 2 gpus per node
  - `DP4` - 2 nodes / 2 gpus per node
- `40b_test_config`
  - `TP2 + DP16` - 4 nodes / 8 gpus per node

Additional notes:
- The test configs I used are located in `configs/launcher-test`
- I explicitly disabled checkpoint saving to avoid overloading disk -- this will result in ranks exiting with an error message (missing save path).  
- Examine the output logs of the `MASTER_NODE` to check that the "training" ran to completion (`sbatch` output log prints the `MASTER_NODE`).
- The training logs (iteration time, metrics per step, etc.) are only output on `MASTER_NODE`.

As another sanity check, I manually inspected that  `train-batch-size`, which is automatically derived based on distributed env (along with with user-provided `gradient_accumulation_steps` and `micro_batch_size_per_gpu`, is calculated correctly.
  - Specifically, `train-batch-size` depends on `WORLD_SIZE` being set correctly
      - `tbs` = `dp` * `gas` * `micro_bs_per_gpu`, where `dp` is the data parallel size
      - `dp = global_num_gpus // (MP * PP)`, where `global_num_gpus` is `WORLD_SIZE` and `mp = model parallel size`, `pp = pipeline parallel size`
    - E.g., with 2 nodes / 1 GPU per node, `mbs per GPU = 2`, `gas = 1`, `MP=PP=1` => `train_batch_size = 4`
    - For the 40b test config, where `TP=2, train_micro_batch_size=2, gas=1`, the `train_batch_size = 32` for the 4 node / 8 GPU setup.

## Usage
Most of the changes are isolated to the new `savanna/launcher` directory but some changes were required in the core code base to ensure proper distributed setup (see `arguments.py` and `global_config.py` diffs).

The entrypoint is `savanna/launcher/generate_srun_launcher.py`
- This python script takes a number of `SLURM`, `pyxis`, and `train.py` args
- It will automatically generate a SLURM sbatch script based on user provided args that runs the pipeline described above (setting up env, calling `config_parser` and then running `torchrun` on each node), with all values properly configured.

Given the extensive number of config options, I've done my best to create sensible defaults, which might need to be changed based on your local directory structure:
- `container-mounts`
  - `data` directory - mounts the `evo2` scratch data directory (`/lustre/fs01/portfolios/dir/projects/dir_arc/evo`) to `/data` in the container.  Note that the example data configs in `configs/launcher-test/data_configs` work only under this assumption.
  - `savanna` root directory - mounts the `savanna` root directory to the same directory in the container.  I.e., if your `savanna` root is at `/home/jeromek/savanna`, it will be mapped as such in the container.  This is important, since we are not using `savanna` as an installed python package but rather as a package that is in `PYTHONPATH`. 
- `data-config` and `model-config` - best to situate configs in the `configs` directory of `savanna` root to ensure that they are discoverable within the container, assuming `savanna` root is mapped per default.

The full menu of options:
```
python generate_distributed_launcher.py --help
usage: generate_distributed_launcher.py [-h] [--launcher {torch,deepspeed,srun}] [--account ACCOUNT] [--partition PARTITION] [--num-nodes NUM_NODES] [--num-gpus NUM_GPUS]
                                        [--cpus-per-task CPUS_PER_TASK] [--mem MEM] [--job-time JOB_TIME] [--pyxis] [--job-array] [--no-pyxis] [--container CONTAINER]
                                        [--container-mounts CONTAINER_MOUNTS] [--container-workdir CONTAINER_WORKDIR] [--data-dir DATA_DIR] [--log-dir LOG_DIR] [--root ROOT]
                                        [--train-script TRAIN_SCRIPT] [--data-config DATA_CONFIG] [--model-config MODEL_CONFIG] [--hostlist HOSTLIST]
                                        [--train-args-output TRAIN_ARGS_OUTPUT] [--config-parser CONFIG_PARSER] [--output-dir OUTPUT_DIR] [--checkpoint_dir CHECKPOINT_DIR]
                                        [--use-wandb] [--wandb-host WANDB_HOST] [--wandb-project WANDB_PROJECT] [--wandb-group WANDB_GROUP] [--search-config SEARCH_CONFIG]
                                        [--overrides [OVERRIDES ...]] [--nsys] [--nsys_warmup_steps NSYS_WARMUP_STEPS] [--nsys_num_steps NSYS_NUM_STEPS]
                                        [--gpu-metrics-device GPU_METRICS_DEVICE] [--cuda-memory-usage CUDA_MEMORY_USAGE] [--cudabacktrace CUDABACKTRACE]
                                        [--python-sampling PYTHON_SAMPLING] [--capture-range CAPTURE_RANGE] [--stats STATS] [--nic-metrics NIC_METRICS] [--show-output SHOW_OUTPUT]
                                        [--trace TRACE] [--sample SAMPLE] [--force-overwrite FORCE_OVERWRITE] [--stop-on-exit STOP_ON_EXIT] [--wait WAIT] [--enable-each-rank-log]
                                        [--master_port MASTER_PORT] [--rdzv_backend RDZV_BACKEND] [--max-restarts MAX_RESTARTS] [--verbose]
                                        job_name

Generate launcher SLURM script launching distributed training jobs without `pdsh`.

positional arguments:
  job_name              `SLURM` - name of the SLURM job.

options:
  -h, --help            show this help message and exit
  --launcher {torch,deepspeed,srun}
                        Type of launcher (default: srun)
  --account ACCOUNT     `SLURM` - account to use. (default: dir_arc)
  --partition PARTITION
                        `SLURM` - partition (default: pool0)
  --num-nodes NUM_NODES
                        `SLURM` -number of nodes. (default: 2)
  --num-gpus NUM_GPUS   `SLURM` - number of GPUs per node. (default: 8)
  --cpus-per-task CPUS_PER_TASK
                        `SLURM` - number of CPUs per task. (default: 4)
  --mem MEM             `SLURM` - memory per node. (default: 100G)
  --job-time JOB_TIME   `SLURM` - job time limit. (default: 01:00:00)
  --pyxis               use `pyxis` to run containerized SLURM jobs. (default: True)
  --job-array           use SLURM job array. (default: False)
  --no-pyxis            disable pyxis (i.e., `srun` in no container mode). (default: True)
  --container CONTAINER
                        `pyxis` - container image (default: /lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/nvidia_evo2_efa_latest.sqsh)
  --container-mounts CONTAINER_MOUNTS
                        `pyxis` - container mounts. (default: $DATA_DIR:/data,$ROOT_DIR:$ROOT_DIR,$OUTPUT_DIR:$OUTPUT_DIR,$CHECKPOINT_DIR:$CHECKPOINT_DIR)
  --container-workdir CONTAINER_WORKDIR
                        `pyxis` - container workdir. (default: $ROOT_DIR)
  --data-dir DATA_DIR   Location of evo2 data. (default: /scratch/fsw/portfolios/dir/projects/dir_arc/evo2/data)
  --log-dir LOG_DIR     Directory to store logs. (default: logs)
  --root ROOT           Root directory where `savanna` repo is located. E.g., `/home/$USER/savanna`. (default: /home/jeromek/savanna-upstream)
  --train-script TRAIN_SCRIPT
                        Training script name, most likely `savanna/train.py`. Note: ensure `savanna` root is mapped correctly in the container. (default: /home/jeromek/savanna-
                        upstream/train.py)
  --data-config DATA_CONFIG
                        Path to data config. Note: ensure the location is relative to location in the *container*. (default: /home/jeromek/savanna-upstream/configs/launcher-
                        test/data_configs/nvidia-og.yml)
  --model-config MODEL_CONFIG
                        Path to model config. Note: ensure the location is relative to location in the *container*. (default: /home/jeromek/savanna-upstream/configs/launcher-
                        test/model_configs/regression_test.yml)
  --hostlist HOSTLIST   Hostlist file name where SLURM host names are stored. Primarily for debugging. (default: hostlist)
  --train-args-output TRAIN_ARGS_OUTPUT
                        Output file for parsed train args. (default: train_args.txt)
  --config-parser CONFIG_PARSER
                        Config parser script name. Only change this if debugging. (default: /home/jeromek/savanna-upstream/launcher/config_parser.py)
  --output-dir OUTPUT_DIR
                        Output directory for logs and other artifacts, defaults to job name. (default: None)
  --checkpoint_dir CHECKPOINT_DIR
                        checkpoint directory (default: /scratch/fsw/portfolios/dir/projects/dir_arc/evo2/checkpoints)
  --use-wandb           Use `wandb` for logging. (default: False)
  --wandb-host WANDB_HOST
                        Wandb host. (default: https://api.wandb.ai)
  --wandb-project WANDB_PROJECT
                        Wandb project. (default: None)
  --wandb-group WANDB_GROUP
                        Wandb group. (default: None)
  --search-config SEARCH_CONFIG
                        Path to yml search config. (default: None)
  --overrides [OVERRIDES ...]
                        Overrides in the form of key=value (e.g., learning_rate=0.01, model.hidden_size=1024) (default: None)
  --nsys                Enable `nsys` profiling. (default: False)
  --nsys_warmup_steps NSYS_WARMUP_STEPS
                        Number of warmup steps before startin `nsys` profiler (default: 10)
  --nsys_num_steps NSYS_NUM_STEPS
                        Number of active `nsys` profiler steps (default: 1)
  --gpu-metrics-device GPU_METRICS_DEVICE
                        `nsys` flag: GPU metrics device (default: none)
  --cuda-memory-usage CUDA_MEMORY_USAGE
                        `nsys` flag: CUDA memory usage flag (default: false)
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
  --trace TRACE         `nsys` flag: Trace options for nsys (default: cuda,nvtx)
  --sample SAMPLE       `nsys` flag: Sampling method (default: none)
  --force-overwrite FORCE_OVERWRITE
                        `nsys` flag: Force overwrite flag (default: true)
  --stop-on-exit STOP_ON_EXIT
                        `nsys` flag: Stop on exit flag (default: true)
  --wait WAIT           `nsys` flag: Wait flag (default: false)
  --enable-each-rank-log
                        Enable each rank log when using deepspeed launcher. (default: False)
  --master_port MASTER_PORT
                        Master port for distributed training, automatically generated if not specified. (default: None)
  --rdzv_backend RDZV_BACKEND
                        `torchrun` - rdzv backend (default: c10d)
  --max-restarts MAX_RESTARTS
                        `torchrun` - maximum number of restarts (default: 0)
  --verbose             Prints script to stdout. (default: False)
  ```

### Examples
I've created an `examples` directory to demonstrate how to use the script generator.

`savanna/launcher/examples/7b-demo.sh`
- Run from `launcher` directory: `./examples/7b-demo.sh`.  
- This will run the 7b config `MP2 + DP8` (model parallel size 2, data parallel size 8) config on 2 nodes / 8 gpus per node.
- Should see output: "SLURM script generated: .../$JOB_NAME/.../7b_xxx.sbatch"
- Submit and check logs: `sbatch .../$JOB_NAME/.../7b_xxx.sbatch`
- Skim the generated SLURM sbatch script, which includes inline comments explaining what is happening at each step.

`savanna/launcher/examples/40b-demo-launcher`
- Same as above, except with 40b config run on 32 GPUs

`savanna/launcher/examples/{7,40}b-nsys-demo.sh`
- Runs the {7,40}b config with `nsys`
- Check the `nsys_reports` directory in the output folder for generated reports, one per node
- `nsys` config
  - Start / stop governed by cudaProfilerApi
  - The default cycle is to warmup for 10 steps and profile for 1 step
  - Traces: `cuda,nvtx,osrt,cudnn,cublas`
  - `nvtx` annotations for main train loop steps (`forward`, `backward`, `optimizer-step`) and torch `autograd` ops
  - See `python generate_srun_launcher.py --help` for all exposed `nsys` profiling knobs 
- NOTE: the 40-b profiles are large, so the output directory is /lustre and not local directory.

`savanna/launcher/examples/7b-search-config-demo.sh`
- Creates multiple scripts based on a template model config and a search config
- The search config is a yaml file with same format as the model config except scalars can be replaced by lists
- For example,
  ```json
  {
    "train_micro_batch_size_per_gpu": [1, 2, 4]
  } 
  ```
- The script will output a folder search_configs with all combinations of the template config substitued with search values: e.g., `config_0__train_micro_batch_size_per_gpu=1.yml`, `search_configs/config_1__train_micro_batch_size_per_gpu=2.yml`, etc.


`savanna/launcher/examples/7b-torch-profiler.sh`
- Runs the 7b config with torch profiler enabled
- Check the output directory for `torchprofiler_traces` folder which will contain per rank tensorboard profiles and key averages
- Default cycle is warmup 3, 
- See `global_config.py` for all available `torch.profiler` options.


**Important**:
- `logging` - the SLURM jobs are configured such that a log is generated for each node of each job step.  The first job step (config parsing) is only executed by the MASTER_NODE while the actual launch is executed on each node.  I.e., if running on 2 nodes, you'll see 3 `srun` output logs, 1 for step 1 for MASTER_NODE for config parsing and 2 for launch / training for both nodes.  Additionally, there will be an `sbatch` log which outputs useful env vars (node list, master node / master port, etc.); ditto for stderr logs.
- If using the `deepspeed` launcher you can enable each rank logging, where each rank writes stdout / stderr to a rank-specific file.  To use this feature, set the `enable_each_rank_log` flag to `true` in your model config.  See `tests/deepspeed-rank-log-test.sh` for an example.  Rank logs will be written to `rank_logs` sub-directory in the output directory.
- `data config` - the data config for all these tests defaults to `configs/launcher-test/data_configs/nvidia-og.yml`, which is the `opengenome.yml` dataset but with paths mapped according to the container mounts described above.  If using a different container mount configuration, you'll need to change these mappings.
- `model config` - ~~make to sure add `use_srun_launcher: true` if using your own model config, otherwise distributed env won't be configured correctly~~ The `generate_distributed_launcher.py` script will automatically set the required args in the model config.
- `wandb - if using `wandb` for monitoring (pass `--use-wandb` to `generate_srun_launcher`), make sure export your WANDB_API_KEY first; script checks for the condition.  

### Code changes
Changes made to the core `savanna` codebase:
- `arguments.py`
   - `consume_deepy_args` - This is typically called from `launch.py`.  I added commandline options to accommodate the modified pipeline.  Should not impact previous launch workflow (ran basic regression test to check that this is indeed so).
   - `calculate_derived` - This is called in `post_init` of `GlobalConfig` initialization.  Added a section for setting `global_num_gpus` correctly based on SLURM env vars.
- `global_config.py` 
  - added `use_srun_launcher` flag to enable this distributed workflow.  MUST set to true for correct distributed configuration.
  - added `srun_launcher_type` - this will be automatically populated by the script generator based on CLI args.
  - added `enable_each_rank_log` flag to enable per rank logging when using `deepspeed` launcher.
  - added 'savanna.distributed.py` for basic env setup checks called in `train.py`
- `train.py` - added debug messages and assertions (only enabled if `use_srun_launcher=True`, so won't impact default workflow) to check distributed setup.

### Caveats
- ~~`torchrun` seems to randomly assign ranks, as `MASTER_ADDR` is not always assigned global rank 0.  Need to verify that this does not result in downstream problems~~ fixed by passing `node_rank` to  `torchrun`.
- ~~`torch.compile` with `cudagraphs` enabled is not working for the 40b in the current container (as of 10/6).  Potential causes -- container env is not an exact replica of the native `evo2` env that the model was tested on in the Arc cluster -- python versions differ (3.10 vs 3.11), packages are not pinned to same versions, etc.~~ Resolved by updating container to latest `nccl`.