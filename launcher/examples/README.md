## Examples

Self-contained examples that demonstrate how to use the script generator `generate_distributed_launcher.py`.

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
  - Start / stop governed by `cudaProfilerApi`
  - The default cycle is to warmup for 10 steps and profile for 1 step
  - Traces: `cuda,nvtx,osrt,cudnn,cublas`
  - `nvtx` annotations for main train loop steps (`forward`, `backward`, `optimizer-step`) and torch `autograd` ops
  - See `python generate_srun_launcher.py --help` for all exposed `nsys` profiling knobs 
- NOTE: the 40-b profiles are large, so the output directory is `/lustre/.../$USER` and not local directory.

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