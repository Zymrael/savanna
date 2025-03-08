## 40b Train Scripts


### Usage

From the `savanna/launcher` directory:
```
./40b/40b-xxx.sh
```

This will output something like the following (paths will be different):
```
python generate_distributed_launcher.py 40b-train-n128-testrun --expandable_segments --use-wandb --launcher srun --job-time 03:00:00 --partition pool0_datahall_a --account dir_arc --container /lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/nvidia_evo2_efa_latest.sqsh --num-nodes 128 --num-gpus 8 --data-config /lustre/fs01/portfolios/dir/users/jeromek/savanna-scratch/configs/40b/shortphase_v3_nvidia.yml --model-config /lustre/fs01/portfolios/dir/users/jeromek/savanna-scratch/configs/40b/40b_train.yml --train-script /lustre/fs01/portfolios/dir/users/jeromek/savanna-scratch/train.py --wandb-project 40b-train --wandb-run-name 40b-train-n128-testrun
----------------------------------------------------------------------------------------------------
No output directory specified. Defaulting to /lustre/fs01/portfolios/dir/users/jeromek/40b-train-n128-testrun/202410211755
Logs will be generated in /lustre/fs01/portfolios/dir/users/jeromek/40b-train-n128-testrun/202410211755/logs
Triton cache will be stored in /lustre/fs01/portfolios/dir/users/jeromek/40b-train-n128-testrun/202410211755/triton_cache
----------------------------------------------------------------------------------------------------
WARNING: Overriding `save` path in model config to /lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n128-testrun/40b_train/202410211755 to ensure local checkpoint is stored on Lustre
Setting remote store location to chkpt/40b-train-n128-testrun/40b_train
Setting use_srun_launcher True in model config.
Setting launcher type to srun
Setting wandb project/group/run name to 40b-train/202410211755/40b-train-n128-testrun
Setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
----------------------------------------------------------------------------------------------------
Model configs saved at /lustre/fs01/portfolios/dir/users/jeromek/40b-train-n128-testrun/202410211755/model_configs
Data config saved at /lustre/fs01/portfolios/dir/users/jeromek/40b-train-n128-testrun/202410211755/data_config
SLURM scripts generated in /lustre/fs01/portfolios/dir/users/jeromek/40b-train-n128-testrun/202410211755/scripts
Args saved at /lustre/fs01/portfolios/dir/users/jeromek/40b-train-n128-testrun/202410211755/args.json
----------------------------------------------------------------------------------------------------
```

Submit the SLURM script generated above (`/lustre/fs01/portfolios/dir/users/jeromek/40b-train-n128-testrun/202410211755/scripts` above) or navigate to the output directory (`/lustre/fs01/portfolios/dir/users/jeromek/40b-train-n128-testrun/202410211755` in the above example) to see all generated artifacts.

**Notes**
- You will need to export your `wandb` key when generating the script, i.e. `export WANDB_API_KEY=xxx` (or remove the `--use-wandb` flag from the script)
- The script assumes you have a user directory at `/lustre/fs01/portfolios/dir/users/$USER`
- It does not mount home: `no-container-mount-home` is passed to `pyxis`.

### Configs

#### 40b-train.sh
- Closest to the final config we will be running

#### 40b-train-no-chkpt.sh
- This is for NVIDIA dummy jobs for testing iteration time, etc.
- Difference from `40b-train.sh` is that checkpointing is disabled and that job time is not set to maximum.

#### 40b-test.sh
- This is for testing at smaller scales
    - overrides training iterations
    - overrides micro batch size and checkpoint number of layers to enable running on 4 nodes
    - optional logging data (mem stats, nvidia-resiliency-ext detector -- not tested extensively, etc.)
    - disables checkpointing

#### 40b-hsd.sh
- This demonstrates usage of the script with Heimdall Straggler Detection enabled
- See the additional flags:
    ```
    ...
     --heimdall_log_straggler \ #Enables hsd
     --heimdall_log_interval $HSD_LOG_INTERVAL \ # Frequency of hsd report
     --heimdall_straggler_minmax_count $MINMAX_COUNT \ # Minmax number of straggler ranks
     --heimdall_straggler_port $HSD_PORT \ # Port for toggling hsd on / off
     ...
     ```
- With `hsd` enabled, you should see log lines:
```
2024-10-23 01:29:30] [INFO] | MnRtt/Rnk: 1446.49ms/13 | MxRtt/Rnk: 1570.32ms/30 | MnPwr/Rnk: 617923.00W/0 | MxPwr/Rnk: 669787.00W/20 | MnTmp/Rnk: 50.00C/0 | MxTmp/Rnk: 61.00C/19 | MnUtl/Rnk: 45.00%/20 | MxUtl/Rnk: 100.00%/3 | MnClk/Rnk: 1920.00MHz/14 | MxClk/Rnk: 1965.00MHz/0 | MnDRtt/Rnk: 1.22ms/15 | MxDRtt/Rnk: 130.33ms/23 | MnEtpt/Rnk: 560.82TF/30 | MxEtpt/Rnk: 608.83TF/13
```
- h/t @pallab-zz for the integration
    
#### Note on checkpointing
- To test resuming training from a checkpoint, pass the additional flags to the `python generate_distributed_launcher` script:
```
CMD="python generate_distributed_launcher.py \
    $JOB_NAME \
    --resume \
    --checkpoint_path $CHECKPOINT_PATH \
    --iteration $ITERATION \
    --expandable_segments \
    --use-wandb \
    --launcher $LAUNCHER \
    --job-time $JOBTIME \
    --partition $PARTITION \
    --account $ACCOUNT \
    --container $CONTAINER \
    --num-nodes $NUM_NODES \
    --num-gpus $NUM_GPUS \
    --data-config $DATA_CONFIG \
    --model-config $MODEL_CONFIG \
    --train-script $TRAIN_SCRIPT \
    --wandb-project $BASE_NAME \
    --wandb-run-name $JOB_NAME \
    --overrides $OVERRIDES"
```
where `CHECKPOINT_PATH` is the path to local checkpoint directory and `ITERATION` is the last checkpoint iteration, which you can find by checking the checkpoint directory.  Local checkpoints are saved in subdirectories within the checkpoint directory with the following prefix `global_step{iteration}` where `iteration` is the last checkpointed step.