from pathlib import Path

LAUNCHER_DIR = Path(__file__).resolve().parent
# Root directory of `savanna`
SAVANNA_ROOT = LAUNCHER_DIR.parent

# Convenience constants when using pyxis
EVO_ONLY_CONTAINER = "rilango/evo2" # only for testing
EFS_EVO_CONTAINER = "/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/nvidia_evo2_efa_latest.sqsh"
#"/lustre/fs01/portfolios/dir/project/dir_arc/containers/clara-discovery+savanna+arc-evo2_efa+nv-latest-cascade-1.5.sqsh"
# Location of uploaded data
DEFAULT_ACCOUNT = "dir_arc"
DEFAULT_PARTITION = "pool0"
DEFAULT_DATA_DIR = "/lustre/fs01/portfolios/dir/projects/dir_arc/evo/data" 
DEFAULT_CHECKPOINT_DIR = "/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints"
# $Variables will be defined in generated bash script
DEFAULT_CONTAINER_MOUNTS = "$DATA_DIR:/data,$ROOT_DIR:$ROOT_DIR,$OUTPUT_DIR:$OUTPUT_DIR,$CHECKPOINT_DIR:$CHECKPOINT_DIR"
DEFAULT_CONTAINER_WORKDIR = "$ROOT_DIR"
DEFAULT_OUTPUT_DIR = "/lustre/fs01/portfolios/dir/users"

DEFAULT_DATA_CONFIG = SAVANNA_ROOT / "configs/launcher-test/data_configs/opengenome.yml"
DEFAULT_MODEL_CONFIG = SAVANNA_ROOT / "configs/launcher-test/model_configs/7b_shc_post_refactor-mp-dp.yml"
DEFAULT_WANDB_HOST = "https://api.wandb.ai"

# Helper functions needed for data loading
MAKE_DATA_HELPERS = "make -C $ROOT_DIR/savanna/data"
INSTALL_SAVANNA = "pip install $ROOT_DIR"

# NOTE: need PYTHONPATH to be evaluated within `srun`
UPDATE_PYTHONPATH = "export PYTHONPATH=$ROOT_DIR:\$PYTHONPATH"
NUM_NSYS_PROFILES = 4

RECORD_STREAMS_ENV_VAR="TORCH_NCCL_AVOID_RECORD_STREAMS"
EXPANDABLE_SEGMENTS="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

TRITON_CACHE_ENV_VAR="TRITON_CACHE_DIR"
MAKE_TRITON_CACHE_DIR="export TRITON_CACHE_DIR=/tmp/triton_cache-\$SLURM_JOB_ID-\$SLURM_STEP_ID-\$SLURM_PROCID-\$SLURM_NODEID && mkdir -p \$TRITON_CACHE_DIR; \\"
INSTALL_CP_REQS="LOCALID=\$SLURM_LOCALID && \
if (( LOCALID == 0 )); then \
pip install arrow ring_flash_attn; \
fi;"