import os
import socket

# MASTER_ADDR, MASTER_PORT,LOCAL_WORLD_SIZE are set manually in the generated SLURM script directly or derived from SLURM vars
# RANK, LOCAL_RANK, WORLD_SIZE are set by the launcher, either `torchrun` or `deepspeed.launcher.launch`
DIST_VARS = ["MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK", "RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"]
SLURM_VARS = ["SLURM_NTASKS", "SLURM_GPUS_ON_NODE", "SLURM_JOB_NUM_NODES", "SLURM_NTASKS_PER_NODE"]
def check_distributed_vars(assert_all=True):
    env_vars = {}
    for var in DIST_VARS:
        if var not in os.environ:
            print(f"ERROR: {var} not set")
            env_vars[var] = None
        else:
            if var == "MASTER_ADDR":
                host = socket.gethostname()
                print(f"HOST: {host}")
                env_vars["HOST"] = host
            print(f"{var}: {os.environ[var]}")
            env_vars[var] = os.environ[var]
    
    if assert_all:
        assert all(var in os.environ for var in DIST_VARS)
    if env_vars["RANK"] == "0":
        assert env_vars["HOST"] == env_vars["MASTER_ADDR"], f"RANK 0: HOST {env_vars['HOST']} != MASTER_ADDR {env_vars['MASTER_ADDR']}"
    return env_vars

def check_slurm_env():
    assert all(var in os.environ for var in SLURM_VARS)
    return {k: os.environ[k] for k in SLURM_VARS}