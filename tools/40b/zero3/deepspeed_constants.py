OPTIMIZER_STATE_KEY = "optimizer_state_dict"
FP32_FLAT_GROUPS_KEY = "fp32_flat_groups"
OPTIMIZER_FP32_FLAT_GROUPS_KEY = "state"
PARTITION_COUNT_KEY = "partition_count"
MODEL_KEY = "module"
MODEL_OPTIMIZER_KEY = "optimizer"
FROZEN_PARAM_SHAPE_KEY = "frozen_param_shapes"
SHARED_PARAMS_KEY = "shared_params"
PARAM_SHAPE_KEY = "param_shapes"
BUFFER_KEY = "buffer_names"
DP_SIZE_KEY = 'dp_world_size'
MP_SIZE_KEY = 'mp_world_size'
MODEL_STATE_KEYS = [
    MODEL_KEY,
    MODEL_OPTIMIZER_KEY,
    PARAM_SHAPE_KEY,
    FROZEN_PARAM_SHAPE_KEY,
    SHARED_PARAMS_KEY,
]
EXTRA_MODEL_STATE_KEYS = [
    "ds_config",
    "args",
    # "data_sampler",
    # "random_ltd",
    # "skipped_steps",
    # "global_steps",
    # "global_samples",
    "iteration",
    "data_loading",
    "random_rng_state",
    "np_rng_state",
    "torch_rng_state",
    "cuda_rng_state",
    "rng_tracker_states",
]
