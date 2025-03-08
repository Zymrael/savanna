import torch

optim_state_path = "/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-checkpoint-tests/4layer_zero3/global_step1000/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"

state_dict = torch.load(optim_state_path)
#  state_dict['optimizer_state_dict'].keys()
#  dict_keys(['zero_stage', 'loss_scaler', 'dynamic_loss_scale', 'overflow', 'partition_count', 'optimizer_state_dict', 'fp32_flat_groups'])
#In [2]: state_dict.keys()
#Out[2]: dict_keys(['optimizer_state_dict', 'ds_config', 'ds_version'])
#state_dict['ds_config]
# #{'train_batch_size': 2,
#  'train_micro_batch_size_per_gpu': 1,
#  'optimizer': {'type': 'Adam',
#   'params': {'lr': 0.0002, 'betas': [0.9, 0.95], 'eps': 1e-08}},
#  'bf16': {'enabled': True},
#  'zero_optimization': {'stage': 3,
#   'prefetch_bucket_size': 500000000,
#   'max_live_parameters': 1000000000,
#   'allgather_partitions': True,
#   'allgather_bucket_size': 500000000,
#   'overlap_comm': True,
#   'reduce_scatter': True,
#   'reduce_bucket_size': 500000000,
#   'contiguous_gradients': True,
#   'cpu_offload': False,
#   'param_persistence_threshold': 0},
#  'steps_per_print': 5}