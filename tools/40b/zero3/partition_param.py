import argparse
import os
import time
from collections import OrderedDict
from multiprocessing import Pool

import torch
from deepspeed_constants import (
    BUFFER_KEY,
    DP_SIZE_KEY,
    EXTRA_MODEL_STATE_KEYS,
    FP32_FLAT_GROUPS_KEY,
    MP_SIZE_KEY,
    OPTIMIZER_STATE_KEY,
    PARAM_SHAPE_KEY,
    PARTITION_COUNT_KEY,
)
from partition_lib import (
    FP8_SHAPE,
    FP8_SHAPE_T,
    aligned_size,
    create_zero3_model_state_path,
    create_zero3_optim_state_path,
    get_all_model_files,
    get_all_optim_files,
    get_all_shard_files,
    get_model_file_by_mp_dp_rank,
    get_model_files_by_mp_rank,
    get_optim_file_by_mp_dp_rank,
    get_optim_files_by_mp_rank,
    get_shard_by_mp_rank,
    is_mlp_weight,
    load_model_config,
    pad_to_multiple_shape,
    pad_to_multiple_tensor,
    padding_size,
)

DEVICE = "cpu"


def partition_param(param, param_name, partition_count: int, rank: int, device: str = DEVICE, verbose=False):

    unpartitioned_numel = param.numel()
    tensor_size = aligned_size(param, partition_count)
    padding = padding_size(param, partition_count)

    partition_size = tensor_size // partition_count
    if verbose:
        print(
            f"{param_name}: unpartitioned_numel: {unpartitioned_numel}, aligned_size: {tensor_size}, padding: {padding}, partition_size: {partition_size}",
            flush=True,
        )

    partitioned_tensor = torch.zeros(partition_size, dtype=param.dtype, device=device)

    start = partition_size * rank
    end = start + partition_size
    one_dim_param = param.contiguous().view(-1)

    # No padding
    if start < unpartitioned_numel and end <= unpartitioned_numel:
        partitioned_tensor.copy_(one_dim_param.narrow(0, start, partition_size))
    else:
        # Padding
        if start < unpartitioned_numel:
            if verbose:
                print(f"{param_name}: Uneven padding {unpartitioned_numel - start} elements", flush=True)
            elems_to_copy = unpartitioned_numel - start
            partitioned_tensor[:elems_to_copy] = one_dim_param.narrow(0, start, elems_to_copy)

    return partitioned_tensor


def create_param_shape_groups(reference_param_shapes, target_param_shapes, verbose=False, pad_mlp_weights=False):
    param_shape_groups = []

    if verbose:
        print(f"Creating {len(reference_param_shapes)} param shape groups")

    for param_shape_group in reference_param_shapes:
        new_param_group = OrderedDict()
        for param_name in param_shape_group.keys():
            assert param_name in target_param_shapes
            new_param_group[param_name] = target_param_shapes[param_name]
            if verbose:
                print(
                    f"Param {param_name} shape: {param_shape_group[param_name]} -> {target_param_shapes[param_name]}"
                )
        param_shape_groups.append(new_param_group)

    return param_shape_groups


def create_dp_partition(param_shape_groups, mp_shard_state_dict, dp_rank, partition_count, verbose=False, pad_mlp_weights=False):
    """
    Create len(param_shape_groups) flat param groups from a given mp_shard for a given dp_rank

    mp_shard: model state dict for a model parallel shard to be partitioned across dp_ranks
    param_shape_groups: list of param_name -> param_shape for each optimizer param group
    dp_rank: data parallel rank to partition the mp_shard across
    partition_count: number of partitions to create
    """
    fp32_flat_groups = []

    for param_shape_group in param_shape_groups:
        fp32_flat_tensor = torch.empty([], dtype=torch.float32, device=DEVICE)
        partitioned_params = []
        for param_name in param_shape_group.keys():
            param = mp_shard_state_dict[param_name]
            assert param.shape == param_shape_group[param_name]

            # Pad mlp weights to the next multiple of 16 to meet TE fp8 shape requirements
            if is_mlp_weight(param_name) and pad_mlp_weights:
                # Pad output dim of w1 and w2 to the next multiple of 16
                if "w1" in param_name or "w2" in param_name:
                    padded_param = pad_to_multiple_tensor(param, multiples=FP8_SHAPE_T)
                    assert padded_param[param.shape[0]:, :].equal(torch.zeros_like(padded_param[param.shape[0]:, :]))
                    assert padded_param[:, param.shape[1]:].equal(torch.zeros_like(padded_param[:, param.shape[1]:]))
                    assert padded_param.shape[0] % FP8_SHAPE_T[0] == 0, f"Expected shape {padded_param.shape} to be a multiple of {FP8_SHAPE_T} after padding, found {padded_param.shape[0] % FP8_SHAPE_T[0]}"
                    assert padded_param.shape[1] % FP8_SHAPE_T[1] == 0, f"Expected shape {padded_param.shape} to be a multiple of {FP8_SHAPE_T} after padding, found {padded_param.shape[1] % FP8_SHAPE_T[1]}"
                    
                    if param.shape[0] % FP8_SHAPE_T[0] != 0:
                        print(f"{os.getpid()}: Padding dim 0 of {param_name} from {param.shape[0]} to {padded_param.shape[0]}", flush=True)
                    else:
                        print(f"{os.getpid()}: No padding needed for dim 0 of {param_name}: {param.shape[0]} -> {padded_param.shape[0]}", flush=True)
                    if param.shape[1] % FP8_SHAPE_T[1] != 0:
                        print(f"{os.getpid()}: Padding dim 1 of {param_name}: {param.shape[1]} -> {padded_param.shape[1]}", flush=True)
                    else:
                        print(f"{os.getpid()}: No padding needed for dim 1 of {param_name}: {param.shape[1]} -> {padded_param.shape[1]}", flush=True)
                    
                # Pad input dim of w3 to the next multiple of 16
                elif "w3" in param_name:
                    padded_param = pad_to_multiple_tensor(param, multiples=FP8_SHAPE)
                    assert padded_param[param.shape[0]:, :].equal(torch.zeros_like(padded_param[param.shape[0]:, :]))
                    assert padded_param[:, param.shape[1]:].equal(torch.zeros_like(padded_param[:, param.shape[1]:]))
                    assert padded_param.shape[0] % FP8_SHAPE[0] == 0, f"Expected shape {padded_param.shape} to be a multiple of {FP8_SHAPE} after padding, found {padded_param.shape[0] % FP8_SHAPE[0]}"
                    assert padded_param.shape[1] % FP8_SHAPE[1] == 0, f"Expected shape {padded_param.shape} to be a multiple of {FP8_SHAPE} after padding, found {padded_param.shape[1] % FP8_SHAPE[1]}"
                    
                    if param.shape[0] % FP8_SHAPE[0] != 0:
                        print(f"{os.getpid()}: Padding dim 0 of {param_name}: {param.shape[0]} -> {padded_param.shape[0]}", flush=True)
                    else:
                        print(f"{os.getpid()}: No padding needed for dim 0 of {param_name}: {param.shape[0]} -> {padded_param.shape[0]}", flush=True)
                    if param.shape[1] % FP8_SHAPE[1] != 0:
                        print(f"{os.getpid()}: Padding dim 1 of {param_name}: {param.shape[1]} -> {padded_param.shape[1]}", flush=True)
                    else:
                        print(f"{os.getpid()}: No padding needed for dim 1 of {param_name}: {param.shape[1]} -> {padded_param.shape[1]}", flush=True)

                param = padded_param
            
            partitioned_param = partition_param(
                param, param_name=param_name, partition_count=partition_count, rank=dp_rank, verbose=verbose
            )
            partitioned_params.append(partitioned_param)
        # Merge all params for this dp rank's model partition into flat tensor
        fp32_flat_tensor = torch.cat(partitioned_params, dim=0)

        fp32_flat_groups.append(fp32_flat_tensor)

    return fp32_flat_groups


def check_fp32_tensors(reference_fp32_flat_groups, target_fp32_flat_groups, source_dp_size, target_dp_size):
    assert len(reference_fp32_flat_groups) == len(target_fp32_flat_groups)

    for ref, target in zip(reference_fp32_flat_groups, target_fp32_flat_groups):
        ref_numels = ref.numel()
        target_numels = target.numel()
        assert ref_numels == target_numels * (
            source_dp_size // target_dp_size
        ), f"Expected {ref_numels} elements, found {target_numels} * {source_dp_size // target_dp_size} elements"
        # assert torch.allclose(ref, target)

    print("All fp32 tensors check passed!")


def reset_optim_state(optimizer_state_dict, fp32_flat_groups, target_model_config):
    optimizer_state_dict["overflow"] = optimizer_state_dict.get("overflow", False)
    optimizer_state_dict["dynamic_loss_scale"] = optimizer_state_dict.get("dynamic_loss_scale", False)
    optimizer_states = optimizer_state_dict[OPTIMIZER_STATE_KEY]

    # Reset optimizer metadata, reset lr; update betas, eps in case these changed in target_model_config
    for param_group in optimizer_states["param_groups"]:
        param_group["step"] = 0
        param_group["lr"] = target_model_config["optimizer"]["params"]["lr"]
        param_group["betas"] = target_model_config["optimizer"]["params"]["betas"]
        param_group["eps"] = target_model_config["optimizer"]["params"]["eps"]

    # Reset optimizer states, assumes Adam optimizer
    optimizer = optimizer_states["state"]
    assert len(optimizer) == len(fp32_flat_groups)

    optimizer_params = optimizer.values()

    for i, (fp32_flat_group, params) in enumerate(zip(fp32_flat_groups, optimizer_params)):
        params["exp_avg"] = torch.zeros(
            fp32_flat_group.numel(), dtype=params["exp_avg"].dtype, device=params["exp_avg"].device
        )
        params["exp_avg_sq"] = torch.zeros(
            fp32_flat_group.numel(), dtype=params["exp_avg_sq"].dtype, device=params["exp_avg_sq"].device
        )

    optimizer_state_dict[OPTIMIZER_STATE_KEY] = optimizer_states

    return optimizer_state_dict


def update_model_state(model_state, sharded_state_dict, param_shape_groups, target_model_config, target_mp_size, target_dp_size, pad_mlp_weights=False):
    # Create new model state
    # Reset model state
    keys = list(model_state.keys())
    for k in keys:
        if k in EXTRA_MODEL_STATE_KEYS:
            model_state.pop(k)

    model_state["iteration"] = 0
    model_state["global_steps"] = 0
    model_state["global_samples"] = 0
    model_state["skipped_steps"] = 0
    model_state["data_sampler"] = None
    model_state["random_ltd"] = None
    
    # Update param shapes
    if pad_mlp_weights:
        for param_shape_group in param_shape_groups:
            for param_name in param_shape_group.keys():
                if is_mlp_weight(param_name):
                    if "w1" in param_name or "w2" in param_name:
                        new_shape = pad_to_multiple_shape(param_shape_group[param_name], multiples=FP8_SHAPE_T)
                        if param_shape_group[param_name][0] % FP8_SHAPE_T[0] != 0:
                            print(f"{os.getpid()}: Padding dim 0 of {param_name}: {param_shape_group[param_name][0]} -> {new_shape[0]}", flush=True)
                        else:
                            print(f"{os.getpid()}: No padding needed for dim 0 of {param_name}: {param_shape_group[param_name][0]} -> {new_shape[0]}", flush=True)
                        if param_shape_group[param_name][1] % FP8_SHAPE_T[1] != 0:
                            print(f"{os.getpid()}: Padding dim 1 of {param_name}: {param_shape_group[param_name][1]} -> {new_shape[1]}", flush=True)
                        else:
                            print(f"{os.getpid()}: No padding needed for dim 1 of {param_name}: {param_shape_group[param_name][1]} -> {new_shape[1]}", flush=True)
                        assert new_shape[0] % FP8_SHAPE_T[0] == 0, f"Expected dim 0 of {param_name} padded shape {new_shape} to be a multiple of {FP8_SHAPE_T[0]} after padding, found {new_shape[0] % FP8_SHAPE_T[0]}"
                        assert new_shape[1] % FP8_SHAPE_T[1] == 0, f"Expected dim 1 of {param_name} padded shape {new_shape} to be a multiple of {FP8_SHAPE_T[1]} after padding, found {new_shape[1] % FP8_SHAPE_T[1]}"

                        param_shape_group[param_name] = new_shape
                    elif "w3" in param_name:
                        new_shape = pad_to_multiple_shape(param_shape_group[param_name], multiples=FP8_SHAPE)
                        if param_shape_group[param_name][0] % FP8_SHAPE[0] != 0:
                            print(f"{os.getpid()}: Padding dim 0 of {param_name}: {param_shape_group[param_name][0]} -> {new_shape[0]}", flush=True)
                        else:
                            print(f"{os.getpid()}: No padding needed for dim 0 of {param_name}: {param_shape_group[param_name][0]} -> {new_shape[0]}", flush=True)
                        if param_shape_group[param_name][1] % FP8_SHAPE[1] != 0:
                            print(f"{os.getpid()}: Padding dim 1 of {param_name}: {param_shape_group[param_name][1]} -> {new_shape[1]}", flush=True)
                        else:
                            print(f"{os.getpid()}: No padding needed for dim 1 of {param_name}: {param_shape_group[param_name][1]} -> {new_shape[1]}", flush=True)
                        assert new_shape[0] % FP8_SHAPE[0] == 0, f"Expected dim 0 of {param_name} padded shape {new_shape} to be a multiple of {FP8_SHAPE[0]} after padding, found {new_shape[0] % FP8_SHAPE[0]}"
                        assert new_shape[1] % FP8_SHAPE[1] == 0, f"Expected dim 1 of {param_name} padded shape {new_shape} to be a multiple of {FP8_SHAPE[1]} after padding, found {new_shape[1] % FP8_SHAPE[1]}"

                        param_shape_group[param_name] = new_shape
    model_state[PARAM_SHAPE_KEY] = param_shape_groups
    # Update lr_scheduler
    updated_lr_scheduler = {}
    updated_lr_scheduler["min_lr"] = target_model_config["min_lr"]
    updated_lr_scheduler["start_lr"] = target_model_config["optimizer"]["params"]["lr"]
    updated_lr_scheduler["num_iters"] = 0
    updated_lr_scheduler["end_iter"] = target_model_config["train_iters"]
    updated_lr_scheduler["decay_style"] = target_model_config["lr_decay_style"]
    updated_lr_scheduler["warmup_iter"] = int(
        target_model_config["warmup"] * target_model_config["train_iters"]
    )
    model_state["lr_scheduler"] = updated_lr_scheduler
    # Copy buffer and extra states
    # Save model state
    keys = model_state['module'].keys()
    buffer_names = model_state[BUFFER_KEY]
    for k in keys:
        if k in buffer_names or 'extra' in k:
            print(f"Copying {k} from mp_shard_state_dict")
            model_state['module'][k] = sharded_state_dict[k]
        else:
            print(f"Skipping {k}:{model_state['module'][k]} from mp_shard_state_dict")
    
    #Update dp_world_size and mp_world_size
    model_state[DP_SIZE_KEY] = target_dp_size
    model_state[MP_SIZE_KEY] = target_mp_size
    return model_state


def create_mp_dp_zero3_states(
    param_shape_groups,
    mp_shard_state_dict,
    optim_state,
    model_state,
    target_model_config,
    dp_rank,
    mp_rank,
    source_dp_size,
    target_dp_size,
    target_mp_size,
    pad_mlp_weights=True,
    verbose=False,
):

    partition_count = target_dp_size
    # First update optim state
    fp32_flat_groups = create_dp_partition(
        param_shape_groups, mp_shard_state_dict, dp_rank, partition_count=partition_count, verbose=True, pad_mlp_weights=pad_mlp_weights
    )
    # reference_fp32_flat_groups = optim_state[OPTIMIZER_STATE_KEY][FP32_FLAT_GROUPS_KEY]
    # check_fp32_tensors(reference_fp32_flat_groups, fp32_flat_groups, source_dp_size, target_dp_size)

    # Create new optim state
    # Remove old ds_config
    optim_state.pop("ds_config")
    # Replace fp32 flat groups
    optim_state[OPTIMIZER_STATE_KEY][FP32_FLAT_GROUPS_KEY] = fp32_flat_groups

    # Reset optimizer states
    optimizer_state_dict = optim_state[OPTIMIZER_STATE_KEY]
    optimizer_state_dict[PARTITION_COUNT_KEY] = partition_count
    optimizer_state_dict = reset_optim_state(optimizer_state_dict, fp32_flat_groups, target_model_config)
    optim_state[OPTIMIZER_STATE_KEY] = optimizer_state_dict

    # Update model state
    model_state = update_model_state(model_state, sharded_state_dict=mp_shard_state_dict, param_shape_groups=param_shape_groups, target_model_config=target_model_config, target_mp_size=target_mp_size, target_dp_size=target_dp_size, pad_mlp_weights=pad_mlp_weights)

    return optim_state, model_state

def create_zero3_states_by_mp_rank(
    reference_checkpoint_dir,
    sharded_checkpoint_dir,
    target_model_config_path,
    target_dp_size,
    source_dp_size,
    target_mp_size,
    mp_rank,
    output_dir,
    pad_mlp_weights=False,
):
    global_start_time = time.time()

    #NOTE: we only need to copy mp-dp independent metadata from zero3 model and optim states, hence we use ref_mp_rank and ref_dp_rank
    # The only mp-dependent data in model state are buffers, which we overwrite from sharded state dict
    # For optim state, we overwrite fp32_flat_groups and reset optimizer states
    ref_mp_rank = ref_dp_rank = 0

    model_state_paths = get_model_files_by_mp_rank(reference_checkpoint_dir, ref_mp_rank)
    optim_state_paths = get_optim_files_by_mp_rank(reference_checkpoint_dir, ref_mp_rank)
    assert len(model_state_paths) == len(optim_state_paths)
    assert len(model_state_paths) == source_dp_size

    # NOTE: This NEEDS to be mp_rank, not ref_mp_rank
    sharded_state_path = get_shard_by_mp_rank(sharded_checkpoint_dir, mp_rank)
    mp_shard = torch.load(sharded_state_path, map_location=DEVICE)
    target_model_config = load_model_config(target_model_config_path)

    # Create target_dp_size states for this mp_rank
    for dp_rank in range(target_dp_size):
        print(f"{os.getpid()}: Creating zero3 states for dp_rank {dp_rank} mp_rank {mp_rank}", flush=True)

        model_state_path = get_model_file_by_mp_dp_rank(model_state_paths, ref_mp_rank, ref_dp_rank)
        optim_state_path = get_optim_file_by_mp_dp_rank(optim_state_paths, ref_mp_rank, ref_dp_rank)
        model_state = torch.load(model_state_path, map_location=DEVICE)
        optim_state = torch.load(optim_state_path, map_location=DEVICE)

        reference_param_shapes = model_state["param_shapes"]
        target_param_shapes = mp_shard["param_shapes"]

        # Split merged param shapes in param groups per zero-3 metadata
        param_shape_groups = create_param_shape_groups(
            reference_param_shapes, target_param_shapes, verbose=False
        )
        state_time = time.time()
        optim_state, model_state = create_mp_dp_zero3_states(
            param_shape_groups=param_shape_groups,
            mp_shard_state_dict=mp_shard["module"],
            optim_state=optim_state,
            model_state=model_state,
            target_model_config=target_model_config,
            dp_rank=dp_rank,
            mp_rank=mp_rank,
            source_dp_size=source_dp_size,
            target_dp_size=target_dp_size,
            target_mp_size=target_mp_size,
            verbose=True,
            pad_mlp_weights=pad_mlp_weights,
        )
        print(
            f"{os.getpid()}: Time taken to create states for mp_rank {mp_rank} dp_rank {dp_rank} : {time.time() - state_time} seconds",
            flush=True,
        )

        # Save optim state
        start_time = time.time()
        #Output paths NEED to be mp_rank and dp_rank to create the correct directory structure
        optim_state_output_path = create_zero3_optim_state_path(dp_rank, mp_rank)
        optim_state_output_path = os.path.join(output_dir, optim_state_output_path)
        torch.save(optim_state, optim_state_output_path)

        # Save model state
        model_state_output_path = create_zero3_model_state_path(dp_rank, mp_rank)
        model_state_output_path = os.path.join(output_dir, model_state_output_path)
        torch.save(model_state, model_state_output_path)
        end_time = time.time()
        print(
            f"{os.getpid()}: Saved optim state and model state to {optim_state_output_path} and {model_state_output_path} in {end_time - start_time} seconds",
            flush=True,
        )
        end_time = time.time()
    
    print(f"{os.getpid()}: Total time taken: {end_time - global_start_time} seconds", flush=True)


def main(args):
    source_dp_size = args.source_dp_size
    source_mp_size = args.source_mp_size
    source_world_size = source_dp_size * source_mp_size
    target_mp_size = args.target_mp_size
    target_dp_size = args.target_dp_size
    target_world_size = target_dp_size * target_mp_size
    # context_len = args.context_len
    # tag = args.tag
    # reference_checkpoint_dir = f"/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/{context_len}/zero3/{tag}"
    # sharded_checkpoint_dir = f"/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/{context_len}/zero1/{tag}"
    # output_dir = f"repartitioned_checkpoints/{context_len}/MP{target_mp_size}DP{target_dp_size}/zero3/global_step0"
    #os.makedirs(args.output_dir, exist_ok=True)

    z3_checkpoint_dir = args.source_zero3_dir
    zero1_checkpoint_dir = args.sharded_zero1_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    model_state_paths = get_all_model_files(z3_checkpoint_dir)
    optim_state_paths = get_all_optim_files(z3_checkpoint_dir)
    assert len(model_state_paths) == len(
        optim_state_paths
    ), f"Expected {len(model_state_paths)} model states and {len(optim_state_paths)} optim states, found {len(model_state_paths)} model states and {len(optim_state_paths)} optim states"
    assert (
        len(model_state_paths) == source_world_size
    ), f"Expected {source_world_size} model states, found {len(model_state_paths)} model states"
    sharded_state_paths = get_all_shard_files(zero1_checkpoint_dir)
    assert (
        len(sharded_state_paths) == target_mp_size
    ), f"Expected {target_mp_size} sharded states, found {len(sharded_state_paths)} sharded states"

    target_model_config_path = args.target_model_config_path
    target_model_config = load_model_config(target_model_config_path)

    # Constants: partition_count, model_config, param_shape_groups
    # Per Dp Per MP Rank: optim_state, model_state, dp_rank, mp_rank
    print(
        f"{os.getpid()}: Resharding from {source_mp_size} mp_size to {target_mp_size} mp_size, {source_dp_size} dp_size to {target_dp_size} dp_size starting from mp_rank {args.start_mp_rank} to mp_rank {args.end_mp_rank}",
        flush=True,
    )
    ranks_to_process = range(args.start_mp_rank, args.end_mp_rank + 1)
    print(f"Ranks to process: {ranks_to_process}", flush=True)
    start_time = time.time()
    if args.num_workers == 1:
        for mp_rank in ranks_to_process:
            print(f"{os.getpid()}: Creating zero3 states for mp_rank {mp_rank}", flush=True)
            create_zero3_states_by_mp_rank(
                z3_checkpoint_dir,
                zero1_checkpoint_dir,
                target_model_config_path,
                target_dp_size=target_dp_size,
                source_dp_size=source_dp_size,
                target_mp_size=target_mp_size,
                mp_rank=mp_rank,
                output_dir=output_dir,
                pad_mlp_weights=args.pad_mlp_weights,
            )
    else:
        with Pool(processes=args.num_workers) as pool:
            pool.starmap(create_zero3_states_by_mp_rank, [(z3_checkpoint_dir, zero1_checkpoint_dir, target_model_config_path, target_dp_size, source_dp_size, target_mp_size, mp_rank, output_dir, args.pad_mlp_weights) for mp_rank in ranks_to_process])

    end_time = time.time()
    print(f"{os.getpid()}: Total time taken to reshard: {end_time - start_time} seconds", flush=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dp_size", type=int, default=256)
    parser.add_argument("--source_mp_size", type=int, default=8)
    parser.add_argument("--target_mp_size", type=int, default=8)
    parser.add_argument("--target_dp_size", type=int, default=4)
    parser.add_argument("--source_zero3_dir", type=str, required=True)
    parser.add_argument("--target_model_config_path", type=str, required=True)
    parser.add_argument("--sharded_zero1_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pad_mlp_weights", action="store_true", help="Pad output dim of w1 and w2 and input dim of w3 to the next multiple of 16 to meet TE fp8 shape requirements")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--start_mp_rank", type=int, default=None)
    parser.add_argument("--end_mp_rank", type=int, default=None)
    args = parser.parse_args()

    #assert args.num_workers == 1, "Only support num_workers=1 for now"
    if args.start_mp_rank is None:
        args.start_mp_rank = 0
    if args.end_mp_rank is None:
        args.end_mp_rank = args.target_mp_size
    print("Args:")
    for k, v in args.__dict__.items():
        print(f"{k}: {v}")
    main(args)
