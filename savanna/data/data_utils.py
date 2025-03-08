import math
import torch
import numpy as np
from typing import List, Tuple
from itertools import zip_longest
from functools import partial

from savanna import mpu, print_rank_0
from savanna.data.indexed_dataset import make_dataset as make_indexed_dataset
from savanna.data.blendable_dataset import BlendableDataset
from savanna.data.sequence_dataset import SequenceDataset
from savanna.data.samplers import DistributedBatchSampler


def make_data_loader(dataset, global_config):
    """Build dataloader given an input dataset."""
    if dataset is None:
        return None
    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = global_config.batch_size * world_size
    num_workers = global_config.num_workers

    # Use a simple sampler with distributed batch sampler.
    sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(
        sampler=sampler,
        batch_size=global_batch_size,
        drop_last=True,
        rank=rank,
        world_size=world_size,
    )
    # Torch dataloader.
    return torch.utils.data.DataLoader(
        dataset, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=True
    )


def build_the_dataset(
    data_prefix,
    name,
    data_impl,
    num_samples,
    seq_length,
    seed,
    skip_warmup,
    build_index_mappings=True,
    enforce_sample_length=False,
    sample_dtype=np.int64,
):
    """Build train/valid/test datasets."""

    indexed_dataset = make_indexed_dataset(data_prefix, data_impl, skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    print_rank_0("    {}:".format(name))
    print_rank_0("     no. of documents:{}".format(total_num_of_documents))
    dataset = None
    documents = np.arange(start=0, stop=total_num_of_documents, step=1, dtype=np.int64)
    dataset = SequenceDataset(
        name,
        data_prefix,
        documents,
        indexed_dataset,
        num_samples,
        seq_length,
        seed,
        build_index_mappings=build_index_mappings,
        enforce_sample_length=enforce_sample_length,
        sample_dtype=sample_dtype,
    )
    return dataset


def build_train_valid_test_datasets(
    data_prefix,
    use_shared_fs,
    data_impl,
    splits_string,
    train_valid_test_num_samples,
    seq_length,
    seed,
    skip_warmup,
    enforce_sample_length=False,
):
    """Build train, valid, and test datasets."""

    # Indexed dataset.
    indexed_dataset = make_indexed_dataset(data_prefix, data_impl, skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(" > dataset split:")

    def print_split_stats(name, index):
        print_rank_0("    {}:".format(name))
        print_rank_0(
            "     document indices in [{}, {}) total of {} "
            "documents".format(splits[index], splits[index + 1], splits[index + 1] - splits[index])
        )

    print_split_stats("train", 0)
    print_split_stats("validation", 1)
    print_split_stats("test", 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index], stop=splits[index + 1], step=1, dtype=np.int64)

            dataset = SequenceDataset(
                name,
                data_prefix,
                documents,
                indexed_dataset,
                train_valid_test_num_samples[index],
                seq_length,
                seed,
                use_shared_fs=use_shared_fs,
                enforce_sample_length=enforce_sample_length,
            )
        return dataset

    train_dataset = build_dataset(0, "train")
    valid_dataset = build_dataset(1, "valid")
    test_dataset = build_dataset(2, "test")

    return train_dataset, valid_dataset, test_dataset


def get_train_valid_test_split_(splits_string, size):
    """Get dataset splits from comma or '/' separated string list."""

    splits = []
    if splits_string.find(",") != -1:
        splits = [float(s) for s in splits_string.split(",")]
    elif splits_string.find("/") != -1:
        splits = [float(s) for s in splits_string.split("/")]
    else:
        splits = [float(splits_string)]
    while len(splits) < 3:
        splits.append(0.0)
    splits = splits[:3]
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] + int(round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    assert len(splits_index) == 4
    assert splits_index[-1] == size
    return splits_index


def get_normalized_weights_and_num_samples(
    weights: List[float], num_samples: int
) -> Tuple[List[float], List[int]]:
    # Normalize weights
    weight_sum = sum(weights)
    assert weight_sum > 0.0
    weights = [weight / weight_sum for weight in weights]
    # Add 0.5% (the 1.005 factor) so in case the blending dataset does
    # not uniformly distribute the number of samples, we still have
    # samples left to feed to the network.
    weighted_num_samples = []
    for weight in weights:
        weighted_num_samples.append(int(math.ceil(num_samples * weight * 1.005)))
    return weights, weighted_num_samples


def build_weighted_datasets(
    global_config,
    train_num_samples,
    valid_num_samples,
    test_num_samples,
    train_weights,
    valid_weights,
    test_weights,
    build_index_mappings=True,
):
    # build individual datasets
    train_datasets, valid_datasets, test_datasets = [], [], []
    for i, (train_path, valid_path, test_path) in enumerate(
        zip_longest(
            global_config.train_data_paths,
            global_config.valid_data_paths,
            global_config.test_data_paths,
        )
    ):
        if global_config.alignment_method == "dpo":
            assert (
                global_config.dpo_data_seq_length is not None
            ), "The dpo_data_seq_length parameter must be set when using DPO"
            seq_length = global_config.dpo_data_seq_length
            sample_dtype = np.float32
            global_config.enforce_sample_length = True
        else:
            seq_length = global_config.seq_length
            sample_dtype = np.int64

        if train_path:
            train_datasets.append(
                build_the_dataset(
                    data_prefix=train_path,
                    name=f"train_{i}",
                    data_impl=global_config.data_impl,
                    num_samples=train_num_samples[i],
                    seq_length=seq_length,
                    seed=global_config.seed,
                    skip_warmup=(not global_config.mmap_warmup),
                    build_index_mappings=build_index_mappings,
                    enforce_sample_length=global_config.enforce_sample_length,
                    sample_dtype=sample_dtype,
                )
            )

        if valid_path:
            valid_datasets.append(
                build_the_dataset(
                    data_prefix=valid_path,
                    name=f"valid_{i}",
                    data_impl=global_config.data_impl,
                    num_samples=valid_num_samples[i],
                    seq_length=seq_length,
                    seed=global_config.seed,
                    skip_warmup=(not global_config.mmap_warmup),
                    build_index_mappings=build_index_mappings,
                    enforce_sample_length=global_config.enforce_sample_length,
                    sample_dtype=sample_dtype,
                )
            )

        if test_path:
            test_datasets.append(
                build_the_dataset(
                    data_prefix=test_path,
                    name=f"test_{i}",
                    data_impl=global_config.data_impl,
                    num_samples=test_num_samples[i],
                    seq_length=seq_length,
                    seed=global_config.seed,
                    skip_warmup=(not global_config.mmap_warmup),
                    build_index_mappings=build_index_mappings,
                    enforce_sample_length=global_config.enforce_sample_length,
                    sample_dtype=sample_dtype,
                )
            )
    return train_datasets, valid_datasets, test_datasets


def weights_by_num_docs(l: list, alpha=0.3):
    """
    Builds weights from a multinomial distribution over groups of data according to the number of
    samples in each group.

    We sample from a group according to the probability p(L) ∝ |L| ** α,
    where p(L) is the probability of sampling from a given group,
          |L| is the number of examples in that datapoint,
          and α is a coefficient that acts to upsample data from underrepresented groups

    Hence α (`alpha`) allows us to control how much to 'boost' the probability of training on low-resource groups.

    See https://arxiv.org/abs/1911.02116 for more details
    """
    if len(l) == 1:
        return [1.0]

    total_n_docs = sum(l)
    unbiased_sample_probs = [i / total_n_docs for i in l]

    probs = [i**alpha for i in unbiased_sample_probs]

    # normalize
    total = sum(probs)
    probs = [i / total for i in probs]

    # weights should be the inverse of the number of samples
    unbiased_sample_probs_inverse = [1 - p for p in unbiased_sample_probs]
    weights = [p * p2 for p, p2 in zip(probs, unbiased_sample_probs_inverse)]

    # normalize
    total = sum(weights)
    weights = [i / total for i in weights]

    return weights


def build_train_valid_test_data_iterators(global_config):
    """XXX"""

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0("> building train, validation, and test datasets ...")

    # Ensure only the first/last pipeline stages have data loaders
    if global_config.is_pipe_parallel:
        is_first_stage = mpu.get_pipe_parallel_rank() == 0
        is_last_stage = mpu.get_pipe_parallel_rank() == mpu.get_pipe_parallel_world_size() - 1
        pipe_load = is_first_stage or is_last_stage
    else:
        pipe_load = True

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0 and pipe_load:
        # Number of train/valid/test samples.
        train_iters = global_config.train_iters
        eval_iters = (train_iters // global_config.eval_interval + 1) * global_config.eval_iters
        test_iters = global_config.eval_iters
        train_val_test_num_samples = [
            train_iters * global_config.train_batch_size,
            eval_iters * global_config.train_batch_size,
            test_iters * global_config.train_batch_size,
        ]
        if global_config.use_checkpoint_num_samples:
            # This overrides the actual number of samples defined above with the number of
            # samples saved in a loaded checkpoint. This is used to prevent re-indexing of
            # the data loading index, which is (partially) defined by the number of samples
            # along with the sequence length and the random seed.
            if global_config.train_val_test_num_samples is None:
                print_rank_0(
                    "> WARNING: `use_checkpoint_num_samples` is set to True but these values "
                    "were not found in the checkpoint, will recompute these values."
                )
            else:
                if train_val_test_num_samples[0] > global_config.train_val_test_num_samples[0] or \
                   train_val_test_num_samples[1] > global_config.train_val_test_num_samples[1] or \
                   train_val_test_num_samples[2] > global_config.train_val_test_num_samples[2]:
                    print_rank_0(
                        "> WARNING (!!!): You are attempting to use an existing data index that "
                        "has fewer samples than required by the current config. This will lead to "
                        "a StopIteration error once the existing data index has been consumed.\n"
                        "> Consider re-indexing by setting `use_checkpoint_num_samples` to False."
                    )
                if train_val_test_num_samples != global_config.train_val_test_num_samples:
                    print_rank_0(
                        "> Overriding the number of (train, val, test) samples specified by "
                        f"the current config: {train_val_test_num_samples} with the values "
                        f"stored in the checkpoint: {global_config.train_val_test_num_samples}."
                    )
                    train_val_test_num_samples = global_config.train_val_test_num_samples
        global_config.train_val_test_num_samples = train_val_test_num_samples

        if global_config.train_data_paths:
            # when individual train / valid / test data paths are provided
            # normalize weight values and get num samples for each dataset
            train_weights, train_num_samples = get_normalized_weights_and_num_samples(
                global_config.train_data_weights, train_val_test_num_samples[0]
            )
            valid_weights, valid_num_samples = get_normalized_weights_and_num_samples(
                global_config.valid_data_weights, train_val_test_num_samples[1]
            )
            test_weights, test_num_samples = get_normalized_weights_and_num_samples(
                global_config.test_data_weights, train_val_test_num_samples[2]
            )

            # build individual datasets
            train_datasets, valid_datasets, test_datasets = build_weighted_datasets(
                global_config,
                train_num_samples,
                valid_num_samples,
                test_num_samples,
                train_weights,
                valid_weights,
                test_weights,
                build_index_mappings=not global_config.weight_by_num_documents,
            )

            if global_config.weight_by_num_documents:
                # gets the number of documents in each datapath
                get_num_docs_list = lambda datasets: [
                    dataset.indexed_dataset.sizes.shape[0] for dataset in datasets
                ]
                train_num_docs, valid_num_docs, test_num_docs = (
                    get_num_docs_list(train_datasets),
                    get_num_docs_list(valid_datasets),
                    get_num_docs_list(test_datasets),
                )

                # builds weights according to alpha + the number of docs
                fn = partial(weights_by_num_docs, alpha=global_config.weighted_sampler_alpha)
                train_weights, valid_weights, test_weights = (
                    fn(train_num_docs),
                    fn(valid_num_docs),
                    fn(test_num_docs),
                )
                (
                    train_weights,
                    train_num_samples,
                ) = get_normalized_weights_and_num_samples(train_weights, train_val_test_num_samples[0])
                (
                    valid_weights,
                    valid_num_samples,
                ) = get_normalized_weights_and_num_samples(valid_weights, train_val_test_num_samples[1])
                test_weights, test_num_samples = get_normalized_weights_and_num_samples(
                    test_weights, train_val_test_num_samples[2]
                )

                # rebuild datasets weighted according to new weights
                train_datasets, valid_datasets, test_datasets = build_weighted_datasets(
                    global_config,
                    train_num_samples,
                    valid_num_samples,
                    test_num_samples,
                    train_weights,
                    valid_weights,
                    test_weights,
                )

            if train_datasets:
                train_ds = BlendableDataset(train_datasets, train_weights)
            if valid_datasets:
                valid_ds = BlendableDataset(valid_datasets, valid_weights)
            if test_datasets:
                test_ds = BlendableDataset(test_datasets, test_weights)
        else:
            # when just data_path is provided
            # split dataset into train, valid and test from data_path
            train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
                data_prefix=global_config.data_path,
                use_shared_fs=global_config.use_shared_fs,
                data_impl=global_config.data_impl,
                splits_string=global_config.split,
                train_valid_test_num_samples=train_val_test_num_samples,
                seq_length=global_config.seq_length,
                seed=global_config.seed,
                skip_warmup=(not global_config.mmap_warmup),
                enforce_sample_length=global_config.enforce_sample_length,
            )

        # Build dataloders.
        train_dataloader = make_data_loader(train_ds, global_config=global_config)
        valid_dataloader = make_data_loader(valid_ds, global_config=global_config)
        test_dataloader = make_data_loader(test_ds, global_config=global_config)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and global_config.train_iters > 0
        do_valid = valid_dataloader is not None and global_config.eval_iters > 0
        do_test = test_dataloader is not None and global_config.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor([int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    if global_config.is_pipe_parallel:
        # Only first/last pipeline stages have data loaders, so pipeline parallelism should
        # broadcast globally instead of just the model parallel group.
        torch.distributed.broadcast(flags, src=0)
    else:
        torch.distributed.broadcast(
            flags,
            mpu.get_model_parallel_src_rank(),
            group=mpu.get_model_parallel_group(),
        )
    global_config.do_train = flags[0].item()
    global_config.do_valid = flags[1].item()
    global_config.do_test = flags[2].item()

    # Shift the start iterations.
    if train_dataloader is not None:
        tokens_per_iteration = global_config.train_batch_size * global_config.seq_length
        if global_config.train_data_token_index is None:
            # Note that this will be *incorrect* if `global_config.train_data_token_index` has
            # not been previously saved and if the number of iterations, grad accum, batch size,
            # world size, or sequence length have been changed.
            global_config.train_data_token_index = (
                global_config.iteration *
                global_config.gradient_accumulation_steps *
                tokens_per_iteration
            )
        if global_config.train_data_token_index % tokens_per_iteration != 0:
            print_rank_0(
                "WARNING: The cursor into the training data token index cannot be perfectly "
                "achieved by the current configuration, will skip {} tokens."
                .format(
                    tokens_per_iteration -
                    (global_config.train_data_token_index % tokens_per_iteration)
                )
            )
        shifted_start_iter = int(math.ceil(global_config.train_data_token_index / tokens_per_iteration))
        train_dataloader.batch_sampler.start_iter = shifted_start_iter % len(train_dataloader)
        global_config.train_data_token_index = shifted_start_iter * tokens_per_iteration
        print_rank_0(
            "setting training data start iteration to {}"
            .format(train_dataloader.batch_sampler.start_iter)
        )

    if valid_dataloader is not None:
        start_iter_val = (
            (global_config.iteration * global_config.gradient_accumulation_steps)
            // global_config.eval_interval
        ) * global_config.eval_iters
        valid_dataloader.batch_sampler.start_iter = start_iter_val % len(valid_dataloader)
        print_rank_0(
            "setting validation data start iteration to {}".format(valid_dataloader.batch_sampler.start_iter)
        )

    # Build iterators.
    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader)
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = iter(valid_dataloader)
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader)
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator


def build_datasets_from_global_config(global_config):
    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0("> building train, validation, and test datasets ...")

    # Ensure only the first/last pipeline stages have data loaders
    if global_config.is_pipe_parallel:
        is_first_stage = mpu.get_pipe_parallel_rank() == 0
        is_last_stage = mpu.get_pipe_parallel_rank() == mpu.get_pipe_parallel_world_size() - 1
        pipe_load = is_first_stage or is_last_stage
    else:
        pipe_load = True

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0 and pipe_load:
        # Number of train/valid/test samples.
        train_iters = global_config.train_iters
        eval_iters = (train_iters // global_config.eval_interval + 1) * global_config.eval_iters
        test_iters = global_config.eval_iters
        train_val_test_num_samples = [
            train_iters * global_config.train_batch_size,
            eval_iters * global_config.train_batch_size,
            test_iters * global_config.train_batch_size,
        ]

        if global_config.train_data_paths:
            # when individual train / valid / test data paths are provided
            # normalize weight values and get num samples for each dataset
            train_weights, train_num_samples = get_normalized_weights_and_num_samples(
                global_config.train_data_weights, train_val_test_num_samples[0]
            )
            valid_weights, valid_num_samples = get_normalized_weights_and_num_samples(
                global_config.valid_data_weights, train_val_test_num_samples[1]
            )
            test_weights, test_num_samples = get_normalized_weights_and_num_samples(
                global_config.test_data_weights, train_val_test_num_samples[2]
            )

            # build individual datasets
            train_datasets, valid_datasets, test_datasets = build_weighted_datasets(
                global_config,
                train_num_samples,
                valid_num_samples,
                test_num_samples,
                train_weights,
                valid_weights,
                test_weights,
                build_index_mappings=not global_config.weight_by_num_documents,
            )

            if global_config.weight_by_num_documents:
                # gets the number of documents in each datapath
                get_num_docs_list = lambda datasets: [
                    dataset.indexed_dataset.sizes.shape[0] for dataset in datasets
                ]
                train_num_docs, valid_num_docs, test_num_docs = (
                    get_num_docs_list(train_datasets),
                    get_num_docs_list(valid_datasets),
                    get_num_docs_list(test_datasets),
                )

                # builds weights according to alpha + the number of docs
                fn = partial(weights_by_num_docs, alpha=global_config.weighted_sampler_alpha)
                train_weights, valid_weights, test_weights = (
                    fn(train_num_docs),
                    fn(valid_num_docs),
                    fn(test_num_docs),
                )
                (
                    train_weights,
                    train_num_samples,
                ) = get_normalized_weights_and_num_samples(train_weights, train_val_test_num_samples[0])
                (
                    valid_weights,
                    valid_num_samples,
                ) = get_normalized_weights_and_num_samples(valid_weights, train_val_test_num_samples[1])
                test_weights, test_num_samples = get_normalized_weights_and_num_samples(
                    test_weights, train_val_test_num_samples[2]
                )

                # rebuild datasets weighted according to new weights
                train_datasets, valid_datasets, test_datasets = build_weighted_datasets(
                    global_config,
                    train_num_samples,
                    valid_num_samples,
                    test_num_samples,
                    train_weights,
                    valid_weights,
                    test_weights,
                )

            if train_datasets:
                train_ds = BlendableDataset(train_datasets, train_weights)
            if valid_datasets:
                valid_ds = BlendableDataset(valid_datasets, valid_weights)
            if test_datasets:
                test_ds = BlendableDataset(test_datasets, test_weights)
        else:
            # when just data_path is provided
            # split dataset into train, valid and test from data_path
            train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
                data_prefix=global_config.data_path,
                use_shared_fs=global_config.use_shared_fs,
                data_impl=global_config.data_impl,
                splits_string=global_config.split,
                train_valid_test_num_samples=train_val_test_num_samples,
                seq_length=global_config.seq_length,
                seed=global_config.seed,
                skip_warmup=(not global_config.mmap_warmup),
                enforce_sample_length=global_config.enforce_sample_length,
            )
        return train_ds, valid_ds, test_ds


def compile_helper():
    """Compile helper function at runtime. Make sure this
    is invoked on a single process."""
    import os
    import subprocess

    # path = os.path.abspath(os.path.dirname(__file__))
    # ret = subprocess.run(["make", "-C", path])
    # if ret.returncode != 0:
    #     print("Making C++ dataset helpers module failed, exiting.")
    #     import sys

    #     sys.exit(1)
