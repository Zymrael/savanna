import pytest
import os
import subprocess
import torch
import torch.distributed

from savanna.arguments import GlobalConfig
from savanna.mpu.initialize import initialize_model_parallel
from savanna.data.data_utils import build_train_valid_test_data_iterators


# Pretend like we are on a rank 0 host.
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "8000"
torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)
initialize_model_parallel(1)

# Compile data helpers.
subprocess.run(
    ["make"],
    cwd="./savanna/data",
    shell=True,
)


def test_training_uniform_lengths():
    """
    Test if we can iterate over correctly processed data.
    """
    # Preprocess data.
    command = """
    python tools/preprocess_data.py \
        --input tests/data/test_uniform_lengths.jsonl \
        --output-prefix tests/data/test_uniform_lengths_pad_eod \
        --dataset-impl mmap \
        --tokenizer-type CharLevelTokenizer \
        --append-eod \
        --enforce-sample-length 18 \
        --workers 1 \
        --log-interval 1
    """
    result = subprocess.run(command, shell=True, capture_output=True)
    assert result.returncode == 0

    global_config = GlobalConfig.from_ymls(["tests/test_configs/test_uniform_lengths.yml"])

    (
        train_data_iterator,
        valid_data_iterator,
        test_data_iterator,
    ) = build_train_valid_test_data_iterators(global_config=global_config)

    for _ in range(5):
        assert next(train_data_iterator)
        assert next(valid_data_iterator)
        assert next(test_data_iterator)


def test_training_nonuniform_lengths():
    """
    Test if the training loop can catch an improperly formatted dataset.
    """
    # Preprocess data.
    command = """
    python tools/preprocess_data.py \
        --input tests/data/test_uniform_lengths.jsonl \
        --output-prefix tests/data/test_uniform_lengths_nopadding \
        --dataset-impl mmap \
        --tokenizer-type CharLevelTokenizer \
        --append-eod \
        --workers 1 \
        --log-interval 1
    """
    result = subprocess.run(command, shell=True, capture_output=True)
    assert result.returncode == 0

    global_config = GlobalConfig.from_ymls(["tests/test_configs/test_uniform_lengths.yml"])
    global_config.train_data_paths = [
        "tests/data/test_uniform_lengths_nopadding_text_CharLevelTokenizer_document"
    ]
    global_config.test_data_paths = [
        "tests/data/test_uniform_lengths_nopadding_text_CharLevelTokenizer_document"
    ]
    global_config.valid_data_paths = [
        "tests/data/test_uniform_lengths_nopadding_text_CharLevelTokenizer_document"
    ]

    with pytest.raises(ValueError) as exc_info:
        (
            train_data_iterator,
            valid_data_iterator,
            test_data_iterator,
        ) = build_train_valid_test_data_iterators(global_config=global_config)

    assert str(exc_info.value).startswith("Found sample with invalid length ")
