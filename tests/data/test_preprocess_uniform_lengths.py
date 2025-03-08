import pytest
import subprocess


def run_preprocess_data_command(command):
    result = subprocess.run(command, shell=True, capture_output=True)
    return result


def test_preprocess_data_without_eod():
    command = """
    python tools/preprocess_data.py \
        --input tests/data/test_uniform_lengths.jsonl \
        --output-prefix tests/data/test_uniform_lengths_pad \
        --dataset-impl mmap \
        --tokenizer-type CharLevelTokenizer \
        --enforce-sample-length 18 \
        --workers 1 \
        --log-interval 1
    """
    result = run_preprocess_data_command(command)
    assert result.returncode == 0


def test_preprocess_data_with_eod():
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
    result = run_preprocess_data_command(command)
    assert result.returncode == 0


def test_preprocess_data_fail_over_by_one_with_eod():
    command = """
    python tools/preprocess_data.py \
        --input tests/data/test_uniform_lengths.jsonl \
        --output-prefix tests/data/test_uniform_lengths_fail \
        --dataset-impl mmap \
        --tokenizer-type CharLevelTokenizer \
        --append-eod \
        --enforce-sample-length 17 \
        --workers 1 \
        --log-interval 1
    """
    result = run_preprocess_data_command(command)
    assert result.returncode == 1


def test_preprocess_data_fail_over_by_one_without_eod():
    command = """
    python tools/preprocess_data.py \
        --input tests/data/test_uniform_lengths.jsonl \
        --output-prefix tests/data/test_uniform_lengths_fail \
        --dataset-impl mmap \
        --tokenizer-type CharLevelTokenizer \
        --enforce-sample-length 16 \
        --workers 1 \
        --log-interval 1
    """
    result = run_preprocess_data_command(command)
    assert result.returncode == 1
