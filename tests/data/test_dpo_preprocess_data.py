import pytest
import subprocess


def run_preprocess_data_command(command):
    result = subprocess.run(command, shell=True, capture_output=True)
    return result


def test_dpo_preprocess_data_autolength():
    command = """
    python tools/preprocess_data_dpo.py \
        --input tests/data/test_dpo.jsonl \
        --output-prefix tests/data/test_dpo_autolength \
        --dataset-impl mmap \
        --tokenizer-type CharLevelTokenizer
    """
    result = run_preprocess_data_command(command)
    assert result.returncode == 0


def test_dpo_preprocess_data_fixedlength():
    command = """
    python tools/preprocess_data_dpo.py \
        --input tests/data/test_dpo.jsonl \
        --output-prefix tests/data/test_dpo_fixedlength \
        --enforce-sample-length 50 \
        --dataset-impl mmap \
        --tokenizer-type CharLevelTokenizer
    """
    result = run_preprocess_data_command(command)
    assert result.returncode == 0


def test_dpo_preprocess_data_fail_fixedlength():
    command = """
    python tools/preprocess_data_dpo.py \
        --input tests/data/test_dpo.jsonl \
        --output-prefix tests/data/test_dpo_fixedlength \
        --enforce-sample-length 10 \
        --dataset-impl mmap \
        --tokenizer-type CharLevelTokenizer
    """
    result = run_preprocess_data_command(command)
    assert result.returncode == 1
