"""
Usage: python tools/preprocess_data_dpo.py \
           --input /path/to/file.jsonl \
           --output-prefix /path/to/output_prefix
           --enforce-sample-length 1000 \
           --dataset-impl mmap \
           --tokenizer-type CharLevelTokenizer

Process data for DPO.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from typing import Any, Dict, Generator

import ftfy
from tqdm import tqdm

from savanna.data import indexed_dataset
from savanna.tokenizer import build_tokenizer


class DPOEncoder:
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use DPOEncoder class as a container for global data
        DPOEncoder.tokenizer = build_tokenizer(self.args)

    def encode(self, jsonl_data: Dict, pad: bool = True) -> list[float]:
        """
        For DPO, each document has:
        1. The integer tokens for the accepted text, followed by a pad token.
        2. The integer tokens for the rejected text, followed by a pad token.
        3. The float logprob value for the accepted text under the reference model,
            followed by a pad token.
        4. The float logprob value for the rejected text under the reference model.
        5. EOD tokens padded up to the maximum sequence length.
        """
        text_accept = jsonl_data['text_accept']
        text_reject = jsonl_data['text_reject']
        ref_logprob_accept = float(jsonl_data['ref_logprob_accept'])
        ref_logprob_reject = float(jsonl_data['ref_logprob_reject'])
        
        if self.args.ftfy:
            text_accept = ftfy.fix_text(text_accept)
            text_reject = ftfy.fix_text(text_reject)

        text_ids_accept = DPOEncoder.tokenizer.tokenize(text_accept)
        text_ids_reject = DPOEncoder.tokenizer.tokenize(text_reject)

        data = (
            text_ids_accept + [ DPOEncoder.tokenizer.pad ] +
            text_ids_reject + [ DPOEncoder.tokenizer.pad ] +
            [ ref_logprob_accept ] + [ DPOEncoder.tokenizer.pad ] +
            [ ref_logprob_reject ]
        )

        if pad:
            if len(data) > self.args.enforce_sample_length:
                raise ValueError(
                    f'Found a document with total length {len(data)} but the max '
                    f'document length is {self.args.enforce_sample_length}.'
                )
            data += [ DPOEncoder.tokenizer.eod ] * \
                (self.args.enforce_sample_length - len(data))

        return data


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input jsonl files or lmd archive(s) - if using multiple archives, put them in a comma separated "
        "list",
    )
    group.add_argument(
        "--enforce-sample-length",
        type=int,
        default=None,
        help="Forces all samples to have the specified length. If shorter, pads up to the length. If longer, throws an error.",
    )
    group = parser.add_argument_group(title="tokenizer")
    group.add_argument(
        "--tokenizer-type",
        type=str,
        required=True,
        choices=[
            "HFGPT2Tokenizer",
            "HFTokenizer",
            "GPT2BPETokenizer",
            "CharLevelTokenizer",
            "TiktokenTokenizer",
        ],
        help="What type of tokenizer to use.",
    )
    group.add_argument(
        "--vocab-file", type=str, default=None, help="Path to the vocab file"
    )
    group.add_argument(
        "--merge-file",
        type=str,
        default=None,
        help="Path to the BPE merge file (if necessary).",
    )
    group.add_argument("--ftfy", action="store_true", help="Use ftfy to clean text")
    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Path to binary output file without suffix",
    )
    group.add_argument(
        "--dataset-impl",
        type=str,
        default="mmap",
        choices=["lazy", "cached", "mmap"],
        help="Dataset implementation to use. Default: mmap",
    )

    args = parser.parse_args()
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.model_parallel_size = 1

    return args

def iterate_jsonl_files(filenames: list[str]) -> Generator[Dict[str, Any], None, None]:
    for filename in filenames:
        with open(filename.strip(), 'r') as f:
            for line in f:
                yield json.loads(line)

def main():
    args = get_args()
    encoder = DPOEncoder(args)
    encoder.initializer()

    fin = iterate_jsonl_files(args.input.split(','))

    if args.enforce_sample_length is None:
        # Auto-compute the maximum sample length.
        print('Auto-computing the maximum sample length. To save time, consider '
              'precomputing this value and passing it as --enforce-sample-length.')
        max_length = 0
        for doc in fin:
            doc_length = len(encoder.encode(doc, pad=False))
            if doc_length > max_length:
                max_length = doc_length
        print(f'Maximum sample length is {max_length}.')
        args.enforce_sample_length = max_length
        fin = iterate_jsonl_files(args.input.split(',')) # Reset generator.

    tokenizer_name = args.tokenizer_type.replace(' ', '')
    output_bin_file = '{}_dpo_{}_{}.bin'.format(
        args.output_prefix, tokenizer_name, 'document'
    )
    output_idx_file = '{}_dpo_{}_{}.idx'.format(
        args.output_prefix, tokenizer_name, 'document'
    )
    builder = indexed_dataset.make_builder(
        output_bin_file,
        impl=args.dataset_impl,
        vocab_size=DPOEncoder.tokenizer.vocab_size,
        dtype=np.float32,
    )

    encoded_docs = (encoder.encode(doc) for doc in fin)

    for doc in tqdm(encoded_docs):
        builder.add_item(np.array(doc, dtype=np.float32))
        builder.end_document()

    builder.finalize(output_idx_file)


if __name__ == '__main__':
    main()
