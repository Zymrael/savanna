"""Processing data for pretraining."""
import argparse
import multiprocessing
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import time
import tqdm
import torch
import ftfy

import zstandard
import ujson as json
import time
from functools import reduce
import jsonlines
import io
from zipfile import ZipFile
import gzip
from math import ceil
import mmap
import multiprocessing as mp
from pathlib import Path

from savanna.tokenizer import build_tokenizer
from savanna.data import indexed_dataset
from threading import Semaphore


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)

    def encode(self, text):
        if self.args.ftfy:
            text = ftfy.fix_text(text)
        ids = {}
        for key in self.args.jsonl_keys:
            doc_ids = []
            text_ids = Encoder.tokenizer.tokenize(text)
            if (
                self.args.enforce_sample_length
                and (len(text_ids) + int(self.args.append_eod)) > self.args.enforce_sample_length
            ):
                raise ValueError(
                    "Detected input text with a length greater than the maximum "
                    f"possible sample length of {self.args.enforce_sample_length}.)"
                )
            if len(text_ids) > 0:
                doc_ids.append(text_ids)
            if self.args.append_eod:
                doc_ids[-1].append(Encoder.tokenizer.eod)
            if self.args.enforce_sample_length:
                # Pad up to max sequence length.
                doc_ids[-1] += [Encoder.tokenizer.pad] * (self.args.enforce_sample_length - len(doc_ids[-1]))
            ids[key] = doc_ids
        return ids, len(text)


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
        "--jsonl-keys",
        nargs="+",
        default=["text"],
        help="space separate listed of keys to extract from jsonl. Defa",
    )
    group.add_argument(
        "--num-docs",
        default=None,
        help="Optional: Number of documents in the input data (if known) for an accurate progress bar.",
        type=int,
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
    group.add_argument("--vocab-file", type=str, default=None, help="Path to the vocab file")
    group.add_argument(
        "--merge-file",
        type=str,
        default=None,
        help="Path to the BPE merge file (if necessary).",
    )
    group.add_argument(
        "--append-eod",
        action="store_true",
        help="Append an <eod> token to the end of a document.",
    )
    group.add_argument(
        "--enforce-sample-length",
        type=int,
        default=None,
        help="Forces all samples to have the specified length. If shorter, pads up to the length. If longer, throws an error.",
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

    group = parser.add_argument_group(title="runtime")
    group.add_argument("--workers", type=int, default=1, help="Number of worker processes to launch")
    group.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Interval between progress updates",
    )
    args = parser.parse_args()
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.model_parallel_size = 1

    return args



########################################################################################
# Reader and utilities from https://github.dev/leogao2/lm_dataformat
########################################################################################

VALID_EXTENSIONS = ['openwebtext.tar.xz', '_data.xz', '.dat.zst', '.jsonl', '.jsonl.zst', '.jsonl.zst.tar', '.json.zst', '.txt', '.zip', '.tar.gz', '.json.gz', '.gz']

def has_valid_extension(file):
    return any([file.endswith(ext) for ext in VALID_EXTENSIONS])


def _listdir_or_file(x):
    if isinstance(x, list):
        return reduce(lambda x, y: x + y, map(listdir_or_file, sorted(x)))
    if os.path.isfile(x):
        return [x]
    elif os.path.isdir(x):
        return [str(Path(x) / fn) for fn in sorted(os.listdir(x))]
    else:
        raise FileNotFoundError(f"{x} not found")

def listdir_or_file(x):
    return list(filter(has_valid_extension, _listdir_or_file(x)))


def handle_jsonl(jsonl_reader, get_meta, autojoin_paragraphs, para_joiner, key='text'):
    for ob in jsonl_reader:
        # naive jsonl where each object is just the string itself, with no meta. For legacy compatibility.
        if isinstance(ob, str):
            assert not get_meta
            yield ob
            continue

        text = ob[key]

        if autojoin_paragraphs and isinstance(text, list):
            text = para_joiner.join(text)

        if get_meta:
            yield text, (ob['meta'] if 'meta' in ob else {})
        else:
            yield text


class Reader:
    def __init__(self, in_path):
        self.in_path = in_path
    
    def stream_data(self, get_meta=False, threaded=False):
        if not threaded:
            yield from self._stream_data(get_meta)
            return

        q = mp.Queue(1000)
        p = mp.Process(target=self._stream_data_threaded, args=(q, get_meta))
        p.start()
        while p.is_alive():
            res = q.get()
            if res is None: break
            yield res
    
    def _stream_data_threaded(self, q, get_meta=False):
        for data in self._stream_data(get_meta):
            q.put(data)
        q.put(None)

    def _stream_data(self, get_meta=False, jsonl_key="text"):
        self.f_name = ""
        files = listdir_or_file(self.in_path)
        if not files:
            raise FileNotFoundError(f"No valid file(s) found in {self.in_path}")
        for f in files:
            self.f_name = f
            if f == 'openwebtext.tar.xz':
                assert not get_meta

                yield from self.read_owt(f)
            elif 'urlsf_subset' in f and f.endswith('_data.xz'):
                assert not get_meta

                yield from self.read_owt_subset(f)
            elif f.endswith('.dat.zst'):
                assert not get_meta

                yield from self.read_dat(f)
            elif f.endswith('.jsonl'):
                yield from self.read_jsonl(f, get_meta, key=jsonl_key)
            elif f.endswith('.jsonl.zst'):
                yield from self.read_jsonl_zst(f, get_meta, key=jsonl_key)
            elif f.endswith('.json.zst'):
                assert not get_meta

                yield from self.read_json(f)
            elif f.endswith('.txt'):
                assert not get_meta
                
                yield from self.read_txt(f)
            elif f.endswith('.zip'):
                assert not get_meta

                yield from self.read_zip(f)
            elif f.endswith('.tar.gz'):
                assert not get_meta

                yield from self.read_tgz(f)
            elif f.endswith('.json.gz'):
                assert not get_meta
                
                yield from self.read_jsongz(f)
            elif f.endswith('.gz'):
                assert not get_meta
               
                yield from self.read_gz(f)
            else:
                # shouldn't be reached
                print(f'Skipping {f} as streaming for that filetype is not implemented')

    def read_txt(self, file):
        with open(file, 'r') as fh:
            yield fh.read()

    def read_zip(self, file):
        archive = ZipFile(file, 'r')
        for f in archive.namelist():
            yield archive.read(f).decode('UTF-8')

    def read_gz(self, file): 
        with gzip.open(file, 'rb') as f:
            for line in f:
                yield line.decode('utf-8')
                
    def read_jsongz(self, file): 
        for line in self.read_gz(file):
            yield json.loads(line)
                
    def read_json(self, file):
        with open(file, 'rb') as fh:
            cctx = zstandard.ZstdDecompressor()
            reader = cctx.stream_reader(fh)
            ob = json.load(reader)
            yield from ob

    def read_dat(self, file):
        with open(file, 'rb') as fh:
            cctx = zstandard.ZstdDecompressor()
            reader = cctx.stream_reader(fh)
            while True:
                ln = reader.read(16).decode('UTF-8')
                if not ln:
                    break

                ln = int(ln)

                yield reader.read(ln).decode('UTF-8')

    def read_jsonl(self, file, get_meta=False, autojoin_paragraphs=True, para_joiner='\n\n', key='text'):
        with jsonlines.open(file) as rdr:
            yield from handle_jsonl(rdr, get_meta, autojoin_paragraphs, para_joiner, key)
            
    def read_jsonl_zst(self, file, get_meta=False, autojoin_paragraphs=True, para_joiner='\n\n', key='text'):
        with open(file, 'rb') as fh:
            cctx = zstandard.ZstdDecompressor()
            reader = io.BufferedReader(cctx.stream_reader(fh))
            rdr = jsonlines.Reader(reader)
            yield from handle_jsonl(rdr, get_meta, autojoin_paragraphs, para_joiner, key)



def yield_from_files(fnames: list, semaphore):
    """
    Iterator over input documents using lm_dataformat. Should be able to handle jsons / texts /
    other compressed formats. Also filters out empty documents.

    :param fnames: list of filenames
    """

    def yielder(fname, semaphore):
        for f in filter(lambda x: x, Reader(fname).stream_data()):
            semaphore.acquire()
            yield f

    for fname in fnames:
        semaphore.acquire()

        yield from yielder(fname, semaphore)


def main():
    args = get_args()
    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")

    # build a semaphore object to stop `yield_from_files` from getting ahead of encoder.encode and
    # hence building up memory
    semaphore = Semaphore(10000 + args.workers)

    # use multiprocessing to iterate over input documents
    fin = yield_from_files(args.input.split(","), semaphore)
    print(fin)
    if args.workers > 1:
        pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, fin, chunksize=25)
    else:
        encoder.initializer()
        encoded_docs = (encoder.encode(doc) for doc in fin)

    # make a dataset builder for each key in args.jsonl_keys
    # each key will output to a different file beginning with args.output_prefix
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    tokenizer_name = tokenizer.name.replace(" ", "")
    for key in args.jsonl_keys:
        output_bin_files[key] = "{}_{}_{}_{}.bin".format(args.output_prefix, key, tokenizer_name, "document")
        output_idx_files[key] = "{}_{}_{}_{}.idx".format(args.output_prefix, key, tokenizer_name, "document")
        builders[key] = indexed_dataset.make_builder(
            output_bin_files[key],
            impl=args.dataset_impl,
            vocab_size=tokenizer.vocab_size,
        )

    # actually do tokenization
    proc_start = time.time()
    total_bytes_processed = 0
    pbar = tqdm.tqdm()
    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed

        # release semaphore so `yield_from_files` can add another file to the buffer
        semaphore.release()

        # add each tokenized document / sentence
        for key, sentences in doc.items():
            for sentence in sentences:
                builders[key].add_item(np.array(sentence, dtype=builders[key].dtype))
            # tell the builder that a document has finished
            builders[key].end_document()

        # log progress
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            pbar.set_description(
                f"Processed {i}{'' if args.num_docs is None else '/' + str(args.num_docs)} documents ({i / elapsed} docs/s, {mbs} MB/s)."
            )
            if i != 0:
                pbar.update(args.log_interval)

    # save output file
    for key in args.jsonl_keys:
        builders[key].finalize(output_idx_files[key])


if __name__ == "__main__":
    main()
