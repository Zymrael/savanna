from savanna.data.indexed_dataset import MMapIndexedDataset
from transformers import AutoTokenizer

import argparse

# get the first argument as a file name, and an output file
parser = argparse.ArgumentParser()
parser.add_argument("file_name", help="the file name to read")
args = parser.parse_args()

ds = MMapIndexedDataset(args.file_name)

tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

print(tok.decode(ds[0]))
print(tok.decode(ds[10]))
print(tok.decode(ds[100]))
