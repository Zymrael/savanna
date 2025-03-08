import pandas as pd
import tarfile

import pathlib

path_to_dataset = pathlib.Path("/var/cr01_data/long_finetune/zlib-books-1k-100k")

# process all .tar.parquet folders
# for folder in path_to_dataset.iterdir():
# read all parquet files
for file in path_to_dataset.iterdir():
    if file.suffix == ".parquet":
        try:
            df = pd.read_parquet(file, engine="pyarrow")
            print(df.head())
            df.to_json("output.json", orient="records", lines=True)
        except:
            pass
