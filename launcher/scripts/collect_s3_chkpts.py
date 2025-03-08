import argparse
import datetime
import os
import re

import boto3
import pandas as pd
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

# Initialize S3 client
s3 = boto3.client('s3')


def get_s3_client():
    try:
        s3_client = boto3.client(
        's3',
        aws_access_key_id='AKIAR6DSHXXL3N5AVXG3',
        aws_secret_access_key='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',
)
        # s3_client.list_buckets()
        return s3_client
    
    except (NoCredentialsError, PartialCredentialsError):
        print("AWS credentials not found in environment. Please enter them manually.")
        aws_access_key = input("Enter AWS Access Key: ")
        aws_secret_key = input("Enter AWS Secret Key: ")
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
        return s3_client


def list_and_parse_s3_objects(bucket_name, prefix=''):
    # List all objects in the specified bucket with the prefix
    objects = []
    raw_objs = []
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        while True:
            if 'Contents' in response:
                for obj in response['Contents']:
                    objects.append(obj['Key'] + f" {obj['Size']} bytes")
                    raw_objs.append(obj)
                    
            # Check if there's a continuation token (more objects to list)
            if 'NextContinuationToken' in response:
                response = s3.list_objects_v2(
                    Bucket=bucket_name,
                    Prefix=prefix,
                    ContinuationToken=response['NextContinuationToken']
                )
            else:
                break
    except Exception as e:
        print("Error listing objects:", e)
        return None

    return raw_objs

def format_df(df):
    df.columns = [col.strip() for col in df.columns]
    # common_prefix = prefix.split("/")[0]
    # print(f"Common prefix: {common_prefix}")
    df[['prefix', 'global_step', 'object_path']] = df['Key'].str.extract(rf'^(.*)/(global_step\d+)/(.*)$')
    df = df.drop("Key", axis=1)
    df = df.rename({"LastModified": "last_modified", "Size": "bytes"}, axis=1)
    df = df[['prefix','global_step','object_path', 'last_modified', 'bytes']]
    df['global_step'] = df['global_step'].str.extract(r'(\d+)').astype(int)
    # Add column of bytes in human readable format in KB, MB, and GB
    df['bytes_human'] = df['bytes'].apply(lambda x: f"{x/1024/1024:.2f} MB")
    return df.sort_values('global_step')

def summarize_df(df):
    print(f"Found {len(df)} objects across {len(df['global_step'].unique())} global steps")
    print(f"Total size: {df['bytes'].sum() / 1e9:.2f} GB")
    # print(f"Global steps: {df['global_step'].unique()}")
    steps = df.groupby("global_step")["bytes"].describe().reset_index()
    steps["count"] = steps["count"].astype(int)
    summary_stats = steps[["global_step", "count", "min","50%", "max"]]
    pd.options.display.max_categories = 100
    pd.options.display.max_columns = 100
    pd.options.display.max_colwidth = 1000
    
    print(summary_stats)
    return df

def save_df(df, output_path):
    print(f"Saving to {output_path}")
    
    df.to_csv(output_path, index=False)
    return df

def create_output_path(prefix, ts, output_dir="s3_logs"):
    fp = prefix.replace("/","_")
    fp = f"{fp}_{ts}.csv"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{fp}"
    return output_path

if __name__ == "__main__":

    BUCKET = "nv-arc-dna-us-east-1"
    PREFIX = "chkpt/40b-train-n256-v2/40b_train_v2"
    
    parser = argparse.ArgumentParser(description="List and parse S3 objects")
    parser.add_argument("--bucket", default=BUCKET, help="S3 bucket name")
    parser.add_argument("--prefix", default=PREFIX, help="S3 object prefix")
    parser.add_argument("--log_file", default=None, help="Log file to parse")
    args = parser.parse_args()
    print(f"Bucket: {args.bucket}, Prefix: {args.prefix}")

    output_path = create_output_path(args.prefix, datetime.datetime.now().strftime("%Y%m%d%H"))

    if not os.path.exists(output_path):
        print(f"Collecting s3 objects from {args.bucket} with prefix {args.prefix}")
        bucket_data = list_and_parse_s3_objects(args.bucket, prefix=args.prefix)
        df = pd.DataFrame(bucket_data)
        df = format_df(df)
        save_df(df, output_path)
    else:
        print(f"File already exists: {output_path}")
        df = pd.read_csv(output_path)

    summarize_df(df)