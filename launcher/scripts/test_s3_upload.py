import os

import boto3

# AWS S3 client
s3_client = boto3.client('s3')

# Configuration
bucket_name = "nv-arc-dna-us-east-1"
s3_key = "chkpt/tests/test.txt"
local_file_path = "scripts/test.txt"

# Step 1: Create a local file to upload
with open(local_file_path, "w") as file:
    file.write("This is the original content.")

# Step 2: Upload the file to S3
print(f"Uploaded original file to {bucket_name}/{s3_key}")
s3_client.upload_file(local_file_path, bucket_name, s3_key)

# Step 3: Modify the file locally
with open(local_file_path, "w") as file:
    file.write("This is the new content, overwriting the original.")

#Step 4: Upload the modified file to the same S3 key
s3_client.upload_file(local_file_path, bucket_name, s3_key)
print(f"Uploaded modified file to {bucket_name}/{s3_key}")

# Step 5: Download the file from S3 to verify the content
downloaded_file_path = "downloaded_test_file.txt"
s3_client.download_file(bucket_name, s3_key, downloaded_file_path)

with open(downloaded_file_path, "r") as file:
    content = file.read()

print(f"Content of the file after re-upload: {content}")
print()
# # # Cleanup
# # os.remove(local_file_path)
# # os.remove(downloaded_file_path)
