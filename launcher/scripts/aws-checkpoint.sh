BASE_BUCKET=s3://nv-arc-dna-us-east-1
REMOTE_DIR=chkpt/40b-train-n256/40b_train #chkpt/40b-async-interval-test-n4/40b_train #chkpt/40b-test-n4-chkpt-test/40b_train
ITERATION=2000
BUCKET=$BASE_BUCKET/$REMOTE_DIR

FLAGS="--recursive --summarize"
CMD="/home/jeromek/aws-cli/bin/aws s3 ls $FLAGS $BUCKET/global_step$ITERATION"

echo $CMD
eval $CMD

# DEST=/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/s3-check
# mkdir -p $DEST

# LIST_CHKPT="~/aws-cli/bin/aws s3 ls --recursive --human-readable --summarize $BUCKET | tee step$ITERATION.txt"
# RM_CHKPT="~/aws-cli/bin/aws s3 rm --dryrun --recursive $BUCKET"
# CP_CMD="~/aws-cli/bin/aws s3 cp --recursive $BUCKET $DEST"
# echo $CP_CMD
