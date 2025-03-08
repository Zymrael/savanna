BASE_BUCKET=s3://nv-arc-dna-us-east-1
REMOTE_DIR=chkpt/40b-train-n256-v2/40b_train_v2 #chkpt/40b-async-interval-test-n4/40b_train #chkpt/40b-test-n4-chkpt-test/40b_train
ITERATION=278000
BUCKET=$BASE_BUCKET/$REMOTE_DIR
SOURCE=$BUCKET/global_step$ITERATION

FLAGS="--recursive --summarize"
CMD="/home/jeromek/aws-cli/bin/aws s3 ls $FLAGS $BUCKET/global_step$ITERATION"

# echo $CMD
# eval $CMD

DEST=/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-backup/global_step$ITERATION
mkdir -p $DEST
# ls -lth $DEST

# DRY_CP_CMD="~/aws-cli/bin/aws s3 cp --recursive --dryrun $SOURCE $DEST"
# echo $DRY_CP_CMD
# eval $DRY_CP_CMD

# CP_CMD="~/aws-cli/bin/aws s3 cp --recursive $SOURCE $DEST"
# echo $CP_CMD
# eval $CP_CMD

#--exact-timestamps
SYNC_CMD="~/aws-cli/bin/aws s3 sync $SOURCE $DEST"
echo $SYNC_CMD
eval $SYNC_CMD