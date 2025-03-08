SOURCE_MP_SIZE=8
SOURCE_DP_SIZE=256
TARGET_MP_SIZE=16
TARGET_DP_SIZE=2
CONTEXT_LEN="8K"
NUM_WORKERS=1
SOURCE_ZERO3_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/${CONTEXT_LEN}/zero3/global_step516000"
SHARDED_ZERO1_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/${CONTEXT_LEN}/interleaved/zero1/MP${TARGET_MP_SIZE}/global_step516000"
OUTPUT_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/${CONTEXT_LEN}/interleaved/zero3/MP${TARGET_MP_SIZE}DP${TARGET_DP_SIZE}/padded/global_step0"

CMD="python partition_param.py --source_mp_size $SOURCE_MP_SIZE --source_dp_size $SOURCE_DP_SIZE --target_mp_size $TARGET_MP_SIZE --target_dp_size $TARGET_DP_SIZE --source_zero3_dir $SOURCE_ZERO3_DIR --sharded_zero1_dir $SHARDED_ZERO1_DIR --output_dir $OUTPUT_DIR --num_workers $NUM_WORKERS --pad_mlp_weights"

echo $CMD
eval $CMD