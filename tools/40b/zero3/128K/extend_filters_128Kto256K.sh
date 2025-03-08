SOURCE_CONTEXT_LEN=128K
TARGET_CONTEXT_LEN=256K
MP_SIZE=16
DP_SIZE=128
TAG="global_step0"
CHECKPOINT_DIR=/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/${SOURCE_CONTEXT_LEN}/interleaved/zero3/MP${MP_SIZE}DP${DP_SIZE}/padded
OUTPUT_DIR=/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/${TARGET_CONTEXT_LEN}/interleaved/zero3/MP${MP_SIZE}DP${DP_SIZE}/padded

SOURCE_MODEL_CONFIG="/lustre/fs01/portfolios/dir/users/jeromek/savanna-40b-long-ext/configs/40b/model_configs/extension/128K/40b_128K_no_rc.yml"
TARGET_MODEL_CONFIG="/lustre/fs01/portfolios/dir/users/jeromek/savanna-40b-long-ext/configs/40b/model_configs/extension/256K/40b_256K.yml"
START_MP_RANK=${1:-0}
# Calculate MP_SIZE - 1
#END_MP_SIZE=$((${MP_SIZE} - 1))
END_MP_RANK=${2:-${MP_SIZE}}

NUM_WORKERS=-1
CMD="python extend_zero3_checkpoint.py $CHECKPOINT_DIR \
--output_dir $OUTPUT_DIR \
--mp_size $MP_SIZE \
--dp_size $DP_SIZE \
--source_model_config $SOURCE_MODEL_CONFIG \
--target_model_config $TARGET_MODEL_CONFIG \
--tag $TAG \
--num_workers $NUM_WORKERS \
--start_mp_rank $START_MP_RANK \
--end_mp_rank $END_MP_RANK"

echo $CMD
eval $CMD

