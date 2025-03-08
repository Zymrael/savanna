SOURCE_MP_SIZE=8
SOURCE_DP_SIZE=256
TARGET_MP_SIZE=16
TARGET_DP_SIZE=64
CONTEXT_LEN="128K"
NUM_WORKERS=1
SOURCE_TAG="global_step12500"
OUTPUT_TAG="global_step0"
SOURCE_ZERO3_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/${CONTEXT_LEN}/zero3/${SOURCE_TAG}"
SHARDED_ZERO1_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/${CONTEXT_LEN}/interleaved/zero1/MP${TARGET_MP_SIZE}/${SOURCE_TAG}"
TARGET_MODEL_CONFIG_PATH="/lustre/fs01/portfolios/dir/users/jeromek/savanna-40b-long-ext/configs/40b/model_configs/extension/256K/40b_256K.yml"
OUTPUT_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/${CONTEXT_LEN}/interleaved/zero3/MP${TARGET_MP_SIZE}DP${TARGET_DP_SIZE}/padded/${OUTPUT_TAG}"
START_MP_RANK=${1:-0}
END_MP_RANK=${2:-${TARGET_MP_SIZE}}

CMD="python partition_param.py --source_mp_size $SOURCE_MP_SIZE --source_dp_size $SOURCE_DP_SIZE --target_mp_size $TARGET_MP_SIZE --target_dp_size $TARGET_DP_SIZE \
--source_zero3_dir $SOURCE_ZERO3_DIR \
--sharded_zero1_dir $SHARDED_ZERO1_DIR \
--output_dir $OUTPUT_DIR \
--num_workers $NUM_WORKERS \
--pad_mlp_weights \
--target_model_config_path $TARGET_MODEL_CONFIG_PATH \
--start_mp_rank $START_MP_RANK \
--end_mp_rank $END_MP_RANK"

echo $CMD
eval $CMD