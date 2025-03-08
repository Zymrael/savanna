SOURCE_CONTEXT_LEN=128K
TARGET_CONTEXT_LEN=1M
MP_SIZE=64
DP_SIZE=16
TAG="global_step0"
ROOT_DIR="/lustre/fs01/portfolios/dir/users/jeromek/savanna-40b-ctx-fixed"

#Model configs
SOURCE_MODEL_CONFIG="${ROOT_DIR}/configs/40b/model_configs/extension/128K/40b_128K_no_rc.yml"
TARGET_MODEL_CONFIG="${ROOT_DIR}/configs/40b/model_configs/extension/1M/40b_1M.yml"

# Repartitioned checkpoints
CHECKPOINT_DIR=/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/${SOURCE_CONTEXT_LEN}/interleaved/zero3/MP${MP_SIZE}DP${DP_SIZE}/padded
OUTPUT_DIR=/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/${TARGET_CONTEXT_LEN}/interleaved/zero3/MP${MP_SIZE}DP${DP_SIZE}/padded

# Processes all ranks START_MP_RANK <= rank <= END_MP_RANK
START_MP_RANK=${1:-0}
END_MP_RANK=${2:-${MP_SIZE}}

# Each process handles a model / optim state path, so no need to restrict to rank
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
# eval $CMD