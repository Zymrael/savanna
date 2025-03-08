CHECKPOINT_DIR=/lustre/fs01/portfolios/dir/users/jeromek/savanna-data-debug/tools/40b/zero3/test_extended_checkpoints/32layer/8K #"/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-checkpoint-tests/4layer_zero3/"
OUTPUT_DIR="refactored/32layer/16K"
MP_SIZE=8
DP_SIZE=4

SOURCE_MODEL_CONFIG="/lustre/fs01/portfolios/dir/users/jeromek/savanna-data-debug/configs/40b/model_configs/tests/checkpoint_loading/32layer_zero3.yml" #"/lustre/fs01/portfolios/dir/users/jeromek/savanna-data-debug/configs/40b/model_configs/tests/checkpoint_loading/4layer_zero3.yml"
TARGET_MODEL_CONFIG="/lustre/fs01/portfolios/dir/users/jeromek/savanna-data-debug/configs/40b/model_configs/tests/checkpoint_loading/32layer_zero3-extended.yml"

CMD="python extend_zero3_checkpoint.py $CHECKPOINT_DIR --output_dir $OUTPUT_DIR --mp_size $MP_SIZE --dp_size $DP_SIZE --source_model_config $SOURCE_MODEL_CONFIG --target_model_config $TARGET_MODEL_CONFIG"

echo $CMD
eval $CMD

