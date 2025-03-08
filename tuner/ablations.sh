SAVANNA_ROOT=$(realpath ..)
CONFIGS_DIR=$SAVANNA_ROOT/configs/launcher-test
DATA_CONFIG=$CONFIGS_DIR/data_configs/opengenome.yaml
MODEL_CONFIG=$CONFIGS_DIR/model_configs/ablations_nvidia/7b_stripedhyena2_base_4M.yml
# echo $DATA_CONFIG
# echo $MODEL_CONFIG
WANDB_PROJECT="7b-ablations
python tuner.py --data-config $DATA_CONFIG --template $MODEL_CONFIG --wandb_project