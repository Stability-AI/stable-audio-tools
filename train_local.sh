#!/usr/bin/env bash
set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <wandb_project> <wandb_run_name>"
    exit 1
fi

WANDB_PROJECT=$1
WANDB_RUN_NAME=$2

BASE_DIR=/home/askrbayern/Projects/audio-compression/stable-audio-tools-fsq
MODEL_CONFIG=${BASE_DIR}/stable_audio_tools/configs/model_configs/test/RVQ_config.json
DATA_DIR=/home/askrbayern/Projects/audio-compression/7-stable-audio/train_folder/
INPUT_AUDIO=${BASE_DIR}/test_folder/test.mp3
OUTPUT_DIR=${BASE_DIR}/test_folder

BATCH_SIZE=1
NUM_WORKERS=6
MAX_EPOCHS=10

echo "Model config: $MODEL_CONFIG"
echo "Project: $WANDB_PROJECT"
echo "Run name: $WANDB_RUN_NAME"

python train_start.py \
  --model-config $MODEL_CONFIG \
  --data-dir $DATA_DIR \
  --input-audio $INPUT_AUDIO \
  --output-dir $OUTPUT_DIR \
  --batch-size $BATCH_SIZE \
  --num-workers $NUM_WORKERS \
  --max-epochs $MAX_EPOCHS \
  --wandb-project $WANDB_PROJECT \
  --wandb-run-name $WANDB_RUN_NAME