#!/usr/bin/env bash
set -e

BASE_DIR=/home/askrbayern/Projects/audio-compression/stable-audio-tools
MODEL_CONFIG=${BASE_DIR}/stable_audio_tools/configs/model_configs/test/FSQ_config.json
DATA_DIR=/home/askrbayern/Projects/audio-compression/7-stable-audio/train_folder/
INPUT_AUDIO=${BASE_DIR}/test_folder/test.mp3
POST_OUTPUT=${BASE_DIR}/test_folder/test_post.mp3

BATCH_SIZE=1
NUM_WORKERS=6
MAX_EPOCHS=50

python train_server.py \
  --model-config $MODEL_CONFIG \
  --data-dir $DATA_DIR \
  --input-audio $INPUT_AUDIO \
  --post-output $POST_OUTPUT \
  --batch-size $BATCH_SIZE \
  --num-workers $NUM_WORKERS \
  --max-epochs $MAX_EPOCHS