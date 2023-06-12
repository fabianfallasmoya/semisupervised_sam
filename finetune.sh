#!/bin/bash

GPU_NUM=0
export CUDA_VISIBLE_DEVICES=${GPU_NUM}
ROOT="../coco_bear/"
OUTPUT_FOLDER="finetune"
NUM_CLASSES=1
LOSS="mae"
OPTIM="sgd"
# SEMI_PERCENTAGE=1.0
DATASET="cocobear"
BATCH_SIZE=1
BATCH_SIZE_VAL=1
IMG_RESOLUTION=1024
NEW_SAMPLE_SIZE=256
NUMA=0
SEED=10
SAM_MODEL="b"
# current_model='TIMM NAME'

set -x
python fine_tune_sam.py \
    --root ${ROOT} \
    --num-classes 1 \
    --load-pretrained \
    --loss ${LOSS} \
    --optim ${OPTIM} \
    --dataset ${DATASET} \
    --batch-size ${BATCH_SIZE} \
    --batch-size-val ${BATCH_SIZE_VAL} \
    --img-resolution ${IMG_RESOLUTION} \
    --new-sample-size ${NEW_SAMPLE_SIZE} \
    --numa ${NUMA} \
    --seed ${SEED} \
    --sam-model ${SAM_MODEL} \
    --output-folder ${OUTPUT_FOLDER}
    # --semi-percentage ${SEMI_PERCENTAGE} \
    # --timm-model ${current_model} \