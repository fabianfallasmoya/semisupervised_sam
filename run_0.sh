GPU_NUM=3
export CUDA_VISIBLE_DEVICES=${GPU_NUM}
MODELS=( 'TIMM NAME' ) 
ROOT="../../../share/semi_supervised/COCO_BEARS/coco_bear/"
NUM_CLASSES=1
LOSS="mae"
OPTIM="sgd"
SEMI_PERCENTAGE=1.0
DATASET="coco2017"
BATCH_SIZE=4
BATCH_SIZE_VAL=1
IMG_RESOLUTION=1024
NEW_SAMPLE_SIZE=256
NUMA=0
SEED=10
SAM_MODEL="h"

for current_model in "${MODELS[@]}"
do
    python run.py \
        --root ${ROOT} \ 
        --num-classes 1 \
        --load-pretrained \
        --timm-model ${current_model} \
        --loss ${LOSS} \
        --optim ${OPTIM} \
        --use-semi-split \
        --semi-percentage ${SEMI_PERCENTAGE} \
        --dataset ${DATASET} \
        --batch-size ${BATCH_SIZE} \
        --batch-size-val ${BATCH_SIZE_VAL} \
        --img-resolution ${IMG_RESOLUTION} \
        --new-sample-size ${NEW_SAMPLE_SIZE} \
        --numa ${NUMA} \
        --seed ${SEED} \
        --sam-model ${SAM_MODEL}
    sleep 55
done