GPU_NUM=3
export CUDA_VISIBLE_DEVICES=${GPU_NUM}
MODELS=( 'resnet10t.c3_in1k' ) 
ROOT="../../../share/semi_supervised/pineapples/fruto_maduro/" #"../../../share/semi_supervised/COCO_BEARS/coco_bear/"
NUM_CLASSES=1
LOSS="mae"
OPTIM="sgd"
SEMI_PERCENTAGE=1.
DATASET="coco2017"
BATCH_SIZE=4
BATCH_SIZE_VAL=1
IMG_RESOLUTION=1024
NEW_SAMPLE_SIZE=256
SEED=10
SAM_MODEL="h"
AUG_METHOD="rand_augmentation"
OUTPUT_FOLDER="testing"
RUN_NAME="temp0"

for current_model in "${MODELS[@]}"
do
    python run.py \
        --root ${ROOT} \
        --num-classes ${NUM_CLASSES} \
        --load-pretrained \
        --timm-model ${current_model} \
        --loss ${LOSS} \
        --optim ${OPTIM} \
        --val-freq 1 \
        --use-semi-split \
        --semi-percentage ${SEMI_PERCENTAGE} \
        --dataset ${DATASET} \
        --batch-size ${BATCH_SIZE} \
        --batch-size-val ${BATCH_SIZE_VAL} \
        --aug-method ${AUG_METHOD} \
        --img-resolution ${IMG_RESOLUTION} \
        --new-sample-size ${NEW_SAMPLE_SIZE} \
        --numa ${GPU_NUM} \
        --output-folder ${OUTPUT_FOLDER} \
        --run-name ${RUN_NAME} \
        --seed ${SEED} \
        --sam-model ${SAM_MODEL}
    sleep 55
done