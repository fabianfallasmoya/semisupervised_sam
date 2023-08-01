GPU_NUM=3
export CUDA_VISIBLE_DEVICES=${GPU_NUM}
DATASET_NAME="PINEAPPLES2_5mts_nosplits"
ROOT="../../../share/semi_supervised/pineapples/${DATASET_NAME}" 
NUM_CLASSES=1
DATASET="coco2017"
BATCH_SIZE_LABELED=1
BATCH_SIZE_UNLABELED=1
OOD_LABELED_SAMPLES=1
OOD_UNLABELED_SAMPLES=2
NEW_SAMPLE_SIZE=256
SEED=10
SAM_MODEL="h"
OOD_THRESH=0.8
OOD_BINS=15
MODELS=( 'resnet10t.c3_in1k' ) 
METHODS=( 'sam_alone' 'sam_fewshot_single_class' 'sam_fewshot_two_classes' 'sam_ood' )


for current_model in "${MODELS[@]}"
do

    for curr_method in "${METHODS[@]}"
    do
        python methods.py \
            --root ${ROOT} \
            --num-classes ${NUM_CLASSES} \
            --load-pretrained \
            --timm-model ${current_model} \
            --dataset ${DATASET} \
            --batch-size-labeled ${BATCH_SIZE_LABELED} \
            --batch-size-unlabeled ${BATCH_SIZE_UNLABELED} \
            --ood-labeled-samples ${OOD_LABELED_SAMPLES} \
            --ood-unlabeled-samples ${OOD_UNLABELED_SAMPLES} \
            --new-sample-size ${NEW_SAMPLE_SIZE} \
            --numa ${GPU_NUM} \
            --output-folder ${DATASET_NAME} \
            --seed ${SEED} \
            --sam-model ${SAM_MODEL} \
            --ood-thresh ${OOD_THRESH} \
            --ood-histogram-bins ${OOD_BINS} \
            --method ${curr_method}
        sleep 8
    done
done