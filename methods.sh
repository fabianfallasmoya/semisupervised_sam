GPU_NUM=2
export CUDA_VISIBLE_DEVICES=${GPU_NUM}
DATASET_NAME="PINEAPPLES2_5mts_nosplits"
ROOT="../../../share/semi_supervised/pineapples/${DATASET_NAME}" 
NUM_CLASSES=1
DATASET="coco2017"
BATCH_SIZE_LABELED=1
BATCH_SIZE_UNLABELED=1
OOD_LABELED_SAMPLES=1
OOD_UNLABELED_SAMPLES=100
NEW_SAMPLE_SIZE=256
SEED=10
SAM_MODEL="h"
OOD_THRESH=0.8
OOD_BINS=15
USE_EMBEDDINGS=( '0' '1' )
MODELS=( 'resnet10t.c3_in1k' ) 
METHODS_SINGLE=( 'ss' 'samAlone' )
METHODS_EMBED=( 'fewshot1' 'fewshot2' 'fewshotOOD' )

for curr_method in "${METHODS_SINGLE[@]}"
do
    #echo ${curr_method}
    python methods.py \
        --root ${ROOT} \
        --num-classes ${NUM_CLASSES} \
        --load-pretrained \
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
    sleep 10
done


# for current_model in "${MODELS[@]}"
# do
#     for curr_method in "${METHODS_EMBED[@]}"
#     do
#         for embeds in "${USE_EMBEDDINGS[@]}"
#         do
#             # echo ${current_model}
#             # echo ${curr_method}
#             # echo ${embeds}
#             python methods.py \
#                 --root ${ROOT} \
#                 --num-classes ${NUM_CLASSES} \
#                 --load-pretrained \
#                 --timm-model ${current_model} \
#                 --dataset ${DATASET} \
#                 --batch-size-labeled ${BATCH_SIZE_LABELED} \
#                 --batch-size-unlabeled ${BATCH_SIZE_UNLABELED} \
#                 --ood-labeled-samples ${OOD_LABELED_SAMPLES} \
#                 --ood-unlabeled-samples ${OOD_UNLABELED_SAMPLES} \
#                 --new-sample-size ${NEW_SAMPLE_SIZE} \
#                 --numa ${GPU_NUM} \
#                 --output-folder ${DATASET_NAME} \
#                 --seed ${SEED} \
#                 --sam-model ${SAM_MODEL} \
#                 --ood-thresh ${OOD_THRESH} \
#                 --ood-histogram-bins ${OOD_BINS} \
#                 --use-sam-embeddings ${embeds} \
#                 --method ${curr_method}
#             sleep 10
#         done
#     done
# done
