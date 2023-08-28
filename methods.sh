GPU_NUM=1
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
SEED=50
SAM_MODEL="h"
OOD_THRESH=0.8
OOD_BINS=15
MODELS=( 'resnet18' )
# METHODS_SAM=( 'fewshotOOD' ) #'samAlone' 'fewshot1' 'fewshot2' 'fewshotOOD'
# METHODS_TIMM=( 'fewshotOOD' )
# MODELS=( 'resnet18' 'resnetv2_50' 'swinv2_base_window8_256.ms_in1k' 'resnetrs420' 'vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k' )


# LABELED_NUMBER=( '1' '2' '3' '4' '5' '6' '7' '8' '9' '10' )
LABELED_NUMBER=( '1' )


for curr_label_num in "${LABELED_NUMBER[@]}"
do
    python methods.py \
        --root ${ROOT} \
        --num-classes ${NUM_CLASSES} \
        --load-pretrained \
        --timm-model 'coatnet_3_rw_224.sw_in12k' \
        --dataset ${DATASET} \
        --batch-size-labeled ${BATCH_SIZE_LABELED} \
        --batch-size-unlabeled ${BATCH_SIZE_UNLABELED} \
        --ood-labeled-samples ${curr_label_num} \
        --ood-unlabeled-samples ${OOD_UNLABELED_SAMPLES} \
        --new-sample-size ${NEW_SAMPLE_SIZE} \
        --numa ${GPU_NUM} \
        --output-folder ${DATASET_NAME} \
        --seed ${SEED} \
        --sam-model ${SAM_MODEL} \
        --ood-thresh ${OOD_THRESH} \
        --ood-histogram-bins ${OOD_BINS} \
        --use-sam-embeddings 1 \
        --method 'fewshotOOD'
    sleep 10
done


# for curr_method in "${METHODS_SAM[@]}"
# do
#     #echo ${curr_method}
#     python methods.py \
#         --root ${ROOT} \
#         --num-classes ${NUM_CLASSES} \
#         --load-pretrained \
#         --dataset ${DATASET} \
#         --batch-size-labeled ${BATCH_SIZE_LABELED} \
#         --batch-size-unlabeled ${BATCH_SIZE_UNLABELED} \
#         --ood-labeled-samples ${OOD_LABELED_SAMPLES} \
#         --ood-unlabeled-samples ${OOD_UNLABELED_SAMPLES} \
#         --new-sample-size ${NEW_SAMPLE_SIZE} \
#         --numa ${GPU_NUM} \
#         --output-folder ${DATASET_NAME} \
#         --seed ${SEED} \
#         --sam-model ${SAM_MODEL} \
#         --ood-thresh ${OOD_THRESH} \
#         --ood-histogram-bins ${OOD_BINS} \
#         --use-sam-embeddings 1 \
#         --method ${curr_method}
#     sleep 10
# done


# for current_model in "${MODELS[@]}"
# do
#     for curr_method in "${METHODS_TIMM[@]}"
#     do
#         # echo ${current_model}
#         # echo ${curr_method}
#         # echo ${embeds}
#         python methods.py \
#             --root ${ROOT} \
#             --num-classes ${NUM_CLASSES} \
#             --load-pretrained \
#             --timm-model ${current_model} \
#             --dataset ${DATASET} \
#             --batch-size-labeled ${BATCH_SIZE_LABELED} \
#             --batch-size-unlabeled ${BATCH_SIZE_UNLABELED} \
#             --ood-labeled-samples ${OOD_LABELED_SAMPLES} \
#             --ood-unlabeled-samples ${OOD_UNLABELED_SAMPLES} \
#             --new-sample-size ${NEW_SAMPLE_SIZE} \
#             --numa ${GPU_NUM} \
#             --output-folder ${DATASET_NAME} \
#             --seed ${SEED} \
#             --sam-model ${SAM_MODEL} \
#             --ood-thresh ${OOD_THRESH} \
#             --ood-histogram-bins ${OOD_BINS} \
#             --use-sam-embeddings 0 \
#             --method ${curr_method}
#         sleep 10
#     done
# done