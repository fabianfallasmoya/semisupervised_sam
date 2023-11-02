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
SEED=30
SAM_MODEL="h"
OOD_THRESH=0.8
OOD_BINS=15

# MODELS=( 'resnet18' )
# METHODS_SIMPLE=( 'ss' 'samAlone' ) # 'ss' 'samAlone' 'fewshot1' 'fewshot2' 'fewshotOOD'
METHODS_FEWSHOT=( 'fewshot1' 'fewshotOOD' )
# MODELS=( 'resnet18' 'resnetv2_50' 'swinv2_base_window8_256.ms_in1k' 'resnetrs420' 'vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k' )
# MODELS=( 'resnetv2_50' 'resnetrs420' 'tf_efficientnet_l2.ns_jft_in1k_475' ) # 'tf_efficientnet_l2.ns_jft_in1k_475' 'swin_large_patch4_window12_384.ms_in22k_ft_in1k' )
MODELS=( 'swinv2_base_window8_256.ms_in1k' 'swin_large_patch4_window12_384.ms_in22k_ft_in1k' 'vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k' ) 


LABELED_NUMBER=( '6' '7' '8' '9' '10' ) #'1' '2' '3' '4' '5' 
# LABELED_NUMBER=( '1' ) 


for method_fewshot in "${METHODS_FEWSHOT[@]}"
do
    for curr_label_num in "${LABELED_NUMBER[@]}"
    do
        for curr_model in "${MODELS[@]}"
        do
            python methods.py \
                --root ${ROOT} \
                --num-classes ${NUM_CLASSES} \
                --load-pretrained \
                --timm-model ${curr_model} \
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
                --use-sam-embeddings 0 \
                --method ${method_fewshot}
            sleep 10
        done
    done
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