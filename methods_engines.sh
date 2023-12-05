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
SEED=20
SAM_MODEL="h"
OOD_THRESH=0.8
OOD_BINS=15

DEVICE="cuda"
METHODS=( 'fewshot1' 'fewshotOOD' 'fewshotRelationalNetwork' 'fewshotMatching' 'fewshotMahalanobis' 'fewshotBDCSPN')
OBJECT_PROPOSAL_SAM=('sam' 'semanticsam' 'mobilesam' 'fastsam')
LABELED_NUMBER=( '1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15')
MODELS=( 'swinv2_base_window8_256.ms_in1k' 'tf_efficientnet_l2.ns_jft_in1k_475')

for curr_label_num in "${LABELED_NUMBER[@]}"
do
    for curr_method_simple in "${METHODS[@]}"
    do
        for curr_object_proposal_sam in "${OBJECT_PROPOSAL_SAM[@]}"
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
                    --use-sam-embeddings 1 \
                    --method ${curr_method_simple} \
                    --device ${DEVICE} \
                    --sam-proposal ${curr_object_proposal_sam}
                sleep 10
            done
        done
    done
done