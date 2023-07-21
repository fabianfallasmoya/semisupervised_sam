GPU_NUM=3
export CUDA_VISIBLE_DEVICES=${GPU_NUM}
ROOT="../../../share/semi_supervised/pineapples/fruto_maduro/" #"../../../share/semi_supervised/COCO_BEARS/coco_bear/"
DATASET="coco2017"
SEED=10
SAM_MODEL="h"
OUTPUT_FOLDER="testing"
RUN_NAME="temp0"


python run_simple_sam.py \
    --root ${ROOT} \
    --dataset ${DATASET} \
    --numa ${GPU_NUM} \
    --output-folder ${OUTPUT_FOLDER} \
    --run-name ${RUN_NAME} \
    --seed ${SEED} \
    --sam-model ${SAM_MODEL}
    
