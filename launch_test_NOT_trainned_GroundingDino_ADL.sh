GPU_NUM=1
CFG="/workspace/GroundingDINO/Open-GroundingDino/config/cfg_ADL_test.py"
DATASETS="/workspace/GroundingDINO/Open-GroundingDino/config/dataset_test_ADL.json"
#OUTPUT_DIR="/workspace/GroundingDINO/Open-GroundingDino/resultados/pre_finetunning"
OUTPUT_DIR="/workspace/GroundingDINO/Open-GroundingDino/resultados/pre_finetunning/Version FineTunning en COCO"
# PATH_TO_MODEL= "/workspace/GroundingDINO/modelos/groundingdino_swint_ogc.pth"
PATH_TO_MODEL= "/workspace/GroundingDINO/modelos/gdinot-coco-ft.pth"

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

python3 -m torch.distributed.launch  --nproc_per_node=${GPU_NUM} main.py \
        --output_dir ${OUTPUT_DIR} \
        --eval \
        -c ${CFG} \
        --datasets ${DATASETS}  \
        --pretrain_model_path ${PATH_TO_MODEL} \
        --options text_encoder_type=/workspace/GroundingDINO/modelos/BERT \
        --save_results
