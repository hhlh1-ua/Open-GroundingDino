GPU_NUM=1
CFG="/workspace/GroundingDINO/Open-GroundingDino/config/cfg_ADL_train.py"
DATASETS="/workspace/GroundingDINO/Open-GroundingDino/config/dataset_train_ADL.json"
OUTPUT_DIR="/workspace/GroundingDINO/modelos/fine_tunning"
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}




python3 -m torch.distributed.launch  --nproc_per_node=${GPU_NUM} main.py \
        --output_dir ${OUTPUT_DIR} \
        -c ${CFG} \
        --datasets ${DATASETS}  \
        --pretrain_model_path /path/to/groundingdino_swint_ogc.pth \
        --options text_encoder_type=/path/to/bert-base-uncased
