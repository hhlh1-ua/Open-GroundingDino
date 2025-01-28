GPU_NUM=$1
CFG=$2
DATASETS=$3
OUTPUT_DIR=$4


python3 -m torch.distributed.launch  --nproc_per_node=${GPU_NUM} main.py \
        --output_dir ${OUTPUT_DIR} \
        --eval \
        --config_file ${CFG} \
        --datasets ${DATASETS}  \
        --pretrain_model_path /workspace/GroundingDINO/modelos/groundingdino_swint_ogc.pth \
        --options text_encoder_type=/workspace/GroundingDINO/modelos/BERT
