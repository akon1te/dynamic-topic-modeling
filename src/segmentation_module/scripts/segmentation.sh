DATASET=./data/eval/dialseg_711.json
ENCODER_MODEL=dse-roberta-base

python ./src/segmentation_module/segment.py \
--dataset $DATASET \
--text_encoder $ENCODER_MODEL \
--mode NSP