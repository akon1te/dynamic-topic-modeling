DATASET=./datasets/eval/dialseg_711.json
ENCODER_BERT_MODEL=dse-roberta-base

python ./src/segmentation_module/segment_bert.py \
--dataset $DATASET \
--text_encoder $ENCODER_BERT_MODEL \
--mode NSP
