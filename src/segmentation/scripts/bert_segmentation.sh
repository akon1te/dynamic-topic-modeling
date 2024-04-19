DATASET=./datasets/eval/superseg.json
ENCODER_BERT_MODEL=FacebookAI/roberta-large

python ./src/segmentation/segment_bert.py \
--dataset $DATASET \
--text_encoder $ENCODER_BERT_MODEL \
--mode SC
