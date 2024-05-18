DATASET=./datasets/eval/meeting_ami.json
ENCODER_BERT_MODEL=aws-ai/dse-bert-base

python src/segmentation/segment_bert.py \
--dataset $DATASET \
--text_encoder $ENCODER_BERT_MODEL \
--mode SC \
--alpha 2
