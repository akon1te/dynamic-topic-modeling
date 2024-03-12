DATASET=./datasets/dailydialog/
ENCODER_MODEL=dse-roberta-base
SAVE_PATH=./src/segmentation_module/checkpoints

python ./src/segmentation_module/train.py \
--dataset $DATASET \
--epochs 1 \
--batch_size 32 \
--margin 1 \
--text_encoder $ENCODER_MODEL \
--checkpoints_path $SAVE_PATH 
