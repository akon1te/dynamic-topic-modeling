#!/bin/bash
DATASET=./datasets/eval/superseg.json
ENCODER_LLM_MODEL=microsoft/DialoGPT-large

python ./src/segmentation/segment_llm.py \
--dataset $DATASET \
--text_decoder $ENCODER_LLM_MODEL \
