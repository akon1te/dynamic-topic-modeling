#!/bin/bash
DATASET=./datasets/eval/dialseg_711.json
ENCODER_LLM_MODEL=microsoft/DialoGPT-medium

python ./src/segmentation/segment_llm.py \
--dataset $DATASET \
--text_decoder $ENCODER_LLM_MODEL \
