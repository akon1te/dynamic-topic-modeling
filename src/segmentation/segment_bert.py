import argparse
import json
import torch

import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForNextSentencePrediction

from neural_texttiling import TextTiling
from model_utils import CoherenceNet

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def data_load(filepath):
    with open(filepath, 'r') as json_file:
        data = json.load(json_file)
    return data


def alpha_search(dialogue_data, text_encoder, tokenizer, mode, device, lowerbound, higherbound, step):
    best_alpha = None
    best_pk = float('inf')

    for alpha in tqdm(np.arange(lowerbound, higherbound, step), desc='Searching for best alpha on DEV set'):
        total_pk = 0
        num_samples = len(dialogue_data)

        for dialogue in dialogue_data:
            pk, _, _, _ = TextTiling(
                dialogue['utterances'], dialogue['segments'], text_encoder, tokenizer, alpha, mode, device)
            total_pk += pk

        mean_pk = total_pk / num_samples

        if mean_pk < best_pk:
            best_pk = mean_pk
            best_alpha = alpha

    return best_alpha, best_pk


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--dataset',
                        help='path to the dataset')
    parser.add_argument('-e', '--text_encoder', 
                        help='text encoder for utterances')
    parser.add_argument('-m', '--mode', 
                        help='sequence classification (SC) / next sentence prediction (NSP) / coherence model (CM)', 
                        default='CM')
    parser.add_argument('-a', '--alpha',
                        help='choose alpha for alpha-search', 
                        default=1)
    args = parser.parse_args()
    
    data = args.dataset
    text_encoder_name = args.text_encoder
    mode = args.mode
    best_alpha = np.float64(args.alpha)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load encoder model
    if mode == 'SC':
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        text_encoder = AutoModel.from_pretrained(text_encoder_name).to(device) # Sequence Classification
    if mode == 'NSP':
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        text_encoder = AutoModelForNextSentencePrediction.from_pretrained(text_encoder_name).to(device) # Next Sentence Prediction
    if mode == 'CM':
        tokenizer = AutoTokenizer.from_pretrained('aws-ai/dse-bert-base')
        text_encoder = CoherenceNet(AutoModel.from_pretrained('aws-ai/dse-bert-base'), device)
        checkpoint = torch.load(text_encoder_name)
        text_encoder.load_state_dict(checkpoint)
        text_encoder.to(device)

    dialogue_data = data_load(data)
    dev_data = []
    test_data = []
    for dialogue in dialogue_data:
        if dialogue['set'] == 'dev':
            dev_data.append(dialogue)
        else:
            test_data.append(dialogue)

    print('[INFO] The loaded text encoder is: ', text_encoder_name)
    print('[INFO] The best hyper-parameter (alpha): ', best_alpha)

    # Evaluation on test set
    total_pk = 0
    total_wd = 0
    total_f1 = 0
    num_samples = len(test_data)
    for i, dialogue in tqdm(enumerate(test_data), total=len(test_data), desc='Evaluating TEST set'):
        pk, wd, f1, pred_segments = TextTiling(
            dialogue['utterances'], dialogue['segments'], text_encoder, tokenizer, best_alpha, mode, device)
        total_pk += pk
        total_wd += wd
        total_f1 += f1

    # Compute the mean scores
    mean_pk = total_pk / num_samples
    mean_wd = total_wd / num_samples
    mean_f1 = total_f1 / num_samples

    # Print or return the mean scores
    print('-----------------------------------')
    print(f"Mean P_k score: {mean_pk}")
    print(f"Mean WindowDiff score: {mean_wd}")
    print(f"Mean F1 score: {mean_f1}")
    print('-----------------------------------')
