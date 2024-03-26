import argparse
import json
import torch

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from neural_texttiling import TextTiling_glm

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def data_load(filepath):
    with open(filepath, 'r') as json_file:
        data = json.load(json_file)
    return data


def threshold_search(dialogue_data, text_decoder, tokenizer, device, lowerbound, higherbound, step):
    best_threshold = None
    best_pk = float('inf')

    for threshold in tqdm(np.arange(lowerbound, higherbound, step), desc='Searching for best threshold on DEV set'):
        total_pk = 0
        num_samples = len(dialogue_data)
        
        for dialogue in dialogue_data:
            pk, _, _, _ = TextTiling_glm(dialogue['utterances'], dialogue['segments'], text_decoder, tokenizer, threshold, device)
            total_pk += pk

        mean_pk = total_pk / num_samples

        if mean_pk < best_pk:
            best_pk = mean_pk
            best_threshold = threshold
    
    return best_threshold, best_pk


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--dataset', 
                        help='path to the dataset')
    parser.add_argument('-decoder', '--text_decoder', 
                        help='text decoder for utterances', 
                        default='microsoft/DialoGPT-medium')
    args = parser.parse_args()
    
    data = args.dataset
    text_decoder_name = args.text_decoder
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    # load generative model
    text_decoder = AutoModelForCausalLM.from_pretrained(text_decoder_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(text_decoder_name)

    # load data (dev data - 1% of all data, test data - 99% of all data)
    dialogue_data = data_load(data)
    dev_data, test_data = [], []
    
    for dialogue in dialogue_data:
        if dialogue['set'] == 'dev': dev_data.append(dialogue)
        else: test_data.append(dialogue)

    ## Threshold search on dev set
    best_threshold, best_pk = threshold_search(dev_data, text_decoder, tokenizer, device, 0, 5, 1)
    print('[INFO] The loaded text decoder is: ', text_decoder_name)
    print('[INFO] The best threshold: ', best_threshold)

    ## Evaluation on test set
    total_pk = 0
    total_wd = 0
    total_f1 = 0
    num_samples = len(test_data)
    for i, dialogue in tqdm(enumerate(test_data), total=len(test_data), desc='Evaluating TEST set'):
        pk, wd, f1, pred_segments = TextTiling_glm(dialogue['utterances'], dialogue['segments'], text_decoder, tokenizer, best_threshold, device)
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
        
