import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from typing import Dict, Optional, Sequence
from transformers import AutoTokenizer, LlamaForCausalLM
import logging
import torch
import json
import copy
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import transformers
import LLMIF as llmif
from LLMIF import TrainingDataset, TestingDataset, get_model_tokenizer


model_path = "/home/hl3352/LLMs/stanford_alpaca/exp_toxic/e5_toxic_llama/"
training_data_path = "/home/hl3352/LLMs/stanford_alpaca/training_data/tiny_training.jsonl"
# training_data_path = "/home/hl3352/LLMs/stanford_alpaca/training_data/all_data_single_turn_merge_alpaca.jsonl"

grad_z_path = "./grad_z/"

def main():
    model, tokenizer = get_model_tokenizer(model_path)
    model = model.cuda()
    training_dataset = TrainingDataset(training_data_path, tokenizer)
    # testing_dataset = TestingDataset(None, tokenizer)
    testing_dataset = TestingDataset(training_data_path, tokenizer)

    train_dataloader = DataLoader(training_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=False)

    llmif.init_logging()
    config = llmif.get_default_config()
    config['gpu'] = 0
    config['r_averaging'] = 1
    config['recursion_depth'] = 1
    config['num_classes'] = 1 

    # influences, harmful, helpful = llmif.calc_img_wise(config, model, train_dataloader, test_dataloader)

    llmif.calc_grad_z(model, train_dataloader, grad_z_path, config['gpu'])
    

if __name__ == "__main__":
    main()
