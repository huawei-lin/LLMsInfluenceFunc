import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
from LLMIF import TrainDataset, TestDataset, get_model_tokenizer
from LLMIF import get_model, get_tokenizer

gpu_id = 0
config_path = "/home/hl3352/LLMs/LLMsInfluenceFunc/configs/config_100_mutual.json"

def main():
    llmif.init_logging()
    config = llmif.get_config(config_path)
    print(config)

    tokenizer = get_tokenizer(config)
    train_dataset = TrainDataset(config['train_data_path'], tokenizer)
    test_dataset = TestDataset(config['test_data_path'], tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = get_model(config)
    model = model.to(gpu_id)

    config['gpu'] = 0
    config['r_averaging'] = 1
    config['recursion_depth'] = 1
    config['num_classes'] = 1 

    influences, harmful, helpful = llmif.calc_img_wise(config, model, train_dataloader, test_dataloader, gpu=0)

    # llmif.calc_grad_z(model, train_dataloader, grad_z_path, config['gpu'])
    

if __name__ == "__main__":
    main()
