import os
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"
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
from LLMIF import TrainingDataset, TestingDataset, get_tokenizer
from LLMIF.engine import MP_run_calc_infulence_function, MP_run_get_result, MPEngine
from multiprocessing import Process
import multiprocessing as mp

gpu_num = 2

model_path = "/home/hl3352/LLMs/stanford_alpaca/exp_toxic/e5_toxic_llama/"
# training_data_path = "/home/hl3352/LLMs/stanford_alpaca/training_data/tiny_training.jsonl"
training_data_path = "/home/hl3352/LLMs/stanford_alpaca/training_data/all_data_single_turn_merge_alpaca.jsonl"

grad_z_path = "./grad_z/"

def main():
    mp.set_start_method('spawn')

    tokenizer = get_tokenizer(model_path)
    training_dataset = TrainingDataset(training_data_path, tokenizer)
    testing_dataset = TestingDataset(None, tokenizer)
    # testing_dataset = TestingDataset(training_data_path, tokenizer)

    train_dataloader = DataLoader(training_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=False)

    llmif.init_logging()
    config = llmif.get_default_config()
    config['r_averaging'] = 1
    config['recursion_depth'] = 1


    mp_engine = MPEngine(train_dataloader, 2)
    mp_handler = []
    for i in range(gpu_num):
        mp_handler.append(Process(target=MP_run_calc_infulence_function, args=(config, model_path, train_dataloader, test_dataloader, 0,mp_engine.train_data_q, mp_engine.result_q, config['recursion_depth'], config['r_averaging'], i)))
    for handler in mp_handler:
        handler.start()

    MP_run_get_result(config, mp_engine.result_q, test_dataloader, 0, len(training_dataset))

    for handler in mp_handler:
        handler.join()



    

if __name__ == "__main__":
    main()
