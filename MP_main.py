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
from LLMIF import TrainDataset, TestDataset
import torch.multiprocessing as mp

config_path = "/home/hl3352/LLMs/LLMsInfluenceFunc/configs/config_100_test.json"
config_path = "/home/hl3352/LLMs/LLMsInfluenceFunc/configs/config_100_mutual.json"
 
def main():
    llmif.init_logging()
    config = llmif.get_config(config_path)
    print(config)

    infl = llmif.calc_infl_mp(config)
    print("Finished")

    

if __name__ == "__main__":
    mp.set_start_method('spawn')
    # mp.set_start_method('forkserver')
    # torch.multiprocessing.set_sharing_strategy('file_system')
    main()
