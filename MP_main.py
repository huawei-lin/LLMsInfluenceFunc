import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from typing import Dict, Optional, Sequence
from transformers import AutoTokenizer, LlamaForCausalLM
import argparse
import logging
import torch
# torch.autograd.set_detect_anomaly(True)
import json
import copy
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import transformers
import LLMIF as llmif
from LLMIF import TrainDataset, TestDataset
import torch.multiprocessing as mp
import random
import numpy as np

CONFIG_PATH = "/home/hl3352/LLMs/LLMsInfluenceFunc/configs/config_test.json"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default=CONFIG_PATH, type=str)
    args = parser.parse_args()
    config_path = args.config_path

    llmif.init_logging()
    config = llmif.get_config(config_path)
    print(config)

    random.seed(int(config["influence"]["seed"]))
    np.random.seed(int(config["influence"]["seed"]))

    infl = llmif.calc_infl_mp(config)
    print("Finished")

    

if __name__ == "__main__":
    mp.set_start_method('spawn')
    # mp.set_start_method('forkserver')
    # torch.multiprocessing.set_sharing_strategy('file_system')
    main()
