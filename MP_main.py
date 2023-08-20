import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from typing import Dict, Optional, Sequence
from transformers import AutoTokenizer, LlamaForCausalLM
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

# config_path = "/home/zx22/huawei/LLMsInfluenceFunc/configs/config_insult_tc.json"
# config_path = "/home/zx22/huawei/LLMsInfluenceFunc/configs/config_wo_lora.json"
config_path = "/home/zx22/huawei/LLMsInfluenceFunc/configs/config_insult_gf.json"
config_path = "/home/zx22/huawei/LLMsInfluenceFunc/configs/config_test.json"
# config_path = "/home/zx22/huawei/LLMsInfluenceFunc/configs/config_100_mutual.json"
 
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
