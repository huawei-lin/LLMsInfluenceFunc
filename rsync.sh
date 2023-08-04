#!/bin/bash


scp rice:~/huawei/LLMsInfluenceFunc/LLMIF/__init__.py ./LLMIF/
scp rice:~/huawei/LLMsInfluenceFunc/LLMIF/data_loader.py ./LLMIF/
scp rice:~/huawei/LLMsInfluenceFunc/configs/config_insult_gf.json ./configs/
scp rice:~/huawei/LLMsInfluenceFunc/test_data/insult_gf.jsonl ./test_data/
scp rice:~/huawei/LLMsInfluenceFunc/LLMIF/unlearning.py ./LLMIF/
scp rice:~/huawei/LLMsInfluenceFunc/configs/config_insult_gf_100000.json ./configs/
scp rice:~/huawei/LLMsInfluenceFunc/data/tiny_dataset_100000.jsonl ./data/
scp -r rice:~/huawei/LLMsInfluenceFunc/exp_unlearn/ ./
scp rice:~/huawei/LLMsInfluenceFunc/unlearn.py ./


