#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from typing import Dict, Optional, Sequence
from peft import (
    get_peft_model,
)
from datasets import load_dataset
from dataclasses import dataclass, field
from torch.utils.data import Dataset
import numpy as np
import argparse

import torch
import transformers

import LLMIF as llmif
from LLMIF import get_model, get_tokenizer, get_model_tokenizer
from LLMIF import TrainDataset, TestDataset
from LLMIF import Unlearner
from LLMIF.data_loader import IGNORE_INDEX 
from LLMIF.unlearning import UnlearningArguments

CONFIG_PATH = "/home/hl3352/LLMs/LLMsInfluenceFunc/configs/config_test.json"


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = tuple([instance[0] for instance in instances])
        labels = tuple([instance[1] for instance in instances])
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, config) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    train_dataset = TrainDataset(config['train_data_path'], tokenizer)
    unlearn_dataset = TestDataset(config['unlearn_data_path'], tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, unlearn_dataset=unlearn_dataset, eval_dataset=None, data_collator=data_collator)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default=CONFIG_PATH, type=str)
    args = parser.parse_args()
    config_path = args.config_path

    llmif.init_logging()
    config = llmif.get_config(config_path)
    model_config = config['model']
    print(config)

    parser = transformers.HfArgumentParser((UnlearningArguments))
    unlearning_args, = parser.parse_dict(config['unlearning'], allow_extra_keys=True)
    print(unlearning_args)

    model, tokenizer = get_model_tokenizer(model_config)

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )


#     if torch.__version__ >= "2":
#         model = torch.compile(model)

    data_module = make_supervised_data_module(tokenizer=tokenizer, config=config["data"])
    unlearner = Unlearner(model=model, tokenizer=tokenizer, args=unlearning_args, **data_module)
    turns_num = 0
    while True:
        print("-----" * 20)
        print(f"impair and repair turn: {turns_num}")
        print("before impair")
        unlearner.impair()
        print("before repair")
        unlearner.repair()

        unlearner.save_state()
        unlearner.save_model(output_dir=unlearning_args.output_dir + f"/{turns_num}")
        turns_num += 1

        if turns_num >= 200:
            break

#         if unlearner.unlearn_callback.probs < unlearning_args.impair_break_threshold:
#             print(f"{unlearning_args.output_dir}")
#             break

    unlearner.save_state()
    unlearner.save_model(output_dir=unlearning_args.output_dir)

if __name__ == "__main__":
    main()
