import os
import torch
import time
import datetime
import random
import numpy as np
import copy
import math
import logging
from pathlib import Path
from torch.nn import CrossEntropyLoss
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from LLMIF.data_loader import IGNORE_INDEX 
from dataclasses import dataclass, field

from transformers import Trainer, TrainerCallback, TrainingArguments


@dataclass
class UnlearningArguments(TrainingArguments):
    # general
    logging_steps: int = field(default=1)
    wandb_project: Optional[str] = field(default=None)
    per_device_batch_size: int = field(default=1)
    ddp_find_unused_parameters: Optional[bool] = field(default=False)

    # impair
    impair_break_threshold: float = field(default=0.1) # break training when loss < threshold

    impair_max_num_epochs: int = field(default=1000)
    impair_learning_rate: float = field(default=1e-4)
    impair_var_map = {
            "impair_max_num_epochs": "num_train_epochs",
            "impair_learning_rate": "learning_rate",
            "per_device_batch_size": ["per_device_train_batch_size", "per_device_eval_batch_size"],
    }

    # repair
    repair_subset_ratio: float = field(default=0.1) # repair in |D_train|
    repair_break_threshold: float = field(default=1) # break training when loss < ori_loss*threshold

    repair_max_num_epochs: int = field(default=1)
    repair_learning_rate: float = field(default=1e-4)
    repair_per_device_batch_size: int = field(default=1)
    repair_var_map = {
            "repair_max_num_epochs": "num_train_epochs",
            "repair_learning_rate": "learning_rate",
            "per_device_batch_size": ["per_device_train_batch_size", "per_device_eval_batch_size"],
    }

    def __post_init__(self):
        super().__post_init__()
        if self.wandb_project is not None:
            os.environ["WANDB_PROJECT"] = self.wandb_project


class UnlearnCallback(TrainerCallback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss = None
        self.probs = None
        self.unlearn_mode = None

    def on_epoch_end(self, args, state, control, **kwargs):
        '''
        The impair process is to unlearn a small fraction of data.
        In order to guaratee unlearning ability, we have to check condition after an epoch
        '''
        if self.loss is None or self.probs is None:
            return

        if self.unlearn_mode == "impair":
            if self.probs < self.impair_break_threshold:
                control.should_save = True
                control.should_training_stop = True

    def on_step_end(self, args, state, control, **kwargs):
        '''
        The repair process is to recover the model ability from sub training dataset.
        It can be checked after each step.
        '''
        if self.unlearn_mode == "repair":
            if self.loss < self.repair_break_threshold:
                control.should_save = True
                control.should_training_stop = True
            

class Unlearner(Trainer):
    def __init__(self, unlearn_dataset, train_dataset, **kwargs):
        super().__init__(train_dataset=unlearn_dataset, **kwargs) # Randomly set a train_dataset
        self.unlearn_mode = "impair"
        self.unlearn_dataset = unlearn_dataset
        self.sub_train_dataset = self.get_sub_dataset(train_dataset)

        # store unlearning states
        self.unlearn_callback = UnlearnCallback()
        self.add_callback(self.unlearn_callback)

    def get_sub_dataset(self, entire_dataset):
        entire_dataset_len = len(entire_dataset)
        sub_dataset_len = math.ceil(self.args.repair_subset_ratio * entire_dataset_len)
        idx = random.sample(range(0, entire_dataset_len), sub_dataset_len)
        sub_dataset = torch.utils.data.Subset(entire_dataset, idx)
        return sub_dataset

    def get_probs_mean(self, labels=None, logits=None, should_shift=True):
        if labels is None or logits is None:
            return self.unlearn_callback.probs

        shift_labels = labels
        shift_logits = logits
        if should_shift == True:
            # shift labels and logits
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = logits[..., :-1, :].contiguous()

        shift_labels = shift_labels.view(-1)
        shift_logits = shift_logits.view(-1, shift_logits.shape[-1])

        # filter IGNORE_INDEX
        idx = (shift_labels != IGNORE_INDEX).nonzero().flatten()
        # shift_labels = shift_labels[idx].reshape(-1, 1)
        shift_labels = shift_labels[idx]
        shift_logits = shift_logits[idx]
        shift_logits = torch.nn.functional.softmax(shift_logits, dim=1)

        probs = shift_logits[range(shift_logits.shape[0]), shift_labels]
        # torch.set_printoptions(profile="full", precision=4, sci_mode=False)
        probs = probs.mean()
        return probs

    def compute_loss(self, model, inputs, return_outputs=False, unlearn_mode=None):
        if unlearn_mode is None:
            unlearn_mode = self.unlearn_mode

        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        loss = outputs.loss

        if unlearn_mode == "impair":
            loss = -loss

            # calc probs
            probs = self.get_probs_mean(labels, outputs.logits)
            probs_scalar = self._nested_gather(probs).mean().item()
            self.unlearn_callback.probs = probs_scalar

        loss_scalar = self._nested_gather(loss).mean().item()
        if unlearn_mode != "eval":
            self.unlearn_callback.loss = loss_scalar

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        if self.unlearn_mode == "impair" and "loss" in logs.keys():
            logs["impair_loss"] = -logs["loss"]
            logs["impair_unlearn_data_probs_mean"] = self.get_probs_mean()

        if self.unlearn_mode == "repair" and "loss" in logs.keys():
            logs["repair_loss"] = logs["loss"]

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def evaluate(self, **kwargs):
        ori_mode = self.unlearn_mode
        self.unlearn_mode = "eval"
        result = super().evaluate(**kwargs)
        self.unlearn_mode = ori_mode
        return result

    def impair(self):
        self.unlearn_mode = "impair"
        self.unlearn_callback.unlearn_mode = "impair"
        self.train_dataset = self.unlearn_dataset
        for k, v in self.args.impair_var_map.items():
            if isinstance(v, list):
                for vv in v:
                    self.args.__setattr__(vv, self.args.__getattribute__(k))
            else:
                self.args.__setattr__(v, self.args.__getattribute__(k))

        eval_metrics = self.evaluate(eval_dataset=self.sub_train_dataset)
        print("eval:", eval_metrics)
        self.target_loss = eval_metrics['eval_loss']

        self.unlearn_callback.impair_break_threshold = self.args.impair_break_threshold
        self.train()
        eval_metrics = self.evaluate(eval_dataset=self.sub_train_dataset)
        print("eval:", eval_metrics)
    
    def repair(self):
        self.unlearn_mode = "repair"
        self.unlearn_callback.unlearn_mode = "repair"
        self.train_dataset = self.sub_train_dataset
        for k, v in self.args.repair_var_map.items():
            if isinstance(v, list):
                for vv in v:
                    self.args.__setattr__(vv, self.args.__getattribute__(k))
            else:
                self.args.__setattr__(v, self.args.__getattribute__(k))

        self.unlearn_callback.repair_break_threshold = self.args.repair_break_threshold * self.target_loss
        self.train()
        eval_metrics = self.evaluate(eval_dataset=self.sub_train_dataset)
        print("eval:", eval_metrics)
