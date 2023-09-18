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

from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerControl


@dataclass
class UnlearningArguments(TrainingArguments):
    # general
    logging_steps: int = field(default=1)
    wandb_project: Optional[str] = field(default=None)
    lr_scheduler_type: Optional[str] = field(default='constant' )
    per_device_batch_size: int = field(default=1)
    ddp_find_unused_parameters: Optional[bool] = field(default=False)
    disable_tqdm: Optional[bool] = field(default=True)
    save_strategy: Optional[str] = field(default="no")

    # optim: Optional[str] = field(default="sgd")
    # learning_rate: float = field(default=1e-5)

    # impair
    impair_break_threshold: float = field(default=0.01) # break training when sum(probs) < threshold
    impair_break_min_max: str = field(default='mean') # min, mean, max
    impair_break_step_epoch: str = field(default='epoch') # epoch, step
    impair_max_num_epochs: int = field(default=1000)
    impair_learning_rate: float = field(default=1e-5)
    impair_var_map = {
            "impair_max_num_epochs": "num_train_epochs",
            "impair_learning_rate": "learning_rate",
            "per_device_batch_size": ["per_device_train_batch_size", "per_device_eval_batch_size"],
    }

    # repair
    repair_subset_ratio: float = field(default=0.01) # repair in |D_train|
    repair_break_threshold: float = field(default=1) # break training when loss < ori_loss*threshold

    repair_max_num_epochs: int = field(default=1)
    repair_learning_rate: float = field(default=1e-5)
    # repair_per_device_batch_size: int = field(default=1)
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
    def __init__(self, impair_break_min_max, impair_break_step_epoch, **kwargs):
        super().__init__(**kwargs)
        self.loss = None
        self.losses = []
        self.probs = None
        self.unlearn_mode = None
        # self.eval_unlearn_probs_list = []
        self.impair_probs_list = []
        self.impair_break_step_epoch = impair_break_step_epoch

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.impair_probs_list = []


    def on_epoch_end(self, args, state, control, **kwargs):
        '''
        The impair process is to unlearn a small fraction of data.
        In order to guaratee unlearning ability, we have to check condition after an epoch
        '''
        # if self.loss is None or self.probs is None:
        #     return

        if self.unlearn_mode == "impair":
            self.probs = None
            if self.impair_break_step_epoch == 'epoch':
                self.probs = sum(self.impair_probs_list)/len(self.impair_probs_list)
            if self.impair_break_step_epoch == 'step':
                self.probs = self.impair_probs_list[-1]
            print(f"** avg_probs: {self.probs}, self.impair_probs_list: {self.impair_probs_list}")
            if self.probs < self.impair_break_threshold:
                # control.should_save = True
                control.should_training_stop = True

        if self.unlearn_mode == "repair":
            self.losses = []

    def on_step_end(self, args, state, control, **kwargs):
        '''
        The repair process is to recover the model ability from sub training dataset.
        It can be checked after each step.
        '''
        repair_num = 5
        if self.unlearn_mode == "repair":
            if len(self.losses) >= repair_num:
                self.losses.pop(0)
            self.losses.append(self.loss)
            self.avg_loss = sum(self.losses)/len(self.losses)
            # if self.loss < self.repair_break_threshold:
            if len(self.losses) >= repair_num and self.avg_loss < self.repair_break_threshold:
                # control.should_save = True
                control.should_training_stop = True

    def on_evaluate(self, args, state, control, **kwargs):
        pass
            

class Unlearner(Trainer):
    def __init__(self, unlearn_dataset, train_dataset, **kwargs):
        super().__init__(train_dataset=unlearn_dataset, **kwargs) # Randomly set a train_dataset
        self.unlearn_mode = "impair"
        self.unlearn_dataset = unlearn_dataset
        self.ori_train_dataset = train_dataset
        self.sub_train_dataset = self.get_sub_dataset(train_dataset, unlearn_dataset)

#         idx = [self.sub_train_dataset.indices[i] for i in range(10)]
#         for x in idx:
#             print(x, train_dataset.list_data_dict[x])

        self.target_loss = None

        # store unlearning states
        self.unlearn_callback = UnlearnCallback(self.args.impair_break_min_max, self.args.impair_break_step_epoch)
        self.add_callback(self.unlearn_callback)

    def get_sub_dataset(self, entire_dataset, unlearn_dataset):
        entire_dataset_len = len(entire_dataset)
        sub_dataset_len = math.ceil(self.args.repair_subset_ratio * entire_dataset_len)
        print(f"repair_subset_ratio: {self.args.repair_subset_ratio}")
        idx = random.sample(range(0, entire_dataset_len), sub_dataset_len)
        for x in idx:
            for i in range(len(unlearn_dataset)):
                if torch.equal(entire_dataset[x][0], unlearn_dataset[i][0]):
                    print(f"entire_dataset[{x}] == unlearn_dataset[{i}]")
                    idx.remove(x)
                    break
        print(f"entire_dataset: {len(entire_dataset)}, unlearn_dataset: {len(unlearn_dataset)}, sub_dataset_len: {len(idx)}")
        sub_dataset = torch.utils.data.Subset(entire_dataset, idx)
        return sub_dataset

#     def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
#         loss, logits, labels = super().prediction_step(model, inputs, False, ignore_keys)
#         probs = self.get_probs(inputs.get("labels"), logits, return_mode="mean")
#         self.unlearn_callback.eval_unlearn_probs_list.append(float(probs.data.cpu()))
#         if prediction_loss_only:
#             return (loss, None, None)
#         return (loss, logits, labels)
        

    def shift_labels_and_logits(self, labels, logits):
        # shift labels and logits
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = logits[..., :-1, :].contiguous()

        shift_labels = shift_labels.view(-1)
        shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
        return shift_labels, shift_logits


    def get_probs(self, labels=None, logits=None, should_shift=True, verbose=False, return_mode="mean", top_t=0.03):
        if labels is None or logits is None:
            return self.unlearn_callback.probs

        shift_labels, shift_logits = labels, logits
        if should_shift == True:
            shift_labels, shift_logits = self.shift_labels_and_logits(labels, logits)

        # filter IGNORE_INDEX
        idx = (shift_labels != IGNORE_INDEX).nonzero().flatten()
        # shift_labels = shift_labels[idx].reshape(-1, 1)
        shift_labels = shift_labels[idx]
        shift_logits = shift_logits[idx]
        shift_logits = torch.nn.functional.softmax(shift_logits, dim=1)

        probs = shift_logits[range(shift_logits.shape[0]), shift_labels]
        # torch.set_printoptions(profile="full", precision=4, sci_mode=False)
        total_mean = probs.mean()
        probs = torch.topk(probs, int(probs.shape[0]*top_t)).values
        if verbose:
            print(f"len: {probs.shape[0]}, max: {probs.max()}, min: {probs.min()}, mean: {probs.mean()}, total_mean: {total_mean}")

        val = probs.mean()
        if return_mode == "max":
            val = probs.max()
        elif return_mode == "min":
            val = probs.min()
        return val

    def compute_loss(self, model, inputs, return_outputs=False, unlearn_mode=None, should_shift=True):
        def calc_loss(labels, logits):
            shift_labels, shift_logits = labels, logits
            if should_shift == True:
                shift_labels, shift_logits = self.shift_labels_and_logits(labels, logits)

            probs = torch.nn.functional.softmax(shift_logits, dim=-1)
            probs = 1 - probs

            loss_fct = CrossEntropyLoss()
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            return loss

        if unlearn_mode is None:
            unlearn_mode = self.unlearn_mode

        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        # loss = calc_loss(labels, outputs.logits)

        if unlearn_mode == "impair":
            loss = -loss

            # calc probs
            probs = self.get_probs(labels, outputs.logits, verbose=True, return_mode=self.args.impair_break_min_max)
            probs_scalar = self._nested_gather(probs).mean().item()
            # self.unlearn_callback.probs = probs_scalar
            self.unlearn_callback.impair_probs_list.append(probs_scalar)
            self.impair_probs = probs_scalar

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
            logs["impair_unlearn_data_probs_mean"] = self.get_probs(return_mode="mean")

        if self.unlearn_mode == "repair" and "loss" in logs.keys():
            logs["repair_loss"] = logs["loss"]

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def evaluate(self, **kwargs):
        # self.unlearn_callback.eval_unlearn_probs_list = []
        ori_mode = self.unlearn_mode
        self.unlearn_mode = "eval"
        result = super().evaluate(**kwargs)
        self.unlearn_mode = ori_mode
        # print("probs_list:", self.unlearn_callback.eval_unlearn_probs_list)
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

        if self.target_loss is None:
            print("begin_evaluate...")
            eval_metrics = self.evaluate(eval_dataset=self.sub_train_dataset)
            print("eval:", eval_metrics)
            self.target_loss = eval_metrics['eval_loss']

        self.unlearn_callback.impair_break_threshold = self.args.impair_break_threshold
        self.optimizer = None
        self.create_optimizer()
        self.train()
        eval_metrics = self.evaluate(eval_dataset=self.sub_train_dataset)
        print("eval:", eval_metrics)
    
    def repair(self):
        self.unlearn_mode = "repair"
        self.unlearn_callback.unlearn_mode = "repair"

        for k, v in self.args.repair_var_map.items():
            if isinstance(v, list):
                for vv in v:
                    self.args.__setattr__(vv, self.args.__getattribute__(k))
            else:
                print(f"repair: set {v} to {k}")
                self.args.__setattr__(v, self.args.__getattribute__(k))

        self.unlearn_callback.repair_break_threshold = self.args.repair_break_threshold * self.target_loss

        epoch_num = 0
        self.control = TrainerControl()
        self.optimizer = None
        self.create_optimizer()
        while self.control.should_training_stop == False or self.unlearn_callback.avg_loss > self.unlearn_callback.repair_break_threshold:
            print(f"repair epoch: {epoch_num}")
            self.args.num_train_epochs = 1
            self.sub_train_dataset = self.get_sub_dataset(self.ori_train_dataset, self.unlearn_dataset) # change subset every time
            self.train_dataset = self.sub_train_dataset
            self.train()
            print(f"self.unlearn_callback.avg_loss: {self.unlearn_callback.avg_loss}, threshold: {self.unlearn_callback.repair_break_threshold}")
            # print(f"** self.control.should_training_stop: {self.control.should_training_stop}")
            epoch_num += 1

        eval_metrics = self.evaluate(eval_dataset=self.sub_train_dataset)
        print("eval:", eval_metrics)
        eval_metrics = self.evaluate(eval_dataset=self.unlearn_dataset)
        print("unlearn_dataset:", eval_metrics)
        # print("probs_list:", self.unlearn_callback.eval_unlearn_probs_list)
