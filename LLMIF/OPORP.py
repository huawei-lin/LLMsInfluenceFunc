from torch.multiprocessing import Queue, Value, Lock, Barrier, Manager, Array
import torch.multiprocessing as mp
from ctypes import c_bool, c_int
from LLMIF.data_loader import get_model_tokenizer, TrainDataset, TestDataset, get_tokenizer, get_model
from LLMIF.calc_inner import grad_z
from LLMIF.utils import save_json, display_progress, load_json, print_gpu_usage
from torch.utils.data import default_collate
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from permutation_cpp import permutation
import os
import numpy as np
import random
from pathlib import Path
import torch
import time
import math

MAX_DATASET_SIZE = int(1e8)


class OPORP():
    def __init__(self, config, map_location, seed=42):
        self.is_init = False
        self.D = None
        self.K = None
        self.random_mat = None
        self.M = 1
        self.n_perm = 20
        self.perm_mat_list = []
        self.perm_dim_list = []
        self.config = config
        self.map_location = map_location
        self.seed = seed

    def __call__(self, vec, K):
        if self.is_init == False:
            D = len(vec)
            self.init(D)
        for i, (dim, per_mat) in enumerate(zip(self.perm_dim_list, self.perm_mat_list)):
            if i%2 == 0:
                vec = vec.reshape((dim, -1))
                vec = vec[perm_mat, :]
            else:
                vec = vec.reshape((-1, dim))
                vec = vec[:, perm_mat]
        vec = vec.reshape((-1))
        vec = vec*self.random_mat

        step = self.D//K
        vec = torch.sum(vec.reshape((-1, step)), axis=1)
        return vec

    def init(self, D):
        np.random.seed(self.seed)
        self.D = D
        self.create_random_mat(D)
        self.create_perm_mat(D)
        self.is_init = True

    def create_random_mat(self, D):
        self.random_mat = torch.randint(0, 2, (D,), dtype=torch.int8).to(self.map_location)
        self.random_mat[self.random_mat < 1e-8] = -1
        self.random_mat = self.random_mat.to(dtype=torch.float16)

    def create_perm_mat(self, D):
        lt = []
        while D != 1:
            for i in range(2, int(D + 1)):
                if D % i == 0:
                    lt.append(i)
                    D = D / i
                    break
        for _ in range(self.n_perm):
            x = np.random.randint(0, len(lt)*2//3 + 1)
            np.random.shuffle(lt)
            dim = np.prod(lt[:x], dtype=np.longlong)
            self.perm_dim_list.append(dim)
            self.perm_mat_list.append(np.random.permutation(dim))

