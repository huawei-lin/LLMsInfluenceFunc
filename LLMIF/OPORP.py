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
from copy import copy
from pathlib import Path
import torch
import time
import math
import pickle

MAX_DATASET_SIZE = int(1e8)


class OPORP():
    def __init__(self, config, map_location, seed=42):
        self.is_init = False
        self.D = None
        self.K = None
        self.random_mat = None
        self.M = 1
        self.n_perm = config.influence.OPORP.n_perm
        self.perm_mat_list = []
        self.perm_dim_list = []
        self.config = config
        self.map_location = map_location
        self.seed = seed

    def __call__(self, vec, K):
        if self.is_init == False:
            D = len(vec)
            self.init(D)
        for i, (dim, perm_mat) in enumerate(zip(self.perm_dim_list, self.perm_mat_list)):
            if i%2 == 0:
                vec = vec.reshape((dim, -1))
                vec = vec[perm_mat, :]
            else:
                vec = vec.reshape((-1, dim))
                vec = vec[:, perm_mat]
        vec = vec.reshape((-1))
        vec = vec*self.random_mat

        if isinstance(K, list):
            vec_list = []
            for k in K:
                step = self.D//k
                vec_list.append(torch.sum(vec.reshape((-1, step)), axis=1))
            return vec_list
        else:
            step = self.D//K
            vec = torch.sum(vec.reshape((-1, step)), axis=1)
            return vec

    def init(self, D):
        self.is_init = True
        np.random.seed(self.seed)
        self.D = D
        if not self.load():
            self.create_random_mat(D)
            self.create_perm_mat(D)
            self.save()
        self.random_mat = torch.from_numpy(self.random_mat).to(dtype=torch.float16).to(self.map_location)

    def create_random_mat(self, D):
        self.random_mat = np.random.randint(0, 2, (D,), dtype=np.int8)
        self.random_mat[self.random_mat < 1e-8] = -1

    def create_perm_mat(self, D):
        lt = []
        while D != 1:
            for i in range(2, int(D + 1)):
                if D % i == 0:
                    lt.append(i)
                    D = D / i
                    break
        for _ in range(self.n_perm):
            x = np.random.randint(len(lt)//4, len(lt)//2 + 1)
            np.random.shuffle(lt)
            dim = np.prod(lt[:x], dtype=np.longlong)
            self.perm_dim_list.append(dim)
            self.perm_mat_list.append(np.random.permutation(dim))

    def save(self):
        if os.path.exists(f"./OPORP_D{self.D}_n{self.n_perm}.obj"):
            return
        with open(f"OPORP_D{self.D}_n{self.n_perm}.obj", 'wb') as f:
            pickle.dump(self, f);

    def load(self):
        if not os.path.exists(f"./OPORP_D{self.D}_n{self.n_perm}.obj"):
            return False
        with open(f"OPORP_D{self.D}_n{self.n_perm}.obj", 'rb') as f:
            new_obj = pickle.load(f)
        map_location = self.map_location
        self.__dict__ = copy(new_obj.__dict__)
        self.map_location = map_location
        return True

