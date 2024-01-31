import os
from torch.multiprocessing import Queue, Value, Lock, Barrier, Manager, Array
import torch.multiprocessing as mp
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from torch.utils.data import default_collate
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from LLMIF.calc_inner import grad_z
from LLMIF.data_loader import get_model_tokenizer, TrainDataset, TestDataset
from LLMIF.data_loader import get_dataset_size, read_data
from LLMIF.influence_function import calc_s_test_single
from LLMIF.utils import save_json, display_progress, load_json, init_logging, get_config
import numpy as np
import time
import json
from pathlib import Path
from copy import copy
import logging
import datetime
from tqdm import tqdm
import gc

import pickle
from sklearn.decomposition import TruncatedSVD

MAX_CAPACITY = 2048

def train_dims_reduction(rank, result_q, start_barrier, config, train_num, random_seed):
    model, tokenizer = get_model_tokenizer(config['model'], device_map=f"cuda:{rank}")
    model = model.to(rank)

    train_dataset = TrainDataset(config['data']['train_data_path'], tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

    train_dataset_size = len(train_dataset)
    np.random.seed(random_seed)
    idx_list = np.random.choice(train_dataset_size, train_num)
    print(idx_list, len(idx_list))
    start_barrier.wait()

    for i, idx in enumerate(idx_list):
        z, t, input_len, real_id = train_loader.dataset[idx]
        z = train_loader.collate_fn([z])
        t = train_loader.collate_fn([t])
        if z.dim() > 2:
            z = torch.squeeze(z, 0)
        if t.dim() > 2:
            t = torch.squeeze(t, 0)

        grad_z_vec = grad_z(z, t, input_len, model, gpu=rank).reshape((-1, 4096))
        keep_list = np.random.choice(grad_z_vec.shape[0], int(grad_z_vec.shape[0]*config["dimentionality"]["keep_p"]))
        grad_z_vec_cpu = grad_z_vec[keep_list, :].cpu().numpy()

        del grad_z_vec
        gc.collect()
        torch.cuda.empty_cache()

        result_q.put((idx, grad_z_vec_cpu), block=True, timeout=None)


def collect_result(config):
    result_q = Queue(maxsize=MAX_CAPACITY)

    gpu_num = torch.cuda.device_count()
    print(f"{gpu_num} GPUs available!")

    threads_per_gpu = 1
    if "n_threads" in config['dimentionality'].keys():
        threads_per_gpu = int(config['dimentionality']['n_threads'])

    world_size = gpu_num * threads_per_gpu

    total_train_num = int(config["dimentionality"]["train_num"])
    train_num = total_train_num//world_size
    shift = total_train_num - train_num*world_size
    train_num_list = [(train_num + 1) if x < shift else train_num for x in range(world_size)]
    print(f"train_num_list:", train_num_list)

    start_barrier = Barrier(world_size + 1)

    mp_handler = []
    for i in range(gpu_num):
        for j in range(threads_per_gpu):
            mp_handler.append(mp.Process(target=train_dims_reduction, args=(i, result_q, start_barrier, config, train_num_list[i*threads_per_gpu + j], 42*(i*threads_per_gpu + j))))

    for x in mp_handler:
        x.start()

    start_barrier.wait()

    data_mat = None
    for i in tqdm(range(total_train_num)):
        idx, grad_z_vec = result_q.get(block=True) # get a start sign
        r, c = grad_z_vec.shape

        if data_mat is None:
            print(grad_z_vec.shape)
            data_mat = np.zeros((int(config["dimentionality"]["train_num"])*r, c), dtype=np.float32)
        data_mat[r*i:r*(i + 1), :] = grad_z_vec

    print(data_mat.shape)
    for x in mp_handler:
        x.terminate()

#     start_time = time.time()
#     data_mat = torch.tensor(data_mat).cuda(0)
#     data_mat = data_mat.reshape((1, -1, 4096))
#     U, S, Vh = torch.linalg.svd(data_mat, full_matrices=True)
#     # U, s, Vt = rlinalg.rsvd(data_mat, k=int(config["dimentionality"]["n_comps"]), method='standard')
#     end_time = time.time()
#     print("fit:", end_time - start_time)
#     print(torch.dist(data_mat, U @ torch.diag_embed(S) @ Vh))
#     exit()


    start_time = time.time()
    svd = TruncatedSVD(n_components=int(config["dimentionality"]["n_comps"]))
    svd.fit(data_mat)
    end_time = time.time()
    print("fit:", end_time - start_time)
    data_list = svd.transform(data_mat[:, :])
    end_time = time.time()
    print(data_list.shape, end_time - start_time)
    with open(config["dimentionality"]["svd_model_name"], 'wb') as pickle_file:
        pickle.dump(svd, pickle_file)

