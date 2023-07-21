from torch.multiprocessing import Queue
import torch.distributed as dist
import torch
from LLMIF import get_model
from LLMIF.calc_inner import grad_z
from LLMIF.influence_function import calc_s_test_single
from LLMIF.utils import save_json, display_progress
import numpy as np
import time
from pathlib import Path
import logging
import datetime
import os

MAX_CAPACITY = 100


def MP_run_calc_infulence_function(rank, world_size, config, model_path, train_loader, test_loader, test_id, mp_engine):
    print(f"rank: {rank}, world_size: {world_size}")
    model = get_model(model_path)
    model = model.to(rank)
    print(f"CUDA {rank}: Model loaded!")

    z_test, t_test, input_len = test_loader.dataset[test_id]
    z_test = test_loader.collate_fn([z_test])
    t_test = test_loader.collate_fn([t_test])
    s_test_vec = calc_s_test_single(model, z_test, t_test, input_len, train_loader,
                                    rank, recursion_depth=config['recursion_depth'],
                                        r=config['r_averaging'])
    print(f"Got s_test_vec!")

    train_dataset_size = len(train_loader.dataset)
    step = (train_dataset_size + world_size - 1)//world_size
    start = rank * step
    end = min((rank + 1) * step, train_dataset_size)
    print(f"CUDA {rank}: data_range ({start}, {end})")

    for i in range(start, end):
        z, t, input_len, real_id = train_loader.dataset[i]
        z = train_loader.collate_fn([z])
        t = train_loader.collate_fn([t])
        if z.dim() > 2:
            z = torch.squeeze(z, 0)
        if t.dim() > 2:
            t = torch.squeeze(t, 0)
        grad_z_vec = grad_z(z, t, input_len, model, gpu=rank)[0]
        influence = -sum(
            [
                torch.sum(k * j).data.cpu().numpy()
                for k, j in zip(grad_z_vec, s_test_vec)
            ]) / train_dataset_size
        mp_engine.result_q.put((real_id, influence), block=True, timeout=None)
        

def MP_run_get_result(config, mp_engine, test_loader, test_id, train_dataset_size):
    infl_list = [0 for _ in range(train_dataset_size)]
    for i in range(train_dataset_size):

        real_id, influence = mp_engine.result_q.get(block=True)
        infl_list[real_id] = influence
        display_progress("Calc. influence function: ", i, train_dataset_size, cur_time=time.time())

    harmful = np.argsort(infl_list).tolist()
    helpful = harmful[::-1]

    influences = {}
    influences[str(test_id)] = {}
    infl = [x.tolist() for x in infl_list]
    # influences[str(test_id)]['influence'] = infl
    influences[str(test_id)]['test_data'] = test_loader.dataset.list_data_dict[test_id]
    indep_index = sorted(range(len(infl)), key=lambda i: abs(infl[i]))[:100]
    helpful = helpful[:100]
    harmful = harmful[:100]
    influences[str(test_id)]['helpful'] = helpful
    influences[str(test_id)]['harmful'] = harmful
    influences[str(test_id)]['indep'] = indep_index
    influences[str(test_id)]['indep_infl'] = [infl[x] for x in indep_index]
    influences[str(test_id)]['helpful_infl'] = [infl[x] for x in helpful]
    influences[str(test_id)]['harmful_infl'] = [infl[x] for x in harmful]

    outdir = Path(config['outdir'])
    outdir.mkdir(exist_ok=True, parents=True)
    influences_path = outdir.joinpath(f"influence_results_"
                                      f"{train_dataset_size}.json")
    influences_path = save_json(influences, influences_path)
    

class MPEngine:
    def __init__(self, train_dataloader, gpu_num):
        self.result_q = Queue(maxsize=MAX_CAPACITY)
        self.gpu_num = gpu_num
        


