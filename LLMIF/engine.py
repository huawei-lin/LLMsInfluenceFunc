from multiprocessing import Process
from multiprocessing import Queue
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


def MP_run_calc_infulence_function(config, model_path, train_loader, test_loader, test_id, train_data_q, result_q, recursion_depth, r, gpu):
    model = get_model(model_path)
    model = model.cuda(gpu)
    print(f"CUDA {gpu}: Model loaded!")

    z_test, t_test, input_len = test_loader.dataset[test_id]
    z_test = test_loader.collate_fn([z_test])
    t_test = test_loader.collate_fn([t_test])
    s_test_vec = calc_s_test_single(model, z_test, t_test, input_len, train_loader,
                                    gpu, recursion_depth=recursion_depth,
                                        r=r)
    print(f"Got s_test_vec!")

    train_dataset_size = len(train_loader.dataset)
    while True:
        data_item = train_data_q.get()
        if data_item is None:
            break
        z, t, input_len, real_id = data_item
        if z.dim() > 2:
            z = torch.squeeze(z, 0)
        if t.dim() > 2:
            t = torch.squeeze(t, 0)
        grad_z_vec = grad_z(z, t, input_len, model, gpu=gpu)[0]
        influence = -sum(
            [
                torch.sum(k * j).data.cpu().numpy()
                for k, j in zip(grad_z_vec, s_test_vec)
            ]) / train_dataset_size
        result_q.put((real_id, influence))
        

def MP_run_get_result(config, result_q, test_loader, test_id, train_dataset_size):
    infl_list = [0 for _ in range(train_dataset_size)]
    for i in range(train_dataset_size):

        real_id, influence = result_q.get()
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
    print(influences)
    influences_path = save_json(influences, influences_path)
    


class MPEngine:
    def __init__(self, train_dataloader, gpu_num):
        self.train_data_q = Queue()
        self.result_q = Queue()
        self.feed_train_data(train_dataloader, gpu_num)
        self.gpu_num = gpu_num
        
    def feed_train_data(self, train_loader, gpu_num, start: int = 0):
        print("Putting data to multiprocess queue")
        for i in range(start, len(train_loader.dataset)):
            z, t, input_len, real_id = train_loader.dataset[i]
            z = train_loader.collate_fn([z])
            t = train_loader.collate_fn([t])
            self.train_data_q.put((z, t, input_len, real_id))
        for _ in range(gpu_num):
            self.train_data_q.put(None)
        print(f"Finish putting data to multiprocess queue, qsize: {self.train_data_q.qsize()}")


