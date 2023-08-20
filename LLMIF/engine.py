from torch.multiprocessing import Queue, Value, Lock, Barrier
import torch.multiprocessing as mp
from torch.utils.data import default_collate
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from LLMIF.calc_inner import grad_z
from LLMIF.data_loader import get_model_tokenizer, TrainDataset, TestDataset
from LLMIF.data_loader import get_dataset_size, read_data
from LLMIF.influence_function import calc_s_test_single
from LLMIF.utils import save_json, display_progress
import numpy as np
import time
from pathlib import Path
import logging
import datetime
import os

MAX_CAPACITY = 1024

def MP_run_calc_infulence_function(rank, world_size, config, mp_engine):
    print(f"rank: {rank}, world_size: {world_size}")
    model, tokenizer = get_model_tokenizer(config['model'], device_map=f"cuda:{rank}")
    model = model.to(rank)
    print(f"CUDA {rank}: Model loaded!")

    train_dataset = TrainDataset(config['data']['train_data_path'], tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

    test_dataset = TestDataset(config['data']['test_data_path'], tokenizer)
    print(f"CUDA {rank}: Datalodaer loaded!")

    train_dataset_size = len(train_dataset)

    idx = 0
    mp_engine.start_barrier.wait()
    for i in range(len(test_dataset)):
        z_test, t_test, input_len = test_dataset[i]
        z_test = default_collate([z_test])
        t_test = default_collate([t_test])
        s_test_vec = calc_s_test_single(model, z_test, t_test, input_len, train_loader,
                                        rank, recursion_depth=config['influence']['recursion_depth'],
                                        scale=config['influence']['scale'],
                                        r=config['influence']['r_averaging'])
        while True:
            with mp_engine.train_idx.get_lock():
                idx = mp_engine.train_idx.value
                mp_engine.train_idx.value += 1
    
            if idx >= train_dataset_size:
                break
    
            z, t, input_len, real_id = train_loader.dataset[idx]
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
            mp_engine.result_q.put((i, real_id, influence), block=True, timeout=None)
            if influence != influence: # check if influence is Nan
                raise Exception('Got unexpected Nan influence!')

        mp_engine.finished_a_test.wait()
    mp_engine.result_q.put(None, block=True, timeout=None)
        

def MP_run_get_result(config, mp_engine):
    train_dataset_size = get_dataset_size(config['data']['train_data_path'])
    test_dataset_size = get_dataset_size(config['data']['test_data_path'])

    outdir = Path(config['influence']['outdir'])
    outdir.mkdir(exist_ok=True, parents=True)
    influences_path = outdir.joinpath(f"influence_results_"
                                      f"{train_dataset_size}.json")
    influences_path = save_json({}, influences_path, unique_fn_if_exists=True)

    mp_engine.start_barrier.wait()
    mp_engine.result_q.get(block=True) # get a start sign

    test_data_dicts = read_data(config['data']['test_data_path'])

    influences = {}
    influences['config'] = config
    for k in range(test_dataset_size):
        result_start_time = time.time()
        infl_list = [0 for _ in range(train_dataset_size)]

        influences[str(k)] = {}
        influences[str(k)]['test_data'] = test_data_dicts[k]
        for i in range(train_dataset_size):
            result_item = mp_engine.result_q.get(block=True)
            if result_item is None:
                save_json(influences, influences_path, overwrite_if_exists=True)
                raise Exception("Get unexpected result from queue.")
            test_id, real_id, influence = result_item
            if influence != influence: # check if influence is Nan
                raise Exception('Got unexpected Nan influence!')

            if k != test_id:
                raise Exception("Different test id.")
            infl_list[real_id] = influence
            display_progress("Calc. influence function: ", i, train_dataset_size, cur_time=time.time())

            topk_num = 500
    
            if (i + 1)%1000 == 0 or i == train_dataset_size - 1:
                helpful = np.argsort(infl_list).tolist()
                harmful = helpful[::-1]
            
                infl = [ x.tolist() if not isinstance(x, int) else x for x in infl_list ]
                # influences[str(test_id)]['influence'] = infl
                indep_index = sorted(range(len(infl)), key=lambda j: abs(infl[j]))[:topk_num]
                helpful = helpful[:topk_num]
                harmful = harmful[:topk_num]
                influences[str(test_id)]['helpful'] = helpful
                influences[str(test_id)]['harmful'] = harmful
                influences[str(test_id)]['indep'] = indep_index
                influences[str(test_id)]['indep_infl'] = [infl[x] for x in indep_index]
                influences[str(test_id)]['helpful_infl'] = [infl[x] for x in helpful]
                influences[str(test_id)]['harmful_infl'] = [infl[x] for x in harmful]
                influences[str(test_id)]['finished_cnt'] = i + 1
                influences_path = save_json(influences, influences_path, overwrite_if_exists=True)
    
        mp_engine.finished_a_test.wait()
        display_progress("Test samples processed: ", k, test_dataset_size, new_line=True, run_time=time.time()-result_start_time)
        print("-----" * 20)
    influences_path = save_json(influences, influences_path, overwrite_if_exists=True)
    print(influences_path)
    return influences 
    

class MPEngine:
    def __init__(self, world_size):
        self.result_q = Queue(maxsize=MAX_CAPACITY)

        from ctypes import c_int
        self.train_idx = Value(c_int, 0)

        self.start_barrier = Barrier(world_size + 1, action=self.put_none_to_result)
        self.finished_a_test = Barrier(world_size + 1, action=self.action_finished_a_test)

    def action_finished_a_test(self):
        with self.train_idx.get_lock():
            self.train_idx.value = 0
        
    def put_none_to_result(self):
        self.result_q.put(None)


def calc_infl_mp(config):
    gpu_num = torch.cuda.device_count()
    print(f"{gpu_num} GPUs available!")

    mp_engine = MPEngine(gpu_num)

    mp_handler = []
    for i in range(gpu_num):
        mp_handler.append(mp.Process(target=MP_run_calc_infulence_function, args=(i, gpu_num, config, mp_engine,)))
    for x in mp_handler:
        x.start()

    infl = MP_run_get_result(config, mp_engine)
    for x in mp_handler:
        x.join()
    return infl
