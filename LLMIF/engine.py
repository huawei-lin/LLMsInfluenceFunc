from torch.multiprocessing import Queue, Value, Lock, Barrier, Manager, Array
import torch.multiprocessing as mp
from torch.utils.data import default_collate
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ctypes import c_bool, c_int
import torch
from LLMIF.calc_inner import grad_z, calc_loss, get_params, normalize, pad, reshape
from LLMIF.data_loader import get_model_tokenizer, TrainDataset, TestDataset, get_tokenizer, get_model
from LLMIF.data_loader import get_dataset_size, read_data
from LLMIF.influence_function import calc_s_test_single
from LLMIF.utils import save_json, display_progress, load_json
from LLMIF.OPORP import OPORP_multi_k
import numpy as np
import time
import json
from pathlib import Path
from copy import copy
import logging
import datetime
import os
import gc
from torch.autograd import grad
from sys import getsizeof
import deepspeed


MAX_CAPACITY = 2048
MAX_DATASET_SIZE = int(1e8)


def MP_run_calc_infulence_function(rank, world_size, process_id, config, mp_engine, restart = False):
    torch.cuda.set_per_process_memory_fraction(0.48, 0)

    model = None
    tokenizer = None
    print(f"rank: {rank}, world_size: {world_size}")
    tokenizer = get_tokenizer(config.model, device_map=f"cuda:{rank}")
    print(f"CUDA {rank}: Model loaded!")

    train_dataset = TrainDataset(config.data.train_data_path, tokenizer, shuffle=False, begin_id=config.data.begin_id, end_id=config.data.end_id)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

    test_dataset = TestDataset(config.data.test_data_path, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"CUDA {rank}: Datalodaer loaded!")

    train_dataset_size = len(train_dataset)
    with mp_engine.train_dataset_size.get_lock():
        mp_engine.train_dataset_size.value = train_dataset_size
    with mp_engine.test_dataset_size.get_lock():
        mp_engine.test_dataset_size.value = len(test_dataset)

    if config.influence.deepspeed.enable == False:
        # model should be directly load in rank if deepspeed disabled
        model = get_model(config.model, device_map=f"cuda:{rank}")
        model.half()
    else:
        deepspeed_config = load_json(config.influence.deepspeed.config_path)
        model = get_model(config['model'])
        model.half()
        model, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=deepspeed_config,
        )
        model.optimizer.override_loss_scale(1)

    def get_s_test_vec_list():
        s_test_vec_list = []
        test_dataset_size = len(test_dataset)
        for i in range(test_dataset_size):
            z_test, t_test, input_len = test_dataset[i]

            s_test_vec = None
            if config.influence.infl_method == "IF":
                # TODO: implement padding and reshape yet
                s_test_vec = calc_s_test_single(model, x, t, input_len, train_dataset,
                                            gpu=rank, recursion_depth=config.influence.recursion_depth,
                                            scale=config.influence.scale,
                                            r=config.influence.r_averaging)
            else:
                s_test_vec = grad_z(z_test, t_test, input_len, model, gpu=rank, use_deepspeed=config.influence.deepspeed.enable)

            # s_test_vec = OPORP_multi_k(s_test_vec, config, [2**20], map_location=f"cuda:{rank}")[0] #

            if config.influence.offload_test_grad == True or config.influence.calculate_infl_in_gpu == False:
                s_test_vec = s_test_vec.cpu()

            s_test_vec_list.append(s_test_vec)
            display_progress("Calc. s test vector: ", i, test_dataset_size, cur_time=time.time())

        return s_test_vec_list

    s_test_vec_list = []
    if config.influence.skip_test != True:
        s_test_vec_list = get_s_test_vec_list()

    if config.influence.grads_path is not None and len(config.influence.grads_path) != 0:
        Ks = [16, 20, 24]
        grad_paths = []
        for i in range(len(Ks)):
            Ks[i] = 2**Ks[i]
        for cur_K in Ks:
            grad_paths.append(config.influence.grads_path + f"/K{cur_K}")
        for grad_path in grad_paths:
            os.makedirs(grad_path, exist_ok=True)

    if restart == False:
        mp_engine.start_barrier.wait()

    idx = 0
    while True:
        cal_word_infl = -1

        while True:
            with mp_engine.train_idx.get_lock(), mp_engine.finished_idx.get_lock(), mp_engine.cal_word_infl.get_lock():
                idx = mp_engine.train_idx.value
                mp_engine.train_idx.value = (mp_engine.train_idx.value + 1)%train_dataset_size
                if mp_engine.finished_idx[idx] == False:
                    mp_engine.finished_idx[idx] = True
                    cal_word_infl = mp_engine.cal_word_infl[idx]
                    break
            time.sleep(0.002)
    
        if idx >= train_dataset_size:
            break

        try:
            z, t, input_len, real_id = train_loader.dataset[idx]

            if cal_word_infl < 0:

                influence = None
                s_test_vec_t = None
                grad_z_vec = None
                grad_path_name = None
                grad_path_names = []
                if config.influence.grads_path is not None and len(config.influence.grads_path) != 0:
                    grad_path_name = config.influence.grads_path + f"/train_grad_{real_id:08d}.pt"
                    for path in grad_paths:
                        grad_path_names.append(path + f"/train_grad_{real_id:08d}.pt")
                # if os.path.exists(grad_path_names[-1]):
                #     mp_engine.result_q.put((0, idx, real_id, 0), block=True, timeout=None)
                #     continue

#                 if grad_path_name is not None and os.path.exists(grad_path_name):
#                     grad_z_vec = torch.load(grad_path_name, map_location=model.device)
                if grad_z_vec is None:
                    grad_z_vec = grad_z(z, t, input_len, model, gpu=rank, use_deepspeed=config.influence.deepspeed.enable)

                if config.influence.calculate_infl_in_gpu == False:
                    grad_z_vec = grad_z_vec.cpu()


                    # s_test_vec_t = s_test_vec_t.to(dtype=torch.float32)
                    # grad_z_vec_t = grad_z_vec_t.to(dtype=torch.float32)
                    # grad_z_vec_t = OPORP_multi_k(grad_z_vec_t, config, [2**20], map_location=f"cuda:{rank}")[0] #

                    # influence = torch.sum(grad_z_vec_t * s_test_vec_t).numpy()

                    # for grad_path_name, vec in zip(grad_path_names, vec_list):
                    #     torch.save(vec, grad_path_name)
#                     if grad_path_name is not None:
#                         torch.save(grad_z_vec, grad_path_name)

                for i in range(len(test_dataset)):
                    s_test_vec_t = s_test_vec_list[i]
                    if not config.influence.deepspeed.enable and config.influence.calculate_infl_in_gpu:
                        if config.influence.offload_test_grad == True:
                            s_test_vec_t = s_test_vec_t.to(rank)
                    
                    if config.influence.deepspeed.enable and config.influence.calculate_infl_in_gpu:
                        influence = None
                        for k in range(grad_z_vec.shape[0]):
                            x = grad_z_vec[i]
                            y = s_test_vec_t[i]
                            if k == 0:
                                influence = torch.sum(x.to(rank) * y.to(rank))
                            else:
                                influence = influence + torch.sum(x.to(rank) * y.to(rank))
                            x = None
                            y = None
                    else:
                        influence = torch.sum(grad_z_vec * s_test_vec_t)

                    if config.influence.calculate_infl_in_gpu == True:
                        influence = influence.cpu()
                    influence = influence.numpy()

                    if influence != influence: # check if influence is Nan
                        raise Exception('Got unexpected Nan influence!')
                    mp_engine.result_q.put((i, idx, real_id, influence), block=True, timeout=None)

            else:
                ## TODO
                # s_test_vec = [x.data.to(rank) for x in s_test_vec_list[cal_word_infl]]
                _, words_influence = grad_z(z, t, input_len, model, gpu=rank, return_words_loss=True, s_test_vec=s_test_vec_list[cal_word_infl])
                # _, words_influence = grad_z(z, t, input_len, model, gpu=rank, return_words_loss=True, s_test_vec=s_test_vec)
                # print(f"cal_word_infl: {cal_word_infl}, idx: {idx}, real_id: {real_id}, influence: {influence}")
                mp_engine.result_q.put((cal_word_infl, idx, real_id, words_influence), block=True, timeout=None)
        except Exception as e:
            with mp_engine.finished_idx.get_lock():
                mp_engine.finished_idx[idx] = False
                print(e)
            raise e

def MP_run_get_result(config, mp_engine):
    train_dataset_size = 0
    test_dataset_size = 0
    while train_dataset_size == 0 or test_dataset_size == 0:
        with mp_engine.train_dataset_size.get_lock():
            train_dataset_size = mp_engine.train_dataset_size.value
        with mp_engine.test_dataset_size.get_lock():
            test_dataset_size = mp_engine.test_dataset_size.value
        time.sleep(1)
    print(f"train_dataset_size: {train_dataset_size}, test_dataset_size: {test_dataset_size}")

    with mp_engine.train_dataset_size.get_lock(), mp_engine.finished_idx.get_lock():
        if mp_engine.train_dataset_size.value > len(mp_engine.finished_idx):
            raise Exception(f"Size of train dataset larger than MAX_DATASET_SIZE")

    outdir = Path(config.influence.outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    influences_path = outdir.joinpath(f"influence_results_"
                                      f"{train_dataset_size}.json")
    influences_path = save_json({}, influences_path, unique_fn_if_exists=True)

    mp_engine.start_barrier.wait()
    # mp_engine.result_q.get(block=True) # get a start sign

    test_data_dicts = read_data(config.data.test_data_path)

    influences = {}
    influences['config'] = str(config)
    for k in range(test_dataset_size):
        influences[k] = {}
        influences[k]['test_data'] = test_data_dicts[k]
    
    infl_list = [[0 for _ in range(train_dataset_size)] for _ in range(test_dataset_size)]
    real_id2shuffled_id = {}
    shuffled_id2real_id = {}
    
    total_size = test_dataset_size * train_dataset_size
    
    i = 0
    while True:
        try:
            # result_item = mp_engine.result_q.get(block=True, timeout=300)
            result_item = mp_engine.result_q.get(block=True)
        except Exception as e:
            print("Cal Influence Function Finished!")
            break
    
        if result_item is None:
            save_json(influences, influences_path, overwrite_if_exists=True)
            raise Exception("Get unexpected result from queue.")
        test_id, shuffled_id, real_id, influence = result_item
        # print(f"get, i: {i} real_id: {real_id}")
        if influence != influence: # check if influence is Nan
            raise Exception('Got unexpected Nan influence!')

        infl_list[test_id][shuffled_id] = influence
        real_id2shuffled_id[real_id] = shuffled_id
        shuffled_id2real_id[shuffled_id] = real_id
        with mp_engine.finished_idx.get_lock():
            mp_engine.finished_idx[shuffled_id] = True # due to the calculating retrive data by shuffled_id
        display_progress("Calc. influence function: ", i, total_size, cur_time=time.time())
    
        topk_num = int(config.influence.top_k)
    
        # if (i + 1)%5000 == 0 or i == total_size - 1:
        if (i + 1)%10 == 0 or i == total_size - 1:
            for j in range(test_dataset_size):
                harmful_shuffle_ids = np.argsort(infl_list[j]).tolist()
                harmful = [ shuffled_id2real_id[x] for x in harmful_shuffle_ids if x in shuffled_id2real_id.keys() ]
                helpful = harmful[::-1]
            
                infl = [ x.tolist() if not isinstance(x, int) else x for x in infl_list[j] ]
                # words_infl = [ x.tolist() if not isinstance(x, list) else x for x in words_infl_list ]
                # influences[test_id]['influence'] = infl
                helpful_topk = helpful[:topk_num]
                harmful_topk = harmful[:topk_num]
                influences[j]['helpful'] = copy(helpful_topk)
                influences[j]['helpful_infl'] = copy([infl[x] for x in harmful_shuffle_ids[-topk_num:][::-1]])
                influences[j]['harmful'] = copy(harmful_topk)
                influences[j]['harmful_infl'] = copy([infl[x] for x in harmful_shuffle_ids[:topk_num]])
            influences['finished_cnt'] = f"{i + 1}/{total_size}"
            influences_path = save_json(influences, influences_path, overwrite_if_exists=True)
        # print(f"i: {i} real_id: {real_id}")
        i += 1
        if i >= total_size:
            finished = True
            with mp_engine.finished_idx.get_lock():
                for idx in range(train_dataset_size):
                    if mp_engine.finished_idx[idx] == False:
                        print("Warning: i >= total_size, but it have not finished!")
                        finished = False
                        break
            if finished == True:
                break

    # print(helpful, len(helpful))

    if config.influence.cal_words_infl == True:
        # Calculate Word Influence
        for j in range(test_dataset_size):
            word_infl_dict = {}

            infl_num = len(set(influences[j]['helpful'] + influences[j]['harmful']))
            # print(influences[j]['helpful'])
            with mp_engine.train_idx.get_lock(), mp_engine.finished_idx.get_lock(), mp_engine.cal_word_infl.get_lock():
                for x in influences[j]['helpful']:
                    mp_engine.cal_word_infl[real_id2shuffled_id[x]] = j
                    mp_engine.finished_idx[real_id2shuffled_id[x]] = False
                for x in influences[j]['harmful']:
                    mp_engine.cal_word_infl[real_id2shuffled_id[x]] = j
                    mp_engine.finished_idx[real_id2shuffled_id[x]] = False

            i = 0
            # for i in range(infl_num):
            while True:
                try:
                    result_item = mp_engine.result_q.get(block=True, timeout=300)
                except Exception as e:
                    print(e)
                    break
                if result_item is None:
                    save_json(influences, influences_path, overwrite_if_exists=True)
                    raise Exception("Get unexpected result from queue.")
                test_id, shuffled_id, real_id, word_influence = result_item
                # print(f"i: {i}, test_id: {test_id}, real_id: {real_id}")
                with mp_engine.finished_idx.get_lock(), mp_engine.cal_word_infl.get_lock():
                    mp_engine.finished_idx[shuffled_id] = True
                    mp_engine.cal_word_infl[shuffled_id] = -1

                word_infl_dict[real_id] = word_influence.tolist() if not isinstance(word_influence, list) else word_influence
                display_progress(f"Calc. word influence for test {j + 1}/{test_dataset_size}", i, infl_num, cur_time=time.time())
                i += 1
                if i >= infl_num:
                    finished = True
                    with mp_engine.finished_idx.get_lock():
                        for idx in range(train_dataset_size):
                            if mp_engine.finished_idx[idx] == False:
                                print("Warning: i >= total_size, but it have not finished!")
                                finished = False
                                break
                    if finished == True:
                        break
            influences[j]['word_influence'] = word_infl_dict
            try:
                influences_path = save_json(influences, influences_path, overwrite_if_exists=True)
            except Exception as e:
                print(influences)
                print(e)
            # print(done_id, len(done_id))

    # display_progress("Test samples processed: ", k, test_dataset_size, new_line=True, run_time=time.time()-result_start_time)
    # print("-----" * 20)
    influences_path = save_json(influences, influences_path, overwrite_if_exists=True)
    print(influences_path)
    return influences 
    

class MPEngine:
    def __init__(self, world_size):
        self.result_q = Queue(maxsize=MAX_CAPACITY)

        self.train_idx = Value(c_int, 0)

        self.start_barrier = Barrier(world_size + 1)
        # self.start_barrier = Barrier(world_size + 1, action=self.put_none_to_result)
        # self.finished_a_test = Barrier(world_size + 1, action=self.action_finished_a_test)
        self.finished_a_test = Value(c_int, 0)
        self.cur_processes_num = Value(c_int, 0)

        self.train_dataset_size = Value(c_int, 0)
        self.test_dataset_size = Value(c_int, 0)

        self.finished_idx = Array(c_bool, [False for _ in range(MAX_DATASET_SIZE)])

        # -1, doesn't compute word infl.
        # > -1, compute word infl for # test data
        self.cal_word_infl = Array(c_int, [-1 for _ in range(MAX_DATASET_SIZE)])

    def action_finished_a_test(self):
        with self.train_idx.get_lock():
            self.train_idx.value = 0

#     def put_none_to_result(self):
#         self.result_q.put(None)


def calc_infl_mp(config):
    gpu_num = torch.cuda.device_count()
    print(f"{gpu_num} GPUs available!")

    threads_per_gpu = int(config.influence.n_threads)

    if config.influence.grads_path is not None and len(config.influence.grads_path) != 0:
        os.makedirs(config.influence.grads_path, exist_ok=True)

    num_processing = gpu_num * threads_per_gpu
    mp_engine = MPEngine(num_processing)

    mp_handler = []
    mp_args = []
    for i in range(gpu_num):
        for j in range(threads_per_gpu):
            mp_handler.append(mp.Process(target=MP_run_calc_infulence_function, args=(i, gpu_num, i*threads_per_gpu + j, config, mp_engine,)))
            mp_args.append(mp_handler[-1]._args)
    mp_handler.append(mp.Process(target=MP_run_get_result, args=(config, mp_engine)))

    for x in mp_handler:
        x.start()

    while mp_handler[-1].is_alive():
#         cur_processes_num = len([1 for x in mp_handler if x.is_alive()])
#         if cur_processes_num < num_processing + 1:
#             print(f"ready to restart processing, {cur_processes_num}/{num_processing}")
#             for i, x in enumerate(mp_handler):
#                 if x.is_alive() != True:
#                     print(f"start {mp_args[i]}")
#                     mp_handler[i] = mp.Process(target=MP_run_calc_infulence_function, args=mp_args[i] + (True,))
#                     mp_handler[i].start()
#             continue
#         with mp_engine.cur_processes_num.get_lock():
#             mp_engine.cur_processes_num.value = cur_processes_num
        time.sleep(1)

    # infl = MP_run_get_result(config, mp_engine)
    for x in mp_handler:
        x.terminate()
        # x.join()

    # return infl

