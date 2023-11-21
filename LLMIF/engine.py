from torch.multiprocessing import Queue, Value, Lock, Barrier, Manager, Array
import torch.multiprocessing as mp
from torch.utils.data import default_collate
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ctypes import c_bool, c_int, c_double
import torch
from LLMIF.calc_inner import grad_z
from LLMIF.data_loader import get_model_tokenizer, TrainDataset, TestDataset
from LLMIF.data_loader import get_dataset_size, read_data
from LLMIF.influence_function import calc_s_test_single
from LLMIF.utils import save_json, display_progress
import numpy as np
import time
from pathlib import Path
from copy import copy
import logging
import datetime
import os
import collections

MAX_CAPACITY = 2048
MAX_DATASET_SIZE = int(1e8)
BATCH_SIZE = 128

def load_grad_z_vec(begin_idx, end_idx, loaded_idx, grad_z_queue, grad_path_prefix):
    while True:
        with loaded_idx.get_lock():
            l_idx = loaded_idx.value
            if l_idx > end_idx:
                break
            grad_path_name = grad_path_prefix + f"/train_grad_{l_idx:08d}.pt"
            # grad_z_vec = torch.load(grad_path_name, map_locatiol=model.device)
            try:
                if os.path.exists(grad_path_name):
                    # print(f'load: {l_idx}')
                    grad_z_vec = torch.load(grad_path_name, map_location='cpu')
                    grad_z_queue.put((l_idx, grad_z_vec), block=True, timeout=None)
                else:
                    grad_z_queue.put((l_idx, None), block=True, timeout=None)
                loaded_idx.value += 1
            except Exception as e:
                print(grad_path_name)
                raise e
            time.sleep(0.005)

def MP_run_calc_infulence_function(rank, world_size, process_id, process_worldsize, config, mp_engine):
    time_record = collections.defaultdict(lambda: 0)

    print(f"rank: {rank}, world_size: {world_size}")
    model, tokenizer = get_model_tokenizer(config['model'], device_map=f"cuda:{rank}")
    model = model.to(rank)
    print(f"CUDA {rank}: Model loaded!")

    # train_dataset = TrainDataset(config['data']['train_data_path'], tokenizer, shuffle=False, load_idx_list=load_idx_list)
    train_dataset = TrainDataset(config['data']['train_data_path'], tokenizer, shuffle=False)
    train_dataset_size = len(train_dataset)
    with mp_engine.train_dataset_size.get_lock():
        mp_engine.train_dataset_size.value = train_dataset_size

    step = (train_dataset_size + process_worldsize - 1)//process_worldsize
    begin_idx = process_id * step
    end_idx = min(train_dataset_size, (process_id + 1)*step)

    print(begin_idx, end_idx)
    train_dataset = torch.utils.data.Subset(train_dataset, range(begin_idx, end_idx))
    # train_loader = DataLoader(train_dataset[begin_idx: end_idx], batch_size=1, shuffle=False, num_workers=0)
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4, persistent_workers=True)
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

    test_dataset = TestDataset(config['data']['test_data_path'], tokenizer)
    print(f"CUDA {rank}: Datalodaer loaded!")
    with mp_engine.test_dataset_size.get_lock():
        mp_engine.test_dataset_size.value = len(test_dataset)


    s_test_vec_list = []
    test_dataset_size = len(test_dataset)
    for i in range(test_dataset_size):
        z_test, t_test, input_len = test_dataset[i]
        z_test = default_collate([z_test])
        t_test = default_collate([t_test])
        s_test_vec = calc_s_test_single(model, z_test, t_test, input_len, train_dataset=train_dataset,
                                        gpu=rank, recursion_depth=config['influence']['recursion_depth'],
                                        scale=config['influence']['scale'],
                                        r=config['influence']['r_averaging'])
        # s_test_vec = [x.data.cpu() for x in s_test_vec]
        s_test_vec_list.append(s_test_vec)
        display_progress("Calc. s test vector: ", i, test_dataset_size, cur_time=time.time())


    loaded_idx = Value(c_int, begin_idx)
    grad_z_queue = Queue(maxsize=8) # num_workers

    grad_path_name = None
    if config["influence"]["grads_path"] is not None and len(config["influence"]["grads_path"]) != 0:
        grad_path_prefix = config["influence"]["grads_path"]
        mp_handler = []
        for _ in range(1):
            mp_handler.append(mp.Process(target=load_grad_z_vec, args=(begin_idx, end_idx, loaded_idx, grad_z_queue, grad_path_prefix,)))
        for x in mp_handler:
            x.start()
        while grad_z_queue.empty() == True:
            time.sleep(0.1)


    idx = 0
    cal_word_infl = -1
    mp_engine.start_barrier.wait()

    total_time = 0
    for idx, items in zip(range(begin_idx, end_idx), train_dataset):
        try:
            start_time = time.time()
            time_point = time.time()
            z, t, input_len, real_id = items
#             z = train_loader.collate_fn([z])
#             t = train_loader.collate_fn([t])
            z = z.unsqueeze(0)
            t = t.unsqueeze(0)
            if z.dim() > 2:
                z = torch.squeeze(z, 0)
            if t.dim() > 2:
                t = torch.squeeze(t, 0)
            time_record["get_training_data"] += time.time() - time_point
            time_point = time.time()

            if cal_word_infl < 0:
                # grad_z_vec = grad_z(z, t, input_len, model, gpu=rank)
                # grad_z_vec = [x.data.cpu() for x in grad_z_vec]
                grad_z_vec = None
                grad_path_name = None

                time_point = time.time()
                if config["influence"]["grads_path"] is not None and len(config["influence"]["grads_path"]) != 0:
                    grad_path_name = config["influence"]["grads_path"] + f"/train_grad_{real_id:08d}.pt"
                grad_z_idx, grad_z_vec = grad_z_queue.get(block=True)
                time_record["get_grad_z_vec_from_queue"] += time.time() - time_point
                time_point = time.time()

                # print("get", grad_z_idx)
                # if grad_z_idx == idx:
                #     grad_z_vec = grad_z_vec_temp
                # else:
                #     pass
                    # print(f"NO!!!! idx: {idx} grad_z_idx: {grad_z_idx}")
                    # raise Exception(f"NO!!!! idx: {idx} grad_z_idx: {grad_z_idx}")
                if grad_z_vec is None:
                    grad_z_vec = grad_z(z, t, input_len, model, gpu=rank)
                    grad_z_vec = [x.data.cpu() for x in grad_z_vec]
                    if grad_path_name is not None:
                        torch.save(grad_z_vec, grad_path_name)

                time_record["get_grad_z_vec_from_gradz"] += time.time() - time_point
                time_point = time.time()


                for i in range(len(test_dataset)):
                    # s_test_vec = [x.data.to(rank) for x in s_test_vec_list[i]]
                    grad_z_vec = [x.to(model.device) for x in grad_z_vec]
                    influence = -sum(
                        [
                            torch.sum(k * j).data.cpu().numpy()
                            # torch.sum(k * j)
                            for k, j in zip(grad_z_vec, s_test_vec_list[i])
                            # for k, j in zip(grad_z_vec, s_test_vec)
                        ]) / train_dataset_size

                    if influence != influence: # check if influence is Nan
                        raise Exception('Got unexpected Nan influence!')
                    # (test id, shuffle id, original id, influence score)
                    mp_engine.result_q.put((i, idx, real_id, influence), block=True, timeout=None)
                total_time += time.time() - start_time 

                time_record["get_infl"] += time.time() - time_point
                time_record["total_time"] += time.time() - start_time
                time_record["cnt"] += 1

                if time_record["cnt"]%200 == 0:
                    print(f"time: {total_time}")
                    print(f"size: {grad_z_queue.qsize()}")
                    print(f"{time_record}")
                    total_time = 0
                    time_record = collections.defaultdict(lambda: 0)
                # if time_record["cnt"]%100 == 0:
                #     print(time_record)

                    # print(f"idx: {idx}, real_id: {real_id}, influence: {influence}")
            else:
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

    outdir = Path(config['influence']['outdir'])
    outdir.mkdir(exist_ok=True, parents=True)
    influences_path = outdir.joinpath(f"influence_results_"
                                      f"{train_dataset_size}.json")
    influences_path = save_json({}, influences_path, unique_fn_if_exists=True)

    mp_engine.start_barrier.wait()
    # mp_engine.result_q.get(block=True) # get a start sign

    test_data_dicts = read_data(config['data']['test_data_path'])

    influences = {}
    influences['config'] = config
    for k in range(test_dataset_size):
        influences[k] = {}
        influences[k]['test_data'] = test_data_dicts[k]

    # infl_list = [[0 for _ in range(train_dataset_size)] for _ in range(test_dataset_size)]
    real_id2shuffled_id = {}
    shuffled_id2real_id = {}

    total_size = test_dataset_size * train_dataset_size

    i = 0
    total_time = 0
    while True:
        try:
            begin = time.time()
            result_item = mp_engine.result_q.get(block=True, timeout=120)
            total_time += time.time() - begin
            if i%100 == 0:
                # print(total_time)
                total_time = 0
        except Exception as e:
            print("Cal Influence Function Finished!")
            print(e)
            break

        if result_item is None:
            save_json(influences, influences_path, overwrite_if_exists=True)
            raise Exception("Get unexpected result from queue.")
        test_id, shuffled_id, real_id, influence = result_item
        if influence != influence: # check if influence is Nan
            raise Exception('Got unexpected Nan influence!')

        # infl_list[test_id][shuffled_id] = influence
        with mp_engine.infl_list.get_lock():
            mp_engine.infl_list[real_id] = influence
        real_id2shuffled_id[real_id] = shuffled_id
        shuffled_id2real_id[shuffled_id] = real_id
        with mp_engine.finished_idx.get_lock():
            mp_engine.finished_idx[shuffled_id] = True # due to the calculating retrive data by shuffled_id

        with mp_engine.collect_cnt.get_lock():
            mp_engine.collect_cnt.value += 1
            i = mp_engine.collect_cnt.value

        display_progress("Calc. influence function: ", i, total_size, cur_time=time.time())

        # topk_num = int(config['influence']['top_k'])
    
#         if (i + 1)%500 == 0 or i == total_size - 1:
#             print(mp_engine.result_q.qsize())
#             for j in range(test_dataset_size):
#                 helpful_shuffle_ids = np.argsort(infl_list[j]).tolist()
#                 helpful = [ shuffled_id2real_id[x] for x in helpful_shuffle_ids if x in shuffled_id2real_id.keys() ]
#                 harmful = helpful[::-1]
#             
#                 infl = [ x.tolist() if not isinstance(x, int) else x for x in infl_list[j] ]
#                 # words_infl = [ x.tolist() if not isinstance(x, list) else x for x in words_infl_list ]
#                 # influences[test_id]['influence'] = infl
#                 helpful = helpful[:topk_num]
#                 influences[j]['helpful'] = copy(helpful)
#                 influences[j]['helpful_infl'] = copy([infl[x] for x in helpful_shuffle_ids])
#             influences['finished_cnt'] = f"{i + 1}/{total_size}"
#             influences_path = save_json(influences, influences_path, overwrite_if_exists=True)
        # print(f"i: {i} real_id: {real_id}")
    # print(helpful, len(helpful))

    if config['influence']['cal_words_infl'] == True:
        # Calculate Word Influence
        for j in range(test_dataset_size):
            word_infl_dict = {}
            done_id = []

            infl_num = len(influences[j]['helpful'])
            # print(influences[j]['helpful'])
            with mp_engine.train_idx.get_lock(), mp_engine.finished_idx.get_lock(), mp_engine.cal_word_infl.get_lock():
                for x in influences[j]['helpful']:
                    mp_engine.cal_word_infl[real_id2shuffled_id[x]] = j
                    mp_engine.finished_idx[real_id2shuffled_id[x]] = False

            i = 0
            # for i in range(infl_num):
            while True:
                try:
                    result_item = mp_engine.result_q.get(block=True, timeout=120)
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

                done_id.append(real_id)
                word_infl_dict[real_id] = word_influence.tolist() if not isinstance(word_influence, list) else word_influence
                display_progress(f"Calc. word influence for test {j + 1}/{test_dataset_size}", i, infl_num, cur_time=time.time())
                i += 1
            influences[j]['word_influence'] = word_infl_dict
            influences_path = save_json(influences, influences_path, overwrite_if_exists=True)
            # print(done_id, len(done_id))

    # display_progress("Test samples processed: ", k, test_dataset_size, new_line=True, run_time=time.time()-result_start_time)
    # print("-----" * 20)
    influences_path = save_json(influences, influences_path, overwrite_if_exists=True)
    print(influences_path)
    return influences 
    

class MPEngine:
    def __init__(self, world_size, collect_threads_num):
        self.result_q = Queue(maxsize=MAX_CAPACITY)

        self.train_idx = Value(c_int, 0)
        self.collect_cnt = Value(c_int, 0)

        self.start_barrier = Barrier(world_size + collect_threads_num)
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
        self.infl_list = Array(c_double, [0 for _ in range(MAX_DATASET_SIZE)])

    def action_finished_a_test(self):
        with self.train_idx.get_lock():
            self.train_idx.value = 0

#     def put_none_to_result(self):
#         self.result_q.put(None)


def calc_infl_mp(config):
    gpu_num = torch.cuda.device_count()
    print(f"{gpu_num} GPUs available!")

    threads_per_gpu = 1
    if "n_threads" in config['influence'].keys():
        threads_per_gpu = int(config['influence']['n_threads'])

    collect_threads_num = 1

    if config['influence']['grads_path'] is not None and len(config['influence']['grads_path']) != 0:
        os.makedirs(config['influence']['grads_path'], exist_ok=True)

    mp_engine = MPEngine(gpu_num * threads_per_gpu, collect_threads_num)

    mp_handler = []
    for i in range(gpu_num):
        for j in range(threads_per_gpu):
            mp_handler.append(mp.Process(target=MP_run_calc_infulence_function, args=(i, gpu_num, i*threads_per_gpu + j, gpu_num*threads_per_gpu, config, mp_engine,)))
    for i in range(collect_threads_num):
        mp_handler.append(mp.Process(target=MP_run_get_result, args=(config, mp_engine,)))

    for x in mp_handler:
        x.start()

    while mp_handler[-1].is_alive():
        cur_processes_num = len([1 for x in mp_handler if x.is_alive()])
        with mp_engine.cur_processes_num.get_lock():
            mp_engine.cur_processes_num.value = cur_processes_num
        time.sleep(1)

    # infl = MP_run_get_result(config, mp_engine)
    for x in mp_handler:
        x.terminate()
        # x.join()

    # return infl

