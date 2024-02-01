from torch.multiprocessing import Queue, Value, Lock, Barrier, Manager, Array
import torch.multiprocessing as mp
from ctypes import c_bool, c_int
from LLMIF.data_loader import get_model_tokenizer, TrainDataset, TestDataset, get_tokenizer, get_model
from LLMIF.calc_inner import grad_z
from LLMIF.utils import save_json, display_progress, load_json
from torch.utils.data import default_collate
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
from pathlib import Path
import torch
import time

MAX_DATASET_SIZE = int(1e8)

D = None
K = None
M = None

random_mat = None
perm_mat = None


def load_MK(config):
    global K, M
    if K is None:
        K = int(config["influence"]["OPORP_K"])
    if M is None:
        M = 1


def load_random_mat(config, map_location=None):
    global random_mat, M, D
    if random_mat is None:
        if os.path.exists(f"./random_mat_D{D}_M{M}.pt"):
            print(f"./random_mat_D{D}_M{M}.pt")
            random_mat = torch.load(f"./random_mat_D{D}_M{M}.pt", map_location=map_location)
            # random_mat = random_mat.to(vec.device)
            random_mat = random_mat.to(dtype=torch.float16)
            # random_mat = random_mat.cpu()
        else:
            random_mat = torch.randint(0, 2, (M*D,), dtype=torch.int8, device=map_location)
            random_mat[random_mat < 1e-8] = -1
            torch.save(random_mat, f"./random_mat_D{D}_M{M}.pt")


def load_perm_mat(config):
    global perm_mat, M, D
    if perm_mat is None:
        if os.path.exists(f"./perm_mat_D{D}_M{M}.npy"):
            perm_mat = np.load(f"./perm_mat_D{D}_M{M}.npy")
        else:
            perm_mat = np.zeros((M*D))
            for i in range(M):
                perm_mat[i*D:(i + 1)*D] = np.random.permutation(D) + (i * D)
            np.save(f"./perm_mat_D{D}_M{M}.npy", perm_mat)


def OPORP(vec, config, map_location=None):
    global random_mat, perm_mat, M, D, K
    if M is None or K is None:
        load_MK(config)
    if D is None:
        D = len(vec)
        # D = ((len(vec) - 1)//K + 1)*K
    if random_mat is None:
        load_random_mat(config, map_location)
    if perm_mat is None:
        load_perm_mat(config)

    if len(vec) != D:
        vec = torch.nn.functional.pad(vec, [0, D - len(vec)])

    vec = vec.repeat(M)
    vec = vec[perm_mat]

    vec = vec*random_mat

    step = D//K
    vec = torch.sum(vec.reshape((-1, step)), axis=1)
    return vec

def OPORP_multi_k(vec, config, Ks, map_location=None):
    global random_mat, perm_mat, M, D, K
    if M is None or K is None:
        load_MK(config)
    if D is None:
        D = len(vec)
        # D = ((len(vec) - 1)//K + 1)*K
    if random_mat is None:
        load_random_mat(config, map_location)
    if perm_mat is None:
        load_perm_mat(config)

    # vec = vec.repeat(M)
    # if len(vec) != D:
    #     vec = torch.nn.functional.pad(vec, [0, D - len(vec)])
    print(f"vec: {vec.dtype}")

    vec = vec.cpu()
    vec = vec[perm_mat]
    # vec = torch.gather(vec, 0, torch.LongTensor(perm_mat))
    vec = vec.to(map_location)
    vec = vec*random_mat

    ans = []
    for k in Ks:
        step = D//k
        # ans.append(torch.sum(vec.reshape((-1, step)), axis=1).cpu())
        ans.append(torch.sum(vec.reshape((-1, step)), axis=1))

    return ans


def MP_run_calc_infulence_function(rank, world_size, process_id, config, mp_engine):
    print(f"rank: {rank}, world_size: {world_size}")
    model, tokenizer = get_model_tokenizer(config['model'], device_map=f"cuda:{rank}")
    print(f"CUDA {rank}: model, tokenizer loaded!")

    train_dataset = TrainDataset(config['data']['train_data_path'], tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

    train_dataset_size = len(train_dataset)
    with mp_engine.train_dataset_size.get_lock():
        mp_engine.train_dataset_size.value = train_dataset_size

    idx = 0

    global random_mat, perm_mat, M, D, K
    Ks = [12, 16, 20, 24]
    grad_paths = []
    for i in range(len(Ks)):
        Ks[i] = 2**Ks[i]
    for cur_K in Ks:
        grad_paths.append(config["influence"]["grads_path"] + f"_K{cur_K}")
    for grad_path in grad_paths:
        os.makedirs(grad_path, exist_ok=True)

    mp_engine.start_barrier.wait()

    while True:

        while True:
            with mp_engine.train_idx.get_lock(), mp_engine.finished_idx.get_lock():
                idx = mp_engine.train_idx.value
                mp_engine.train_idx.value = (mp_engine.train_idx.value + 1)%train_dataset_size
                if mp_engine.finished_idx[idx] == False:
                    mp_engine.finished_idx[idx] = True
                    break

            time.sleep(0.002)

        if idx >= train_dataset_size:
            break

        try:
            z, t, input_len, real_id = train_loader.dataset[idx]
            z = train_loader.collate_fn([z])
            t = train_loader.collate_fn([t])
            if z.dim() > 2:
                z = torch.squeeze(z, 0)
            if t.dim() > 2:
                t = torch.squeeze(t, 0)

            grad_z_vec = None
            grad_path_name = None
            grad_path_names = []
            if config["influence"]["grads_path"] is not None and len(config["influence"]["grads_path"]) != 0:
                # grad_path_name = config["influence"]["grads_path"] + f"/train_grad_{real_id:08d}.pt"
                for path in grad_paths:
                    grad_path_names.append(path + f"/train_grad_{real_id:08d}.pt")

#             if grad_path_name is not None and os.path.exists(grad_path_name):
#                 try:
#                     grad_z_vec = torch.load(grad_path_name, map_location=model.device)
#                     if isinstance(grad_z_vec, list):
#                         grad_z_vec = torch.cat([x.reshape(-1) for x in grad_z_vec])
#                 except Exception as e:
#                     grad_z_vec = None
#                     print(e)

            if grad_z_vec is None:
                grad_z_vec = grad_z(z, t, input_len, model, gpu=rank)
                vec_list = OPORP_multi_k(grad_z_vec, config, Ks, map_location=f"cuda:{rank}")
                for grad_path_name, vec in zip(grad_path_names, vec_list):
                    torch.save(vec, grad_path_name)

            with mp_engine.finished_cnt.get_lock():
                mp_engine.finished_cnt.value += 1

        except Exception as e:
            with mp_engine.finished_idx.get_lock():
                mp_engine.finished_idx[idx] = False
            print(e)
            raise e


class MPEngine:
    def __init__(self, world_size):
        self.train_idx = Value(c_int, 0)

        self.train_dataset_size = Value(c_int, 0)
        self.start_barrier = Barrier(world_size + 1)

        self.finished_idx = Array(c_bool, [False for _ in range(MAX_DATASET_SIZE)])
        self.finished_cnt = Value(c_int, 0)


def save_OPORP_mp(config):
    gpu_num = torch.cuda.device_count()
    print(f"{gpu_num} GPUs available!")

    threads_per_gpu = 1
    if "n_threads" in config['influence'].keys():
        threads_per_gpu = int(config['influence']['n_threads'])

    if config['influence']['grads_path'] is not None and len(config['influence']['grads_path']) != 0:
        os.makedirs(config['influence']['grads_path'], exist_ok=True)

    mp_engine = MPEngine(gpu_num * threads_per_gpu)

    mp_handler = []
    for i in range(gpu_num):
        for j in range(threads_per_gpu):
            mp_handler.append(mp.Process(target=MP_run_calc_infulence_function, args=(i, gpu_num, i*threads_per_gpu + j, config, mp_engine,)))

    for x in mp_handler:
        x.start()

    train_dataset_size = 0
    while train_dataset_size == 0:
        with mp_engine.train_dataset_size.get_lock():
            train_dataset_size = mp_engine.train_dataset_size.value
        time.sleep(0.01)

    mp_engine.start_barrier.wait()

    finished_cnt = 0
    while finished_cnt != train_dataset_size:
        with mp_engine.finished_cnt.get_lock():
            finished_cnt = mp_engine.finished_cnt.value
        time.sleep(0.1)
        display_progress("Calc. influence function: ", finished_cnt, train_dataset_size, cur_time=time.time())

    # infl = MP_run_get_result(config, mp_engine)
    for x in mp_handler:
        x.terminate()
        # x.join()

    # return infl

