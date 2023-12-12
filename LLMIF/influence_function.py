#! /usr/bin/env python3

import os
import torch
import time
import datetime
import numpy as np
import copy
import logging

from pathlib import Path
from LLMIF.calc_inner import s_test, grad_z
from LLMIF.utils import save_json, display_progress

import numpy as np

IGNORE_INDEX = -100

def calc_s_test(model, test_loader, train_loader, save=False, gpu=-1,
                damp=0.01, scale=25, recursion_depth=5000, r=1, start=0):
    """Calculates s_test for the whole test dataset taking into account all
    training data images.

    Arguments:
        model: pytorch model, for which s_test should be calculated
        test_loader: pytorch dataloader, which can load the test data
        train_loader: pytorch dataloader, which can load the train data
        save: Path, path where to save the s_test files if desired. Omitting
            this argument will skip saving
        gpu: int, device id to use for GPU, -1 for CPU (default)
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
        start: int, index of the first test index to use. default is 0

    Returns:
        s_tests: list of torch vectors, contain all s_test for the whole
            dataset. Can be huge.
        save: Path, path to the folder where the s_test files were saved to or
            False if they were not saved."""
    if save and not isinstance(save, Path):
        save = Path(save)
    if not save:
        logging.info("ATTENTION: not saving s_test files.")

    s_tests = []
    for i in range(start, len(test_loader.dataset)):
        z_test, t_test, input_len = test_loader.dataset[i]
        z_test = test_loader.collate_fn([z_test])
        t_test = test_loader.collate_fn([t_test])

        s_test_vec = calc_s_test_single(model, z_test, t_test, input_len, train_loader,
                                        gpu, damp, scale, recursion_depth, r)

        if save:
            s_test_vec = [s.cpu() for s in s_test_vec]
            torch.save(
                s_test_vec,
                save.joinpath(f"{i}_recdep{recursion_depth}_r{r}.s_test"))
        else:
            s_tests.append(s_test_vec)
        display_progress(
            "Calc. z_test (s_test): ", i-start, len(test_loader.dataset)-start)

    return s_tests, save


def calc_s_test_single(model, z_test, t_test, input_len, train_loader, gpu=-1,
                       damp=0.01, scale=25, recursion_depth=5000, r=1):
    """Calculates s_test for a single test image taking into account the whole
    training dataset. s_test = invHessian * nabla(Loss(test_img, model params))

    Arguments:
        model: pytorch model, for which s_test should be calculated
        z_test: test image
        t_test: test image label
        train_loader: pytorch dataloader, which can load the train data
        gpu: int, device id to use for GPU, -1 for CPU (default)
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.

    Returns:
        s_test_vec: torch tensor, contains s_test for a single test image"""

    min_nan_depth = recursion_depth
    res, nan_depth = s_test(z_test, t_test, input_len, model, train_loader,
                 gpu=gpu, damp=damp, scale=scale,
                 recursion_depth=recursion_depth)
    # res = [x.data.cpu() for x in res]
    min_nan_depth = min(min_nan_depth, nan_depth)
    for i in range(1, r):
        start_time = time.time()
        cur, nan_depth = s_test(z_test, t_test, input_len, model, train_loader,
               gpu=gpu, damp=damp, scale=scale,
               recursion_depth=recursion_depth)
        # cur = [x.data.cpu() for x in cur]
        # res = [a + c for a, c in zip(res, cur)]
        res = res + cur
        min_nan_depth = min(min_nan_depth, nan_depth)
        # display_progress("Averaging r-times: ", i, r, run_time=time.time()-start_time)

    if min_nan_depth != recursion_depth:
        print(f"Warning: get Nan value after depth {min_nan_depth}, current recursion_depth = {min_nan_depth}")
    # res = [a / r for a in res]
    res = res/r

    return res


def pad_process(input_ids, labels):
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    return input_ids, labels


def calc_influence_single(model, train_loader, test_loader, test_id_num, gpu,
                          recursion_depth, r, s_test_vec=None,
                          time_logging=False):
    """Calculates the influences of all training data points on a single
    test dataset image.

    Arugments:
        model: pytorch model
        train_loader: DataLoader, loads the training dataset
        test_loader: DataLoader, loads the test dataset
        test_id_num: int, id of the test sample for which to calculate the
            influence function
        gpu: int, identifies the gpu id, -1 for cpu
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
        s_test_vec: list of torch tensor, contains s_test vectors. If left
            empty it will also be calculated

    Returns:
        influence: list of float, influences of all training data samples
            for one test sample
        harmful: list of float, influences sorted by harmfulness
        helpful: list of float, influences sorted by helpfulness
        test_id_num: int, the number of the test dataset point
            the influence was calculated for"""
    # Calculate s_test vectors if not provided
    if not s_test_vec:
        z_test, t_test, input_len = test_loader.dataset[test_id_num]
        # t_test = t_test[input_len]
        z_test = test_loader.collate_fn([z_test])
        t_test = test_loader.collate_fn([t_test])
        s_test_vec = calc_s_test_single(model, z_test, t_test, input_len, train_loader,
                                        gpu, recursion_depth=recursion_depth,
                                        r=r)


    # Calculate the influence function
    train_dataset_size = len(train_loader.dataset)
    influences = [0 for _ in range(train_dataset_size)]
    # for i in range(train_dataset_size):
    i = 0
    batch_size = 1
    for i in range(0, train_dataset_size, batch_size):
        z, t, input_len, real_index = train_loader.dataset[i:min(train_dataset_size, i + batch_size)]
        # z, t = pad_process(z, t)
        z = train_loader.collate_fn(z)
        t = train_loader.collate_fn(t)
        start_time = time.time()
        if time_logging:
            time_a = datetime.datetime.now()
        if z.dim() > 2:
            z = torch.squeeze(z, 0)
        if t.dim() > 2:
            t = torch.squeeze(t, 0)
        grad_z_vecs = grad_z(z, t, input_len, model, gpu=gpu)
        for real_idx, grad_z_vec in zip(real_index, grad_z_vecs):
            tmp_influence = -sum(
                [
                    ####################
                    # TODO: potential bottle neck, takes 17% execution time
                    # torch.sum(k * j).data.cpu().numpy()
                    ####################
                    torch.sum(k * j).data.cpu().numpy()
                    for k, j in zip(grad_z_vec, s_test_vec)
                ]) / train_dataset_size
            # influences.append(tmp_influence)
            influences[real_idx] = tmp_influence
        end_time = time.time()
        i += z.shape[0]
        display_progress("Calc. influence function: ", i, train_dataset_size, run_time=end_time-start_time, fix_zero_start=False)

    helpful = np.argsort(influences)
    harmful = helpful[::-1]

    return influences, harmful.tolist(), helpful.tolist(), test_id_num


def get_dataset_sample_ids_per_class(class_id, num_samples, test_loader,
                                     start_index=0):
    """Gets the first num_samples from class class_id starting from
    start_index. Returns a list with the indicies which can be passed to
    test_loader.dataset[X] to retreive the actual data.

    Arguments:
        class_id: int, name or id of the class label
        num_samples: int, number of samples per class to process
        test_loader: DataLoader, can load the test dataset.
        start_index: int, means after which x occourance to add an index
            to the list of indicies. E.g. if =3, then it would add the
            4th occourance of an item with the label class_nr to the list.

    Returns:
        sample_list: list of int, contains indicies of the relevant samples"""
    #######
    # NOTE / TODO: here's optimisation potential. We are currently searching
    # for the x+1th sample and when that's found we cancel the loop. we could
    # stop after finding the x'th picture (start_index + num_samples)
    #######
    sample_list = []
    img_count = 0
    for i in range(len(test_loader.dataset)):
        _, t, input_len = test_loader.dataset[i]
        if class_id == t:
            img_count += 1
            if (img_count > start_index) and \
                    (img_count <= start_index + num_samples):
                sample_list.append(i)
            elif img_count > start_index + num_samples:
                break

    return sample_list


def calc_img_wise(config, model, train_loader, test_loader, gpu=-1):
    """Calculates the influence function one test point at a time. Calcualtes
    the `s_test` and `grad_z` values on the fly and discards them afterwards.

    Arguments:
        config: dict, contains the configuration from cli params"""
    influences_meta = copy.deepcopy(config)
    test_sample_num = len(test_loader.dataset)
    test_start_index = 0
    outdir = Path(config['outdir'])
    outdir.mkdir(exist_ok=True, parents=True)

    test_dataset_iter_len = len(test_loader.dataset)
    # Set up logging and save the metadata conf file
    logging.info(f"Running on: {test_sample_num} samples.")

    influences_path = outdir.joinpath(f"influence_results_{test_start_index}_"
                                      f"{test_sample_num}.json")
    influences = {}
    # Main loop for calculating the influence function one test sample per
    # iteration.
    for j in range(test_dataset_iter_len):
        # If we calculate evenly per class, choose the test img indicies
        # from the sample_list instead
        if test_sample_num and test_start_index:
            if j >= len(sample_list):
                logging.warn("ERROR: the test sample id is out of index of the"
                             " defined test set. Jumping to next test sample.")
                next
            i = sample_list[j]
        else:
            i = j

        start_time = time.time()
        influence, harmful, helpful, _ = calc_influence_single(
            model, train_loader, test_loader, test_id_num=i, gpu=gpu,
            recursion_depth=config['recursion_depth'], r=config['r_averaging'])
        end_time = time.time()

        ###########
        # Different from `influence` above
        ###########
        influences[str(i)] = {}
        _, label, input_len = test_loader.dataset[i]
        infl = [x.tolist() for x in influence]
        # influences[str(i)]['influence'] = infl
        influences[str(i)]['test_data'] = test_loader.dataset.list_data_dict[i]
        indep_index = sorted(range(len(infl)), key=lambda i: abs(infl[i]))[:100]
        helpful = helpful[:100]
        harmful = harmful[:100]
        influences[str(i)]['helpful'] = helpful
        influences[str(i)]['harmful'] = harmful
        influences[str(i)]['indep'] = indep_index
        influences[str(i)]['indep_infl'] = [infl[x] for x in indep_index]
        influences[str(i)]['helpful_infl'] = [infl[x] for x in helpful]
        influences[str(i)]['harmful_infl'] = [infl[x] for x in harmful]

       
        if i == 0:
            influences_path = save_json(influences, influences_path, unique_fn_if_exists=True)
        else:
            influences_path = save_json(influences, influences_path, overwrite_if_exists=True)

        display_progress("Test samples processed: ", j, test_dataset_iter_len, new_line=True, run_time=end_time-start_time)
        print("\n\n" + "---" * 20 + "\n\n")

    print("path:", influences_path)

    return influences, harmful, helpful


