#! /usr/bin/env python3

import torch
import gc
import random
import time
from torch.autograd import grad
import torch
from copy import copy
from LLMIF.utils import display_progress, print_gpu_usage
from LLMIF.data_loader import IGNORE_INDEX
import random
from torch.utils.data import default_collate
import torch.nn.functional as F

params = None

def get_params(model, create_if_not_exist=True):
    global params
    if params is not None:
        return params
    if create_if_not_exist == False:
        return None 

    params = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.dim() >= 2:
            # print(f"{name}: ", p.shape)
            params.append(p)
    return params

def normalize(x):
    return F.normalize(x, p=2, dim=0)
    # return x

def pad(x):
    # pad
    D = len(x)
    K = 2**24
    new_D = ((D - 1)//K + 1)*K
    x = F.pad(x, (0, new_D - D), "constant", 0)
    return x

def reshape(x):
    # reshape
    step = 421527552
    n_step = (len(x) - 1)//step + 1
    x = x.reshape((n_step, -1))
    return x


def s_test(z_test, t_test, input_len, model, z_loader, gpu=-1, damp=0.01, scale=25.0,
           recursion_depth=5000, need_reshape=True):
    # model.eval()
    params = get_params(model)

    v = grad_z(z_test, t_test, input_len, model, gpu, need_reshape=False)
    h_estimate = copy(v)


    ################################
    # TODO: Dynamically set the recursion depth so that iterations stops
    # once h_estimate stabilises
    ################################
    min_nan_depth = recursion_depth
    has_nan = False
    for i in range(recursion_depth):
        # take just one random sample from training dataset
        # easiest way to just use the DataLoader once, break at the end of loop
        #########################
        # TODO: do x, t really have to be chosen RANDOMLY from the train set?
        #########################
        start_time = time.time()
        # for x, t, _, _ in z_loader:
        # idx = random.randint(0, len(z_loader.dataset) - 1)
        idx = random.randint(0, len(z_loader) - 1)
        x, t, _, _ = z_loader[idx]
        x = default_collate([x])
        t = default_collate([t])


        if gpu >= 0:
            x, t = x.cuda(gpu), t.cuda(gpu)

        y = model(x)
        y = y.logits
        loss = calc_loss(y, t)


        hv = hvp(loss, params, h_estimate)

        h_estimate_temp = v + (1 - damp) * h_estimate - hv / scale

        if torch.isnan(h_estimate_temp).any() == True:
            print(f"h_estimate has Nan. depth = {i}")
            min_nan_depth = min(min_nan_depth, i)
            has_nan = True
            break

        if has_nan:
            break
        h_estimate = copy(h_estimate_temp)
        # display_progress("Calc. s_test recursions: ", i, recursion_depth, run_time=time.time()-start_time)

    h_estimate = pad(h_estimate)
    if need_reshape == True:
        h_estimate = reshape(h_estimate)

    # h_estimate = torch.cat([x.reshape(-1) for x in h_estimate])
    return h_estimate, min_nan_depth


def calc_loss(y, t):
    """Calculates the loss

    Arguments:
        y: torch tensor, input with size (minibatch, nr_of_classes)
        t: torch tensor, target expected by loss of size (0 to nr_of_classes-1)

    Returns:
        loss: scalar, the loss"""
    # Shift so that tokens < n predict n
    # y = y[..., :-1, :].contiguous()
    # t = t[..., 1:].contiguous()
    # y = y[..., :-1, :]
    # t = t[..., 1:]

    bs, _, vocab_size = y.shape
    y = y.reshape(-1, vocab_size)
    t = t.reshape(-1)

    loss = torch.nn.functional.cross_entropy(y, t)

    return loss


def grad_z(z, t, input_len, model, gpu=-1, return_words_loss=False, s_test_vec=None, need_reshape=True, use_deepspeed=False):
    """Calculates the gradient z. One grad_z should be computed for each
    training sample.

    Arguments:
        z: torch tensor, training data points
            e.g. an image sample (batch_size, 3, 256, 256)
        t: torch tensor, training data labels
        model: torch NN, model used to evaluate the dataset
        gpu: int, device id to use for GPU, -1 for CPU

    Returns:
        grad_z: list of torch tensor, containing the gradients
            from model parameters to loss"""

    z = default_collate([z])
    t = default_collate([t])
    if z.dim() > 2:
        z = torch.squeeze(z, 0)
    if t.dim() > 2:
        t = torch.squeeze(t, 0)

    if gpu >= 0:
        z, t = z.cuda(gpu), t.cuda(gpu)
    # print_gpu_usage("move data to gpu")

    y = model(z)
    y = y.logits
    loss = calc_loss(y, t) # batch_size = 1
    # print_gpu_usage("get loss")

    params = get_params(model, create_if_not_exist=False)
    if params is not None:
        grad_loss = torch.cat([x.reshape(-1) for x in list(grad(loss, params))])
        model.zero_grad(set_to_none=True)
    else:
        grad_loss = None
        if use_deepspeed == True:
            model.backward(loss)
            # grad_loss = torch.cat([normalize(model.optimizer.get_fp32_grad_for_param(p).reshape(-1)) for p in model.parameters() if p.requires_grad == True])
            grad_loss = torch.cat([normalize(model.optimizer.fp32_partitioned_groups_flat[group_idx].grad.narrow(0, dest_offset, num_elements)) \
                    for group_idx, dest_offset, num_elements in model.optimizer.grad_position.values()])
            model.optimizer.zero_grad()
        else:
            loss.backward()
            grad_loss = torch.cat([normalize(p.grad.reshape(-1)) for p in model.parameters() if p.grad is not None])
            model.zero_grad(set_to_none=True)

#         params = get_params(model)
#         grad_loss = torch.cat([x.reshape(-1) for x in list(grad(loss, params))])
#         model.zero_grad(set_to_none=True)
    # print_gpu_usage("after backword")

    grad_loss = pad(grad_loss)
    if need_reshape == True:
        grad_loss = reshape(grad_loss)

    # model.zero_grad(set_to_none=True)
    # gc.collect()
    # torch.cuda.empty_cache()
    # print_gpu_usage("after collect")

    return grad_loss


def hvp(y, w, v):
    """Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.

    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian

    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.

    Raises:
        ValueError: `y` and `w` have a different length."""
    # if len(w) != len(v):
    #     raise(ValueError("w and v must have the same length."))

    # First backprop
    # first_grads = grad(y, w, create_graph=True)
    first_grads = torch.cat([x.reshape(-1) for x in grad(y, w, create_graph=True)])

    # Elementwise products
    # elemwise_products = 0
    # for grad_elem, v_elem in zip(first_grads, v):
    #     elemwise_products += torch.sum(grad_elem * v_elem)
    elemwise_products = torch.sum(first_grads * v)


    # Second backprop
    # return_grads = grad(elemwise_products, w)
    return_grads = torch.cat([x.reshape(-1) for x in grad(elemwise_products, w)])

    return return_grads
