#! /usr/bin/env python3

import torch
import gc
import random
import time
from torch.autograd import grad
from copy import copy
from LLMIF.utils import display_progress
from LLMIF.data_loader import IGNORE_INDEX
import random

def s_test(z_test, t_test, input_len, model, z_loader, gpu=-1, damp=0.01, scale=25.0,
           recursion_depth=5000):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, strochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.

    Arguments:
        z_test: torch tensor, test data points, such as test images
        t_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        z_loader: torch Dataloader, can load the training dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor
        recursion_depth: int, number of iterations aka recursion depth
            should be enough so that the value stabilises.

    Returns:
        h_estimate: list of torch tensors, s_test"""
    v = grad_z(z_test, t_test, input_len, model, gpu)
    h_estimate = tuple(v.copy())


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
        for x, t, _, _ in z_loader:
            if gpu >= 0:
                x, t = x.cuda(gpu), t.cuda(gpu)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                y = model(x)
                y = y.logits
                loss = calc_loss(y, t)
                loss = loss.mean(dim=1)
                params = [ p for p in model.parameters() if p.requires_grad and p.dim() >= 2 ]
                params = params[-10:]
                hv = hvp(loss, params, h_estimate)

            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            gc.collect()

            # Recursively caclulate h_estimate
            h_estimate_temp = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)]

            for _h in h_estimate_temp:
                if torch.isnan(_h).any() == True:
                    print(f"h_estimate has Nan. depth = {i}")
                    min_nan_depth = min(min_nan_depth, i)
                    has_nan = True
                    break
            break

        if has_nan:
            break
        h_estimate = copy(h_estimate_temp)
        # display_progress("Calc. s_test recursions: ", i, recursion_depth, run_time=time.time()-start_time)
    return h_estimate, min_nan_depth


def calc_loss(y, t):
    """Calculates the loss

    Arguments:
        y: torch tensor, input with size (minibatch, nr_of_classes)
        t: torch tensor, target expected by loss of size (0 to nr_of_classes-1)

    Returns:
        loss: scalar, the loss"""
#     # Shift so that tokens < n predict n
#     y = y[..., :-1, :].contiguous()
#     t = t[..., 1:].contiguous()

    bs, _, vocab_size = y.shape
    y = y.reshape(-1, vocab_size)
    t = t.reshape(-1)

    loss = torch.nn.functional.cross_entropy(y, t, reduction='none')
    loss = loss.reshape((bs, -1))
    return loss


def grad_z(z, t, input_len, model, gpu=-1, return_words_loss=False, s_test_vec=None):
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
    model.eval()
    # initialize
    # Only Batch size = 1
    if gpu >= 0:
        z, t = z.cuda(gpu), t.cuda(gpu)
    with torch.cuda.amp.autocast(dtype=torch.float16):
        y = None
        y = model(z)
        y = y.logits
        loss = calc_loss(y, t)[0] # batch_size = 1
        # loss_mean = loss.mean(dim=1)
        loss_mean = loss.mean()
        # Compute sum of gradients from model parameters to loss
        params = [ p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        params = params[-10:]
        # grad_loss = [x.cpu() for x in grad(loss[0], params)]
        words_influence = []
        if return_words_loss == True and s_test_vec is not None:
            for i in range(loss.shape[-1]):
                if t[0, i] == IGNORE_INDEX:
                    # words_losses_result.append(None)
                    words_influence.append(0)
                    continue
                # word_grad = list(x.cpu() for x in word_grad)
                influence = -sum(
                    [
                        torch.sum(k * j).data.cpu().numpy()
                        for k, j in zip(list(grad(loss[i], params, retain_graph=True)), s_test_vec)
                    ])
                words_influence.append(influence)

        grad_loss = list(grad(loss_mean, params))

        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        gc.collect()

        return grad_loss if return_words_loss == False else (grad_loss, words_influence)


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
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = grad(elemwise_products, w)

    return return_grads
