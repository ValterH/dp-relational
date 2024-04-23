"""Contains helper functions used in the code"""

# CDP helpers
# code from https://github.com/terranceliu/dp-query-release

import math
import torch
from typing import List

"""
Functions for converting between concentrated and approximate DP
"""


def cdp_delta(rho, eps, iterations=1000):
    """This function finds the optimal value of delta such that rho-CDP implies
    (eps, delta)-DP.

    Args:
        rho (float): Rho
        eps (float): Epsilon
        iterations (int, optional): Determines the number of iterations to run
            binary search
    Returns:
        delta (float): DELTA.

    (Notes) Delta is calculate by finding the optimal alpha in (1, infinity) via binary
    search. Note that the optimal alpha is at least (1+eps/rho)/2. Thus we only hit
    this constraint when eps<=rho or close to it. This is not an interesting parameter
    regime, as you will inherently get large delta in this regime.
    """
    assert rho >= 0
    assert eps >= 0
    assert iterations > 0
    if rho == 0:  # degenerate case
        return 0

    amin = 1.01  # alpha cannot be due small (numerical stability)
    amax = (eps + 1) / (2 * rho) + 2
    alpha = None
    for _ in range(iterations):
        alpha = (amin + amax) / 2
        derivative = (2 * alpha - 1) * rho - eps + math.log1p(-1.0 / alpha)
        if derivative < 0:
            amin = alpha
        else:
            amax = alpha

    delta = math.exp(
        (alpha - 1) * (alpha * rho - eps) + alpha * math.log1p(-1 / alpha)
    ) / (alpha - 1.0)
    delta = min(delta, 1.0)  # delta <= 1
    return delta


def cdp_eps(rho, delta, iterations=1000):
    """This function finds the smallest value of eps such that rho-CDP implies
    (eps, delta)-DP
    Args:
        rho (float): Rho
        delta (float): Delta
        iterations (int, optional): Determines the number of iterations to run binary
            search
    Returns:
        epsmax (float): Epsilon.
    """
    assert rho >= 0
    assert delta > 0
    assert iterations > 0
    if delta >= 1 or rho == 0:  # if delta>=1 or rho=0, then anything goes
        return 0.0

    epsmin = 0.0  # maintain cdp_delta(rho,eps) >= delta
    epsmax = rho + 2 * math.sqrt(
        rho * math.log(1 / delta)
    )  # maintain cdp_delta(rho,eps) <= delta
    for _ in range(iterations):
        eps = (epsmin + epsmax) / 2
        if cdp_delta(rho, eps) <= delta:
            epsmax = eps
        else:
            epsmin = eps

    return epsmax


def cdp_rho(eps, delta, iterations=1000):
    """This function finds the smallest rho such that rho-CDP implies (eps, delta)-DP
    Args:
        eps (float): Epsilon
        delta (float): Delta
        iterations (int, optional): Determines the number of iterations to run
            binary search
    Returns:
        rhomin (float): Rho.
    """
    assert eps >= 0
    assert delta > 0
    assert iterations > 0
    if delta >= 1:  # if delta >= 1, then anything goes
        return 0.0

    rhomin = 0.0  # maintain cdp_delta(rho,eps) <= delta
    rhomax = eps + 1  # maintain cdp_delta(rhomax,eps) > delta
    for _ in range(iterations):
        rho = (rhomin + rhomax) / 2
        if cdp_delta(rho, eps) <= delta:
            rhomin = rho
        else:
            rhomax = rho

    return rhomin


def get_per_round_privacy_budget(
    epsilon: float, delta: float, num_workloads: int, alpha = None
):
    rho = cdp_rho(epsilon, delta)
    if alpha is None:
        eps0 = 2 * rho / num_workloads
    else:
        eps0 = (2 * rho) / (num_workloads * (alpha**2 + (1 - alpha) ** 2))
    eps0 = math.pow(eps0, 0.5)
    return eps0, rho

"""PyTorch helpers"""

def torch_cat_sparse_coo(tensors: List[torch.Tensor], dim=0, device="cpu"):
    indices_list = []
    values_list = []
    curr_shift = 0
    for tensor in tensors:
        # TODO: dimensionality assertion
        indices = tensor.indices().detach().clone()
        indices[dim, :] += curr_shift
        indices_list.append(indices)
        values_list.append(tensor.values())
        curr_shift += tensor.size(dim)
    
    new_size = list(tensors[0].size())
    new_size[dim] = curr_shift
    
    return torch.sparse_coo_tensor(torch.cat(indices_list, dim=1), torch.cat(values_list), size=new_size, device=device).coalesce()