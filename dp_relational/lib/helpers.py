"""Contains helper functions used in the code"""

# CDP helpers
# code from https://github.com/terranceliu/dp-query-release

import math
import torch
from typing import List

import numpy as np

import mosek
import mosek.fusion

from torch.distributions import OneHotCategorical

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

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

def unbiased_sample(b, m):
    # nudge for finite precision errors
    b[-1] = m - (np.sum(b) - b[-1])
    if m == 0:
        return np.zeros(b.shape, dtype=np.int_)
    if m == 1:
        res = np.zeros(b.shape, dtype=np.int_)
        res[np.searchsorted(np.cumsum(b), np.random.rand())] = 1
        return res
    """ Inputs a vector of probabilities b, and outputs a one-hot vector indicating which were selected """
    if len(b) == m: # all are ones
        return np.ones(b.shape)
    
    b_cumsum = np.cumsum(b)
    indexes = [0]
    prev_val = 0
    while indexes[-1] != len(b):
        next_idx = np.searchsorted(b_cumsum, prev_val + 1, side='right') - 1
        indexes.append(next_idx + 1)
        prev_val = b_cumsum[next_idx]
        
    #print(b, m, indexes)
    indexes = np.array(indexes)
    
    # we now have a list of indices to sample from
    # get their sizes and choose the one to exclude
    item_sizes = indexes[1:] - indexes[:-1]
    prob_sizes = b_cumsum[indexes[1:] - 1]
    prob_sizes[1:] -= b_cumsum[indexes[1:-1] - 1]
    p_sizes_cumsum = np.cumsum(prob_sizes)
    
    # parallelized sampling
    samples = np.random.rand(*prob_sizes.shape) * prob_sizes
    samples[1:] += p_sizes_cumsum[:-1]
    indexes_sampled = np.searchsorted(b_cumsum, samples)
    
    result = np.zeros(b.shape, dtype=np.int_)
    result[indexes_sampled] = 1
    # now go and delete some of them
    deletion_probs = 1 - prob_sizes
    
    num_to_delete = len(prob_sizes) - m
    deleted_indices = np.nonzero(unbiased_sample(deletion_probs, num_to_delete))[0]
    #print("delind", deleted_indices)
    result[indexes_sampled[deleted_indices]] = 0
    # print(b, m, "res", result)
    return result

@torch.no_grad()
def unbiased_sample_torch(b_in, m, device="cpu"):
    assert m >= 0

    b_in = torch.clamp(b_in, 0, 1)
    n = b_in.size(dim=0)
    
    """ Inputs a vector of probabilities b, and outputs a one-hot vector indicating which were selected """
    
    # No shuffling is required in any of the cases below!
    if m == 0:
        return torch.zeros(*b_in.shape, dtype=torch.int, device=device)
    if m == 1:
        idx = torch.multinomial(b_in, 1)
        res = torch.zeros(*b_in.shape, dtype=torch.int, device=device)
        res[idx[0]] = 1
        return res
    if len(b_in) == m: # all are ones
        return torch.ones(*b_in.shape, dtype=torch.int, device=device)
    
    # Create a shuffle of the bs
    shuffle = torch.randperm(n)
    b = b_in[shuffle]

    # there is technically an annoying projection step here...
    # we need to:
    # - bring b into a clamped range
    # - make sure that b's sum is correct...
    b = torch.clamp(b, 0, 1) # correctly clamped
    b_sum = torch.sum(b)

    if b_sum > float(m):
        b *= (float(m) / b_sum)
    
    b_cumsum = torch.cumsum(b, dim=0)
    indexes = [0]
    prev_val = 0
    while indexes[-1] != len(b):
        next_idx = torch.searchsorted(b_cumsum, prev_val + 1, side='right') - 1
        if (next_idx + 1) == indexes[-1]:
            next_idx += 1
        # print(next_idx, len(b), b_cumsum[next_idx], b_cumsum[next_idx + 1], b_cumsum[next_idx + 2])
        indexes.append(next_idx + 1)
        prev_val = b_cumsum[next_idx]
        
    #print(b, m, indexes)
    indexes = torch.tensor(indexes, dtype=int, device=device)
    
    # we now have a list of indices to sample from
    # get their sizes and choose the one to exclude
    prob_sizes = b_cumsum[indexes[1:] - 1]
    prob_sizes[1:] -= b_cumsum[indexes[1:-1] - 1]
    p_sizes_cumsum = torch.cumsum(prob_sizes, dim=0)
    
    # parallelized sampling
    samples = torch.rand(*prob_sizes.shape, device=device) * prob_sizes
    samples[1:] += p_sizes_cumsum[:-1]
    indexes_sampled = torch.searchsorted(b_cumsum, samples)
    # so the issue is that sometimes the indexs will be outside of the range
    indexes_sampled = torch.clamp(indexes_sampled, indexes[:-1], indexes[1:] - 1)
    
    result = torch.zeros(*b.shape, dtype=torch.int, device=device)
    result[indexes_sampled] = 1
    sampled_sum = torch.sum(result)
    # now go and delete some of them
    deletion_probs = 1 - prob_sizes
    
    num_to_delete = len(prob_sizes) - m
    deleted_indices = torch.nonzero(unbiased_sample_torch(deletion_probs, num_to_delete, device=device), as_tuple=True)[0]
    #print("delind", deleted_indices)
    result[indexes_sampled[deleted_indices]] = 0
    deleted_sum = torch.sum(result)
    
    unshuf_order = torch.zeros_like(shuffle)
    unshuf_order[shuffle] = torch.arange(n)
    result[shuffle] = result.clone()
    # print(b, m, "res", result)

    if (torch.sum(result) != m):
        print(b.shape)
        print("m", m)
        print("num_to_delete", num_to_delete)
        print("prob_sizes_len", len(prob_sizes))
        print("s_sum", sampled_sum)
        print("d_sum", deleted_sum)
        raise RuntimeError()
    # print(b, m, "res", result)
    return result

def mirror_descent_torch(Q, b, a, step_size = 0.01, T_mirror = 50):
    # b is a vector whose sum is 1
    assert isinstance(Q, torch.Tensor)
    assert isinstance(b, torch.Tensor)
    assert isinstance(a, torch.Tensor)

    def mirror_descent_update(x, gradient):
    # Choose a suitable step size (e.g., 1/D)

        # Perform the Mirror Descent update
        numer = x * torch.exp(-step_size * gradient)
        denom = torch.sum(x * torch.exp(-step_size * gradient))

        updated_x = numer / denom

        return updated_x

    # Function to compute the gradient of the objective function ||Qb - a||_2^2
    def gradient(Q, b, a):
        return 2 * torch.matmul(Q.T, (torch.matmul(Q, b) - a))

    iters = 0

    # Mirror Descent iterations
    while iters < T_mirror:
        iters += 1
        # Compute the gradient of the objective function
        grad = gradient(Q, b, a)
        # Update using Mirror Descent
        b = mirror_descent_update(b, grad)
        # print("B update step: ", b)
    return b

def mosek_optimize(Q, a, m, N, ):
    # get everything into numpy to work with mosek
    Q = Q.to_dense().numpy(force=True).astype(np.float64)
    a = a.numpy(force=True).astype(np.float64)
    
    M = mosek.fusion.Model("BBSLS")
    
    b = M.variable("b", N, mosek.fusion.Domain.inRange(0.0, 1.0))
    
    # The bound on the norm of the residual
    e = M.variable("e")
    r = mosek.fusion.Expr.sub(a, mosek.fusion.Expr.mul(Q,b))
    # t \geq |r|_2
    M.constraint(mosek.fusion.Expr.vstack(e, r), mosek.fusion.Domain.inQCone())
    M.constraint(mosek.fusion.Expr.sum(b), mosek.fusion.Domain.equalsTo(m))
    
    M.objective(mosek.fusion.ObjectiveSense.Minimize, e)
    M.solve()
    
    res = np.array(M.getVariable("b").level()) #, M.getVariable("e").level()
    M.dispose()
    return res


def GM_torch(inp, rho, n_relationship_orig):
    rand = torch.normal(0.0, (np.sqrt(2)/(n_relationship_orig * np.sqrt(rho))).item(), inp.size())
    return inp + rand
def GM_torch_noise(inp, stddev):
    rand = torch.normal(0.0, stddev, inp.size())
    return inp + rand
def expround_torch(b, m=None, device="cpu"):
    if m is None:
        m = int(torch.sum(b).numpy(force=True))
    b_fixed = b - ((torch.min(b) < 0) * torch.min(b))
    X_dist = torch.distributions.exponential.Exponential(1 / (b_fixed + 1e-15)) # tiny piece to prevent it from giving div0 errors
    X = X_dist.sample()
    # finding the index of top m elements
    values, indices = torch.topk(X, m)
    indices = indices.unsqueeze(0)
    
    bu = torch.sparse_coo_tensor(indices, torch.ones([m], device=device), b.size()).coalesce()
    return bu

def get_relationships_from_sparse(qm, b_round):
    b_round = b_round.coalesce()
    vals = b_round.values()
    indices = b_round.indices()
    idxes_nz = indices[0, torch.nonzero(vals)].cpu()
    
    relationships = np.squeeze(idxes_nz.numpy())
    
    table2_nums = relationships % qm.n_syn2
    table1_nums = (relationships - table2_nums) // qm.n_syn2
    
    return table1_nums, table2_nums

def largest_singular_value(sparse_tensor, tolerance=1e-6, max_iterations=1000):
    """
    Calculate the largest singular value of a sparse tensor using power iteration.
    
    Parameters:
    sparse_tensor (torch.sparse_coo_tensor): Input sparse tensor.
    tolerance (float): Desired error tolerance for convergence.
    max_iterations (int): Maximum number of iterations to perform.
    
    Returns:
    float: Approximation of the largest singular value.
    """
    # Ensure the input is a sparse tensor
    assert sparse_tensor.is_sparse, "Input tensor must be sparse"

    # Initialize a random vector
    N = sparse_tensor.size(1)
    b_k = torch.randn(N, device=sparse_tensor.device)
    b_k = b_k / torch.norm(b_k)  # Normalize the initial vector

    # Power iteration method to find the largest singular value
    singular_value_old = 0.0
    for _ in range(max_iterations):
        # Perform the matrix multiplication
        b_k1 = torch.sparse.mm(sparse_tensor, b_k.view(-1, 1)).view(-1)
        b_k1_norm = torch.norm(b_k1)
        b_k1 = b_k1 / b_k1_norm

        # Perform the matrix multiplication with the transpose
        b_k = torch.sparse.mm(sparse_tensor.t(), b_k1.view(-1, 1)).view(-1)
        b_k_norm = torch.norm(b_k)
        b_k = b_k / b_k_norm

        # Check for convergence
        if torch.abs(b_k_norm - singular_value_old) < tolerance:
            break
        singular_value_old = b_k_norm

    return b_k_norm.item()

def gradient(Q, b, a, m):
    """ Helper function to compute the gradient of the objective function."""
    c = torch.sparse.mm(Q, b) / m ## 
    g = torch.sparse.mm(Q.T, (-a)[:, None] + c) * 2 / m

    return g

# Code source: https://arxiv.org/pdf/1101.6081
def projsplx_torch(b):
    # Step 2: sort in ascending order
    sorted_b, sorted_indices = torch.sort(b)
    cumsum_sorted_b = torch.cumsum(sorted_b, dim=0)
    N = len(b)
    i = N - 1
    t_hat = None
    
    fracs = ((cumsum_sorted_b[-1] - cumsum_sorted_b[:-1]) - 1) / (N - (torch.arange(1, N, device=b.device)))
    vals = sorted_b[:-1]
    # print(fracs, vals)
    slic = (fracs >= vals).nonzero()
    if len(slic) == 0:
        t_hat = (cumsum_sorted_b[-1] - 1) / N
    else:
        t_hat = fracs[slic[-1]]
    #print("b", b.size())
    #print("t_hat", t_hat)
    res = torch.clip(b - t_hat, min=0, max=1)
    #print("res", res.size())
    return res

def one_to_many_project(b, row_size):
    b_cp = b.clone()
    for x in range(0, len(b), row_size):
        b_cp[x:x+row_size] = projsplx_torch(torch.flatten(b[x:x+row_size]))[:, None]
    return b_cp

def one_to_many_sample(b, row_size):
    b_cp = b.clone()
    for x in range(0, len(b), row_size):
        dist = OneHotCategorical(b[x:x+row_size])
        b_cp[x:x+row_size] = dist.sample()
    return b_cp

def pgd_optimize_one_to_many(Q, b, a, m, T, row_size):
    # 1. Calculate the learning rate
    l = largest_singular_value(Q)
    L = (l * l) * 2 / (m*m)
    lr = 1/L
    
    # 2. Perform the PGD
    for t in range(T):
        print(t)
        # 2a. Identify the gradient of the solution
        g = gradient(Q, b, a, m)
        # 2b. Iterate the b value and correctly project it
        # TODO: this may be able to avoid sparsity issues!!!
        b = -(lr * g) + b 
        b = one_to_many_project(b, row_size)
        # b = b[:, None]
    
    return b