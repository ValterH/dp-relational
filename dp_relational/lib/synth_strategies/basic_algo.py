import numpy as np
import torch
from ..qm import QueryManager, QueryManagerBasic, QueryManagerTorch
from ..helpers import cdp_delta, cdp_eps, cdp_rho, get_per_round_privacy_budget, torch_cat_sparse_coo


from ..helpers import expround_torch, GM_torch_noise, GM_torch, mosek_optimize, mirror_descent_torch
from ..helpers import unbiased_sample_torch, unbiased_sample, display_top
from ..helpers import get_relationships_from_sparse

from tqdm import tqdm

import random

import gc

def learn_relationship_vector_basic(qm: QueryManagerBasic, epsilon_relationship=1.0, T=100,
                                   delta_relationship = 1e-5, verbose=False):
    Q = qm.Q
    n_relationship_orig = qm.n_relationship_orig
    n_relationship_synt = qm.n_relationship_synth
    
    def mirror_descent(Q, b, a, step_size = 0.01, T_mirror = 50):
        # b is a vector whose sum is 1
        Q = np.array(Q)
        b = np.array(b)
        a = np.array(a)
        
        assert len(Q[0]) == len(b)
        assert len(Q) == len(a)

        def mirror_descent_update(x, gradient):
        # Choose a suitable step size (e.g., 1/D)

            # Perform the Mirror Descent update
            numer = x * np.exp(-step_size * gradient)
            denom = np.sum(x * np.exp(-step_size * gradient))

            updated_x = numer / denom

            return updated_x

        # Function to compute the gradient of the objective function ||Qb - a||_2^2
        def gradient(Q, b, a):
            return 2.0 * Q.T @ (np.matmul(Q, b) - a)

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
    def GM(inp, rho):
        num = len(inp)
        out = []
        for i in range(num):
            val = inp[i] + np.random.normal(0, np.sqrt(2)/(n_relationship_orig * np.sqrt(rho)))
            out.append(val)
        return out
    def expround(b):
        N = len(b)
        m = np.sum(b)
        X = np.zeros(N)
        for i in range(N):
            X[i] = np.random.exponential(b[i])
        # finding the index of top m elements
        idx = sorted(range(N), key=lambda i: X[i])[-int(m):]
        bu = np.zeros(N)
        for i in idx:
            bu[i] = 1
        return bu

    # number of workloads to compute per iteration
    num_workload_ite = 2

    epsilon_relationship = epsilon_relationship/(qm.rel_dataset.dmax * num_workload_ite)

    # convert to RDP
    rho_rel = cdp_rho(epsilon_relationship, delta_relationship)

    # privacy budget per iteration
    per_round_rho_rel = (rho_rel / 0.0001) if T == 0 else rho_rel / T

    # intialization
    unselected_workload = [i for i in range(len(qm.workload_names))]
    Q_set = []
    noisy_ans = []
    b = np.ones(qm.n_syn1 * qm.n_syn2) / (qm.n_syn1 * qm.n_syn2)

    for t in tqdm(range(T)):

        # TODO: use exponential mechanism!!!
        # Here I randomly choose 2 workloads
        select_workload = random.choices(unselected_workload, k=num_workload_ite)
        unselected_workload = [i for i in unselected_workload if i not in select_workload]

        for i in select_workload:

            curr_workload = qm.workload_names[i]

            ind_low, ind_high = qm.workload_dict[curr_workload]['range_low'], qm.workload_dict[curr_workload]['range_high']

            curr_ans = qm.true_ans[ind_low:(ind_high+1)]

            noisy_curr_ans = GM(curr_ans, per_round_rho_rel)

            for row in noisy_curr_ans:
                noisy_ans.append(row)

            for row in Q[ind_low:(ind_high+1)]:
                Q_set.append(row.tolist())

        b = mirror_descent(Q_set, b, noisy_ans, step_size = 0.01, T_mirror = 50)

    b = b * n_relationship_synt
    b_round = expround(b)
    b_round = b_round.reshape(qm.n_syn1, qm.n_syn2)
    
    return b_round
