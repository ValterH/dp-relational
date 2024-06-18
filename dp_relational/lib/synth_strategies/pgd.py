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

import time

@torch.no_grad()
def learn_relationship_pgd(qm: QueryManagerTorch, epsilon_relationship=1.0, T=100,
                                            delta_relationship = 1e-5, iter_cb=lambda *args: None,
                                            k_new_queries=3, k_choose_from=300, exp_mech_alpha=0.2, verbose=False, device="cpu", eta = 0.01):
    """Full final algorithm described in the paper, with the following features implemented:"""
    """ - Exponential mechanism to choose queries from the set """
    """ - Unbiased estimator (if this actually runs in time)"""
    """ - MOSEK solver for queries"""
    
    """
    Information on parameters:
    qm: a query manager to produce query matrices
    epsilon_relationship: the privacy budget allocated to the relational table
    delta_relationship: the delta privacy parameter
    T: the number of iterations to run
    subtable_size: the size of the subtable to generate. This is related to the alpha parameter by sqrt(subtable_size / (n_syn1 * n_syn2))
    queries_to_reuse: the number of queries that we will evaluate in each iteration. Set to None to run all.
    k_new_queries: number of new queries to add to our set in each iteration
    k_choose_from: number of queries to evaluate when running the exponential mechanism
    """
    assert k_new_queries <= k_choose_from
    assert 0 < exp_mech_alpha < 1
    
    n_relationship_synt = qm.n_relationship_synth
    m = n_relationship_synt # alias
    m_privacy = qm.n_relationship_orig
    
    assert n_relationship_synt < qm.n_syn1 * qm.n_syn2
    
    # convert to RDP
    rho_rel = cdp_rho(epsilon_relationship, delta_relationship)
    
    # privacy parameter
    epsilon0 = np.sqrt((2 * rho_rel) / (k_new_queries * T)) if T != 0 else 100000
    
    # exponential mechanism factor: product before the softmax
    exp_mech_factor = np.sqrt(exp_mech_alpha) * epsilon0 * (m_privacy / qm.rel_dataset.dmax)
    
    # gaussian mechanism standard deviation
    gm_stddev = (np.sqrt(2) / (np.sqrt(1 - exp_mech_alpha) * epsilon0)) * (qm.rel_dataset.dmax / m_privacy)

    # intialization
    unselected_workload = list(range(len(qm.workload_names)))



    ## equivalent to the iterative projection algorithm
    def optimal_project_to_simplex(v, m):
        # Step 1: Clip the vector to ensure it is between 0 and 1
        v = np.clip(v, 0, 1)
        
        # Step 2: Check if the sum is greater than m, if so scale it down
        sum_v = np.sum(v)
        if sum_v > m:
            return v * (m / sum_v)
        
        # Step 3: If the sum is less than m, sort the vector in descending order
        sorted_indices = np.argsort(-v)
        sorted_v = v[sorted_indices]
        
        # Step 4: Find the first index i that satisfies the conditions
        cumsum_sorted_v = np.cumsum(sorted_v)
        N = len(v)
        for i in range(N):
            if sorted_v[i] > 0 and (m - (i + 1)) / (cumsum_sorted_v[-1] - cumsum_sorted_v[i]) <= 1:
                break
        
        # Step 5: Set v_1, ..., v_i to 1
        sorted_v[:i + 1] = 1
        
        # Step 6: Scale remaining elements
        if i + 1 < N:
            remaining_sum = np.sum(sorted_v[i + 1:])
            if remaining_sum > 0:
                scale_factor = (m - (i + 1)) / remaining_sum
                sorted_v[i + 1:] *= scale_factor
        
        # Reorder to the original order
        v[sorted_indices] = sorted_v
        
        return v
    
    # we are now storing old queries!
    selected_workloads = []
    noisy_ans_list = []
    def get_dataset_answer(workload_idx, table1_idxes, table2_idxes):
        w = qm.workload_names[workload_idx]
        # load the workload
        # size: num_queries x (nsyn1*nsyn2)
        true_answer = qm.get_true_answers(w)
        
        offsets_t1 = qm.get_offsets(w, 0, is_synth=True)
        offsets_t2 = qm.get_offsets(w, 1, is_synth=True)
        
        offsets = offsets_t1[table1_idxes] + offsets_t2[table2_idxes]
        values, counts = np.unique(offsets, return_counts=True)
        
        dataset_answer = torch.zeros([qm.workload_dict[w]["range_size"]])
        for val, count in zip(values, counts):
            dataset_answer[val] = count
        dataset_answer /= table1_idxes.shape[0]
        return true_answer, dataset_answer
    
    # initialize a b_round
    rand_idxes = torch.randperm(qm.n_syn1 * qm.n_syn2)[None, :n_relationship_synt]
    b_round = torch.sparse_coo_tensor(indices=rand_idxes, values=torch.ones([n_relationship_synt]),
                                      size=[qm.n_syn_cross], device=device).float().coalesce()
    C = torch.sparse_coo_tensor(indices=torch.empty((2, 0), dtype=torch.long), values=torch.empty(0, dtype=torch.float),
                                      size=[qm.n_syn_cross, qm.n_syn_cross], device=device).float().coalesce()
    c = torch.zeros([n_relationship_synt])
    
    # print(b_round)
    for t in tqdm(range(T)):

        def exp_mech_new_workloads(uselected_workload):
            """ Uses the exponential mechanism to select new workloads """
            
            exp_mech_workload_pool = random.sample(uselected_workload, k=min(k_choose_from, len(uselected_workload)))
            
            # get answers on this dataset
            # if queries are being reused, it makes logical sense to choose worst
            # queries on the whole dataset, not just the current slice.
            # we should not save the query matrices at this point or we will run out of memory
            true_and_dset_answers = [get_dataset_answer(i, table1_idxes, table2_idxes) for i in exp_mech_workload_pool]
            
            errors = [torch.sum(torch.abs(true_answer - dataset_answer)).numpy(force=True) for true_answer, dataset_answer in true_and_dset_answers]
            
            new_workloads = []
            for x in range(k_new_queries):
                # convert into numpy array
                errors_np = np.array(errors)
                
                # now select from this set using the exponential mechanism
                def softmax(v):
                    v_exp = np.exp(v - np.max(v))
                    return v_exp / np.sum(v_exp)
                distribution = softmax(exp_mech_factor * errors_np)
                
                # sample from the distribution
                def sample(dist):
                    cumulative_dist = np.cumsum(dist)
                    r = np.random.rand()
                    return np.searchsorted(cumulative_dist, r)
                new_workload_idx = sample(distribution)
                new_workload = exp_mech_workload_pool[new_workload_idx]
                new_workloads.append(new_workload)
                
                # remove the workload from the pool
                exp_mech_workload_pool.pop(new_workload_idx)
                errors.pop(new_workload_idx) # double check
            
            return new_workloads
                
            new_workloads_this_iter = exp_mech_new_workloads(unselected_workload)
            timers.append((time.time(), "exponential mechanism"))
            
            for i in new_workloads_this_iter:
                unselected_workload.remove(i)
                selected_workloads.append(i)
                
                workload = qm.workload_names[i]
                noisy_ans_list.append(GM_torch_noise(qm.get_true_answers(workload), gm_stddev))
            
            # initialize the Q_set from this list    
            k_val = len(selected_workloads)
            errors = []
            timers.append((time.time(), "begin workload eval"))
            for i in range(len(selected_workloads)):
                workload_idx = selected_workloads[i]
                _, dataset_ans = get_dataset_answer(workload_idx, table1_idxes, table2_idxes) # we can't actually use the true answer here!
                true_ans = noisy_ans_list[i]
                errors.append((torch.sum(torch.abs(true_ans - dataset_ans)).numpy(force=True), i))
            top_errors = sorted(errors)
            curr_workload_idxes = [i for err, i in top_errors]
            iter_selected_workloads = [selected_workloads[i] for i in curr_workload_idxes]
            iter_noisy_ans = torch.cat([noisy_ans_list[i] for i in curr_workload_idxes])
            timers.append((time.time(), "end workload eval"))
            
            for i in iter_selected_workloads:
    
                curr_workload = qm.workload_names[i]
    
                curr_Qmat_full, curr_true_answer = qm.get_query_mat_full_table(curr_workload)
                
                C = C + 1/(n_relationship_synt**2) * curr_Qmat_full.T @ curr_Qmat_full
                c = c + curr_Qmat_full.T @ curr_true_answer
                
                del curr_Qmat_full           


            b_round = b_round - eta * (C@b_round - c)
            b_round = optimal_project_to_simplex(b_round, n_relationship_synt)
            b_round = unbiased_sample_torch(b_round, m=n_relationship_synt, device=device)

            
            # clean TODO: is this necessary?
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # print(timers)
            timers_processed = [(int((timtup[0] - timers[i][0]) * 100000) / 100000, timtup[1]) for i, timtup in enumerate(timers[1:])]
            # print(timers_processed)
        
        iter_cb(qm, b_round, t)
    
    return b_round
