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

@torch.no_grad()
def learn_relationship_vector_torch_priv_cor(qm: QueryManagerTorch, epsilon_relationship=1.0, T=100, T_mirror=150,
                                           num_workload_ite = 2, delta_relationship = 1e-5, subtable_size=100000,
                                           verbose=False, device="cpu"):
    """
    This is another algorithm for learning a relationship vector.

    This samples new workloads each iteration randomly, and then uses a mirror descent to find the new relationship vector.
    It however does not learn the whole table at once: it only takes a slice of size subtable_size and learns the relationships
    in this subset. This speeds up the execution of the algorithm.

    Priv_cor is in the name as originally there was an issue in the privacy budget of a predecessor algorithm.
    """
    n_relationship_orig = qm.n_relationship_orig
    n_relationship_synt = qm.n_relationship_synth
    m_privacy = n_relationship_orig
    
    assert n_relationship_synt < qm.n_syn1 * qm.n_syn2

    # fraction of each table that should be taken    
    table_frac = np.sqrt(subtable_size / (qm.n_syn1 * qm.n_syn2))
    # size of each table to take
    table1_slice_size = int(np.clip(table_frac * qm.n_syn1, 1, qm.n_syn1))
    table2_slice_size = int(np.clip(table_frac * qm.n_syn2, 1, qm.n_syn2))
    cross_slice_size = table1_slice_size * table2_slice_size
    
    # number of workloads to compute per iteration
    # num_workload_ite = 2

    epsilon_relationship = epsilon_relationship/(qm.rel_dataset.dmax * num_workload_ite)

    # convert to RDP
    rho_rel = cdp_rho(epsilon_relationship, delta_relationship)

    # privacy parameter
    epsilon0 = np.sqrt((2 * rho_rel) / (num_workload_ite * T)) if T != 0 else 100000
    
    # exponential mechanism factor: product before the softmax
    exp_mech_alpha = 0
    # exp_mech_factor = np.sqrt(exp_mech_alpha) * epsilon0 * (m_privacy / qm.rel_dataset.dmax)
    
    # gaussian mechanism standard deviation
    gm_stddev = (np.sqrt(2) / (np.sqrt(1 - exp_mech_alpha) * epsilon0)) * (qm.rel_dataset.dmax / m_privacy)

    # intialization
    unselected_workload = [i for i in range(len(qm.workload_names))]
    
    # do not store old queries: this will be too much of a mess
    rand_idxes = torch.randperm(qm.n_syn1 * qm.n_syn2)[None, :n_relationship_synt]
    b_round = torch.sparse_coo_tensor(indices=rand_idxes, values=torch.ones([n_relationship_synt]),
                                      size=[qm.n_syn_cross], device=device).float().coalesce()
    # print(b_round)
    for t in tqdm(range(T)):
        # choose a set to slice
        slice_table1 = torch.randperm(qm.n_syn1, device=device)[:table1_slice_size]
        slice_table2 = torch.randperm(qm.n_syn2, device=device)[:table2_slice_size]
        # identify which cells these are in b
        offsets_table1 = slice_table1.repeat_interleave(table2_slice_size) * qm.n_syn2
        offsets_table2 = slice_table2.repeat(table1_slice_size) # somethings up...
        offsets = offsets_table1 + offsets_table2
        # we will start optimising from here
        sub_num_relationships = int(torch.sparse.sum(torch.index_select(b_round, 0, offsets)).numpy(force=True))
        b_slice = torch.ones([cross_slice_size]).to(device=device).float() / (cross_slice_size)
        
        Q_set = torch.empty((0, cross_slice_size)).to_sparse_coo().to(device=device).float().coalesce()
        noisy_ans = torch.empty((0,)).float()
        
        # TODO: use exponential mechanism!!!
        # Here I randomly choose 2 workloads
        select_workload = random.choices(unselected_workload, k=num_workload_ite)
        # unselected_workload = [i for i in unselected_workload if i not in select_workload] (allow repeated workload selection)

        for i in select_workload:

            curr_workload = qm.workload_names[i]

            curr_Qmat_full, curr_true_answer = qm.get_query_mat_full_table(curr_workload)
            curr_Qmat = torch.index_select(curr_Qmat_full, 1, offsets).coalesce()
            
            del curr_Qmat_full

            noisy_curr_ans = GM_torch_noise(curr_true_answer, gm_stddev)

            noisy_ans = torch.cat([noisy_ans, noisy_curr_ans])

            Q_set = torch_cat_sparse_coo([Q_set, curr_Qmat], device=device)
            
            del curr_Qmat
            del noisy_curr_ans
        
        b_slice = mirror_descent_torch(Q_set, b_slice, noisy_ans.to(device=device), step_size = 0.01, T_mirror = T_mirror)
        
        # put these back into the slice: this is slightly complicated!
        b_slice *= sub_num_relationships
        b_slice_round = expround_torch(b_slice, m=sub_num_relationships, device=device)
        
        # create a mask
        mask = torch.sparse_coo_tensor(offsets[None, :], torch.ones_like(offsets), size=[qm.n_syn_cross], device=device).coalesce()
        b_round.coalesce()
        b_round = b_round - (mask * b_round) # now the area is filled with zeros
        # get nonzero indices in b_slice_round
        nz_indices = torch.squeeze(b_slice_round.indices())
        # lookup what offsets these were in the original tensor
        new_offsets = offsets[nz_indices]
        # create new values
        new_values = torch.sparse_coo_tensor(new_offsets[None, :], torch.ones_like(new_offsets), size=[qm.n_syn_cross], device=device).coalesce()
        b_round = b_round + new_values
        
        # clean TODO: is this necessary?
        del slice_table1
        del slice_table2
        del offsets
        del mask
        del Q_set
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return b_round
