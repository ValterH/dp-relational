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
def learn_relationship_vector_torch(qm: QueryManagerTorch, epsilon_relationship=1.0, T=100, T_mirror=50,
                                    num_workload_ite = 2, delta_relationship = 1e-5,
                                    verbose=False, device="cpu"):
    n_relationship_orig = qm.n_relationship_orig
    n_relationship_synt = qm.n_relationship_synth
    
    assert n_relationship_synt < qm.n_syn1 * qm.n_syn2
    
    # number of workloads to compute per iteration
    # num_workload_ite = 2

    epsilon_relationship = epsilon_relationship/(qm.rel_dataset.dmax * num_workload_ite)

    # convert to RDP
    rho_rel = cdp_rho(epsilon_relationship, delta_relationship)

    # privacy budget per iteration
    per_round_rho_rel = (rho_rel / 0.0001) if T == 0 else rho_rel / T

    # intialization
    unselected_workload = [i for i in range(len(qm.workload_names))]
    Q_set = torch.empty((0, qm.n_syn1 * qm.n_syn2)).to_sparse_coo().to(device=device).float().coalesce()
    noisy_ans = torch.empty((0,)).float()
    b = torch.ones([qm.n_syn1 * qm.n_syn2]).to(device=device).float() / (qm.n_syn1 * qm.n_syn2)

    for t in tqdm(range(T)):

        # TODO: use exponential mechanism!!!
        # Here I randomly choose 2 workloads
        select_workload = random.choices(unselected_workload, k=num_workload_ite)
        unselected_workload = [i for i in unselected_workload if i not in select_workload]

        for i in select_workload:

            curr_workload = qm.workload_names[i]

            curr_Qmat, curr_true_answer = qm.get_query_mat_full_table(curr_workload)

            noisy_curr_ans = GM_torch(curr_true_answer, per_round_rho_rel, n_relationship_orig)

            noisy_ans = torch.cat([noisy_ans, noisy_curr_ans])

            Q_set = torch_cat_sparse_coo([Q_set, curr_Qmat], device=device)

        b = mirror_descent_torch(Q_set, b, noisy_ans, step_size = 0.01, T_mirror=T_mirror)

    b = b * n_relationship_synt
    b_round = expround_torch(b, device=device)
    
    return b_round