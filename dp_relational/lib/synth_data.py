from snsynth import Synthesizer
from .dataset import RelationalDataset
import itertools
import random
import numpy as np
import pandas as pd

import torch

from .helpers import cdp_delta, cdp_eps, cdp_rho, get_per_round_privacy_budget, torch_cat_sparse_coo

from tqdm import tqdm

from .qm import QueryManager, QueryManagerBasic, QueryManagerTorch

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
def GM_torch(inp, rho, num_relationship):
    rand = torch.normal(0.0, (np.sqrt(2)/(num_relationship * np.sqrt(rho))).item(), inp.size())
    return inp + rand
def expround_torch(b):
    m = int(torch.sum(b).numpy(force=True))
    X_dist = torch.distributions.exponential.Exponential(1 / b)
    X = X_dist.sample()
    # finding the index of top m elements
    values, indices = torch.topk(X, m)
    indices = indices.unsqueeze(0)
    
    bu = torch.sparse_coo_tensor(indices, torch.ones([m]), b.size()).coalesce()
    return bu

def compute_single_table_synth_data(df, n1, synthesizer='patectgan', epsilon=3, preprocessor_eps=0.5):
    if synthesizer == 'patectgan':
        synth = Synthesizer.create("patectgan", epsilon=epsilon, verbose=True)
        synth.fit(df, preprocessor_eps=preprocessor_eps)
        dat = synth.sample(n1)
        return dat
    elif synthesizer == 'dpctgan':
        synth = Synthesizer.create("dpctgan", epsilon=epsilon, verbose=True)
        synth.fit(df, preprocessor_eps=preprocessor_eps)
        dat = synth.sample(n1)
        return dat
    elif synthesizer == 'mst':
        synth = Synthesizer.create("mst", epsilon=epsilon, verbose=True)
        synth.fit(df, preprocessor_eps=preprocessor_eps)
        dat = synth.sample(n1)
        return dat
    elif synthesizer == 'aim':
        synth = Synthesizer.create("aim", epsilon=epsilon, verbose=True)
        synth.fit(df, preprocessor_eps=preprocessor_eps)
        dat = synth.sample(n1)
        return dat
    else:
        return df

# TODO: rename
def learn_relationship_vector_basic(qm: QueryManagerBasic, epsilon_relationship=1.0, T=100,
                                   delta_relationship = 1e-5, verbose=False):
    Q = qm.Q
    num_relationship = qm.num_relationship
    n_relationship_synt = num_relationship
    
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
            val = inp[i] + np.random.normal(0, np.sqrt(2)/(num_relationship * np.sqrt(rho)))
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
    per_round_rho_rel = rho_rel / T

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

@torch.no_grad()
def learn_relationship_vector_torch(qm: QueryManagerTorch, epsilon_relationship=1.0, T=100, T_mirror=50,
                                    num_workload_ite = 2, delta_relationship = 1e-5,
                                    verbose=False, device="cpu"):
    num_relationship = qm.num_relationship
    n_relationship_synt = num_relationship
    
    assert n_relationship_synt < qm.n_syn1 * qm.n_syn2
    
    # number of workloads to compute per iteration
    # num_workload_ite = 2

    epsilon_relationship = epsilon_relationship/(qm.rel_dataset.dmax * num_workload_ite)

    # convert to RDP
    rho_rel = cdp_rho(epsilon_relationship, delta_relationship)

    # privacy budget per iteration
    per_round_rho_rel = rho_rel / T

    # intialization
    unselected_workload = [i for i in range(len(qm.workload_names))]
    Q_set = torch.empty((0, qm.n_syn1 * qm.n_syn2)).to_sparse_coo().float().coalesce()
    noisy_ans = torch.empty((0,)).float()
    b = torch.ones([qm.n_syn1 * qm.n_syn2]).float() / (qm.n_syn1 * qm.n_syn2)

    for t in tqdm(range(T)):

        # TODO: use exponential mechanism!!!
        # Here I randomly choose 2 workloads
        select_workload = random.choices(unselected_workload, k=num_workload_ite)
        unselected_workload = [i for i in unselected_workload if i not in select_workload]

        for i in select_workload:

            curr_workload = qm.workload_names[i]

            curr_Qmat, curr_true_answer = qm.get_query_mat_full_table(curr_workload)

            noisy_curr_ans = GM_torch(curr_true_answer, per_round_rho_rel, num_relationship)

            noisy_ans = torch.cat([noisy_ans, noisy_curr_ans])

            Q_set = torch_cat_sparse_coo([Q_set, curr_Qmat])

        b = mirror_descent_torch(Q_set, b, noisy_ans, step_size = 0.01, T_mirror=T_mirror)

    b = b * n_relationship_synt
    b_round = expround_torch(b)
    
    return b_round

@torch.no_grad()
def learn_relationship_vector_torch_masked(qm: QueryManagerTorch, epsilon_relationship=1.0, T=100, T_mirror=150,
                                           num_workload_ite = 2, delta_relationship = 1e-5, subtable_size=100000,
                                           verbose=False, device="cpu"):
    """Learns only some subset of the relationship vector at a time."""
    num_relationship = qm.num_relationship
    n_relationship_synt = num_relationship
    
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

    # privacy budget per iteration
    per_round_rho_rel = rho_rel / T

    # intialization
    unselected_workload = [i for i in range(len(qm.workload_names))]
    
    # do not store old queries: this will be too much of a mess
    rand_idxes = torch.randperm(qm.n_syn1 * qm.n_syn2)[None, :n_relationship_synt]
    b_round = torch.sparse_coo_tensor(indices=rand_idxes, values=torch.ones([n_relationship_synt]),
                                      size=[qm.n_syn_cross]).float().coalesce()
    # print(b_round)
    for t in tqdm(range(T)):
        # choose a set to slice
        slice_table1 = torch.randperm(qm.n_syn1)[:table1_slice_size]
        slice_table2 = torch.randperm(qm.n_syn2)[:table2_slice_size]
        # identify which cells these are in b
        offsets_table1 = slice_table1.repeat_interleave(table2_slice_size) * qm.n_syn2
        offsets_table2 = slice_table2.repeat(table1_slice_size) # somethings up...
        offsets = offsets_table1 + offsets_table2
        # we will start optimising from here
        sub_num_relationships = torch.index_select(b_round, 0, offsets)._nnz()
        b_slice = torch.ones([cross_slice_size]).float() / (cross_slice_size)
        
        Q_set = torch.empty((0, cross_slice_size)).to_sparse_coo().float().coalesce()
        noisy_ans = torch.empty((0,)).float()
        
        # TODO: use exponential mechanism!!!
        # Here I randomly choose 2 workloads
        select_workload = random.choices(unselected_workload, k=num_workload_ite)
        # unselected_workload = [i for i in unselected_workload if i not in select_workload] (allow repeated workload selection)

        for i in select_workload:

            curr_workload = qm.workload_names[i]

            curr_Qmat, curr_true_answer = qm.get_query_mat_full_table(curr_workload)
            curr_Qmat = torch.index_select(curr_Qmat, 1, offsets).coalesce()

            noisy_curr_ans = GM_torch(curr_true_answer, per_round_rho_rel, sub_num_relationships) # TODO: what to use??

            noisy_ans = torch.cat([noisy_ans, noisy_curr_ans])

            Q_set = torch_cat_sparse_coo([Q_set, curr_Qmat])
        
        b_slice = mirror_descent_torch(Q_set, b_slice, noisy_ans, step_size = 0.01, T_mirror = T_mirror)
        
        # put these back into the slice: this is slightly complicated!
        b_slice *= n_relationship_synt
        b_slice_round = expround_torch(b_slice)
        
        # create a mask
        mask = torch.sparse_coo_tensor(offsets[None, :], torch.ones_like(offsets), size=[qm.n_syn_cross]).coalesce()
        b_round = b_round - (mask * b_round) # now the area is filled with zeros
        
        # get nonzero indices in b_slice_round
        nz_indices = torch.squeeze(b_slice_round.indices())
        # lookup what offsets these were in the original tensor
        new_offsets = offsets[nz_indices]
        # create new values
        new_values = torch.sparse_coo_tensor(new_offsets[None, :], torch.ones_like(new_offsets), size=[qm.n_syn_cross]).coalesce()
        b_round = b_round + new_values
    
    return b_round

def make_synthetic_rel_table(qm: QueryManager, b_round):
    ID_1 = qm.rel_dataset.rel_id1_col
    ID_2 = qm.rel_dataset.rel_id2_col
    
    relationship_syn = pd.DataFrame(columns=[ID_1, ID_2])

    for i in range(qm.n_syn1):
        for j in range(qm.n_syn2):
            if b_round[i,j] == 1:
                new_data = pd.DataFrame([{ID_1: i, ID_2: j}])
                relationship_syn = pd.concat([relationship_syn, new_data], ignore_index=True)
    
    return relationship_syn

def make_synthetic_rel_table_sparse(qm: QueryManager, b_round):
    ID_1 = qm.rel_dataset.rel_id1_col
    ID_2 = qm.rel_dataset.rel_id2_col
    
    relationships = np.squeeze(b_round.coalesce().indices().numpy())
    table2_nums = relationships % qm.n_syn2
    table1_nums = (relationships - table2_nums) // qm.n_syn2
    
    relationship_syn = pd.DataFrame(data={
        ID_1: table1_nums,
        ID_2: table2_nums
    })

    return relationship_syn

def evaluate_synthetic_rel_table(qm: QueryManager, relationship_syn):
    ID_1 = qm.rel_dataset.rel_id1_col
    ID_2 = qm.rel_dataset.rel_id2_col
    
    num_relationship_syn = relationship_syn.shape[0]
    ans_syn = np.zeros(qm.num_all_queries)

    for i in range(num_relationship_syn):

        # TODO: assert ID_1 and ID_2 correspond to index

        ID1 = relationship_syn.iloc[i][ID_1]

        ID2 = relationship_syn.iloc[i][ID_2]

        for w in qm.workload_names:
            cols1 = w[0]
            cols2 = w[1]
            v1 = []
            for c1 in cols1:
                v1.append(qm.df1_synth.iloc[ID1][c1])

            v2 = []
            for c2 in cols2:
                v2.append(qm.df2_synth.iloc[ID2][c2])

            ind = qm.query_ind(w, [v1,v2])
            ans_syn[int(ind)] += 1

    ans_syn = ans_syn/num_relationship_syn
    
    ave_error =100 * np.sum(np.abs(ans_syn - qm.true_ans)) / len(qm.true_ans)
    max_error =100 * np.max(np.abs(ans_syn - qm.true_ans))
    
    return (ave_error, max_error)

def synthesize_cross_table(rel_dataset: RelationalDataset, synth='mst', epsilon=3.0, T=100,
                           n_syn1=776, n_syn2=1208, eps1=1.0, eps2=1.0, k=2, dmax=10, verbose=False):
    """Generates synthetic data for a relational dataset""" 
    df1_synth = compute_single_table_synth_data(rel_dataset.table1.df, n_syn1, synth, epsilon=eps1)
    df2_synth = compute_single_table_synth_data(rel_dataset.table2.df, n_syn2, synth, epsilon=eps2)
    
    df_rel = rel_dataset.df_rel
    
    qm = QueryManager(rel_dataset, k=k, df1_synth=df1_synth, df2_synth=df2_synth)
    # TODO!
