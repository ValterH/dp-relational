from snsynth import Synthesizer
from .dataset import RelationalDataset
import itertools
import random
import numpy as np
import pandas as pd

import torch

from .helpers import cdp_delta, cdp_eps, cdp_rho, get_per_round_privacy_budget

from tqdm import tqdm

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

class QueryManager:
    """
    Query manager class.
    Given an input relational dataset, an input k for the desired k-way marginals
    and two pregenerated synthetic tables, returns an interface that allows for the
    calculation of "true" answers from the synthetic dataset.
    
    Also stores query vectors
    """
    def __init__(self, rel_dataset: RelationalDataset, k, df1_synth, df2_synth, verbose=False) -> None:
        self.verbose = verbose
        
        self.rel_dataset = rel_dataset
        self.k = k
        self.df1_synth = df1_synth
        self.df2_synth = df2_synth
        
        self.n_syn1 = df1_synth.shape[0]
        self.n_syn2 = df2_synth.shape[0]
        
        self.num_relationship = self.rel_dataset.df_rel.shape[0]
        
        def make_cross_workloads():
            workload_names = []
            workload_dict = {}
            
            df1 = rel_dataset.table1.df
            df2 = rel_dataset.table2.df
            
            df1_col_dic = rel_dataset.table1.col_dict
            df2_col_dic = rel_dataset.table2.col_dict
            
            range_low = 0
            range_high = 0
            for k1 in range(1,k):
                df1_k1_col = list(itertools.combinations(df1.columns, k1))
                for k2 in range(1, k-k1+1):
                    df2_k2_col = list(itertools.combinations(df2.columns, k2))
                    if k1+k2 == k:
                        for col_comb_1 in df1_k1_col:
                            for col_comb_2 in df2_k2_col:
                                len_range = 1
                                uni_val_1 = []
                                
                                for col1 in col_comb_1:
                                    c1 = df1_col_dic[col1]['count']
                                    uni_val_1.append(c1)
                                    len_range *= c1
                                assert len(uni_val_1) == k1
                                
                                uni_val_2 = []
                                for col2 in col_comb_2:
                                    c2 = df2_col_dic[col2]['count']
                                    uni_val_2.append(c2)
                                    len_range *= c2
                                assert len(uni_val_2) == k2
                                
                                range_high = range_low + len_range - 1 
                                workload_names.append((col_comb_1 , col_comb_2))
                                workload_dict[(col_comb_1 , col_comb_2)] = {"dim_1": uni_val_1, "dim_2": uni_val_2,"range_low": range_low, "range_high": range_high}
                                
                                range_low = range_high + 1
            
            return (workload_names, workload_dict)
        
        self.workload_names, self.workload_dict = make_cross_workloads()
        
        last_workload = self.workload_names[-1]
        self.num_all_queries = self.workload_dict[last_workload]["range_high"] + 1
        
        def calculate_rand_ans():
            rand_ans = np.zeros(self.num_all_queries) # what answers would a random system give?
            for workload_name in self.workload_dict:
                w = self.workload_dict[workload_name]
                rand_ans[w['range_low']: (w['range_high']+1)] = 1.0/(w['range_high'] - w['range_low'] + 1)
            return rand_ans
        def calculate_true_ans():
            df_rel = rel_dataset.df_rel
            num_relationship = df_rel.shape[0]
            
            true_ans = np.zeros(self.num_all_queries)
            for i in range(num_relationship):
                ID1 = df_rel.iloc[i][rel_dataset.rel_id1_col]
                ID2 = df_rel.iloc[i][rel_dataset.rel_id2_col]
                
                for w in self.workload_names:
                    cols1 = w[0]
                    cols2 = w[1]
                    v1 = []
                    for c1 in cols1:
                        v1.append(rel_dataset.table1.df.iloc[ID1][c1])
                        
                    v2 = []
                    for c2 in cols2:
                        v2.append(rel_dataset.table2.df.iloc[ID2][c2])
                    
                    ind = self.query_ind(w, [v1,v2])
                    true_ans[int(ind)] += 1
                    
            true_ans = true_ans/num_relationship
            
            return true_ans
        
        self.rand_ans = calculate_rand_ans()
        self.true_ans = calculate_true_ans()
    
    def query_ind_workload(self, workload):
        return list(range(self.workload_dict[workload]['range_low'],
                          self.workload_dict[workload]['range_high'] + 1))
    
    def query_ind(self, workload, val, zero_index=False):
        assert len(workload) == 2 #since only two tables
        assert len(workload[0]) + len(workload[1]) == self.k # k features?
        assert len(workload[0]) > 0 and len(workload[1]) > 0 # cross table queries
        
        assert len(val) == 2
        assert len(val[0]) == len(workload[0]) and len(val[1]) == len(workload[1])
        
        # TODO: add assert making sure val belongs to the unique value set of workload
        total_val = val[0] + val[1]
        
        q_ind = self.workload_dict[workload]['range_low']
        dim1 = self.workload_dict[workload]['dim_1']
        dim2 = self.workload_dict[workload]['dim_2']
        
        dim = dim1 + dim2
        
        total_dim = 1
        for i in dim:
            total_dim *= i
        
        # reshape
        for i in range(len(total_val)):
            total_dim = total_dim/dim[i]
            q_ind += total_dim * total_val[i]
            
        assert total_dim == 1
        assert q_ind <= self.workload_dict[workload]['range_high']
        
        if zero_index:
            q_ind -= self.workload_dict[workload]['range_low']
        
        return q_ind
    
    def query_ind_df1(self, workload, val1):    
        assert len(workload) == 2 #since only two tables
        assert len(workload[0]) + len(workload[1]) == self.k # k features?
        assert len(workload[0]) > 0 and len(workload[1]) > 0 # cross table queries
        
        assert len(val1) == len(workload[0])
        
        
        q_ind = self.workload_dict[workload]['range_low']
        dim1 = self.workload_dict[workload]['dim_1']
        dim2 = self.workload_dict[workload]['dim_2']
        
        dim = dim1 + dim2
        
        total_dim = 1
        for i in dim:
            total_dim *= i
        
        # reshape
        for i in range(len(val1)):
            total_dim = total_dim/dim[i]
            q_ind += total_dim * val1[i]
        
        q_ind_min = q_ind
        q_ind_max = q_ind
        
        for j in dim2:
            total_dim = total_dim/j
            q_ind_max += total_dim * (j-1)
        
        q = [i for i in range(int(q_ind_min), int(q_ind_max + 1))]
        
        assert max(q) <= self.workload_dict[workload]['range_high']
        
        return q
    
    def query_ind_df2(self, workload, val2):    
        assert len(workload) == 2 #since only two tables
        assert len(workload[0]) + len(workload[1]) == self.k # k features?
        assert len(workload[0]) > 0 and len(workload[1]) > 0 # cross table queries
        
        assert len(val2) == len(workload[1])
        
        
        q_ind = self.workload_dict[workload]['range_low']
        dim1 = self.workload_dict[workload]['dim_1']
        dim2 = self.workload_dict[workload]['dim_2']
        
        total_dim1 = 1
        for i in dim1:
            total_dim1 *= i
        
        q_ind_min = 0
        q_ind_max = 0
        
        for j in dim1:
            total_dim1 = total_dim1/j
            q_ind_max += total_dim1 * (j-1)
        q = [i for i in range(int(q_ind_min), int(q_ind_max + 1))]
        
        
        total_dim2 = 1
        
        for i in dim2:
            total_dim2 *= i
        
        q = [int(i * int(total_dim2)) for i in q]
        
        
        temp = q_ind
        for i in range(len(val2)):
            total_dim2 = total_dim2/dim2[i]
            temp += total_dim2 * val2[i]
        
        q = [int(i + temp) for i in q]
        
        assert max(q) <= self.workload_dict[workload]['range_high']
        return q

    def get_queries(self, rows):
        raise NotImplementedError("Base query manager has no get_queries")

class QueryManagerBasic(QueryManager):
    def __init__(self, rel_dataset: RelationalDataset, k, df1_synth, df2_synth, verbose=False) -> None:
        super().__init__(rel_dataset, k, df1_synth, df2_synth, verbose)
        
        if verbose:
            print("Constructing query matrix")
        # create the query matrix
        Q1 = np.zeros((self.num_all_queries, self.n_syn1 * self.n_syn2))

        for i in tqdm(range(self.n_syn1), disable=not verbose):
            row = self.df1_synth.iloc[i]
            #data indices
            ind_d = [i*self.n_syn2 + j for j in range(self.n_syn2)]
            
            for w in self.workload_names:
                val = list(row[list(w[0])])
                #query indices
                ind_q = self.query_ind_df1(w, val)
                
                Q1[np.ix_(ind_q, ind_d)] = 1
                
                
        Q2 = np.zeros((self.num_all_queries, self.n_syn1 * self.n_syn2))

        for j in tqdm(range(self.n_syn2)):
            row = self.df2_synth.iloc[j]
            #data indices
            ind_d = [i*self.n_syn2 + j for i in range(self.n_syn1)]
            
            for w in self.workload_names:
                val = list(row[list(w[1])])
                #query indices
                ind_q = self.query_ind_df2(w, val)
                
                Q2[np.ix_(ind_q, ind_d)] = 1
        
        Q = Q1 * Q2
        assert sum(sum(Q)) == (len(self.workload_names) * self.n_syn1 * self.n_syn2)
        self.Q = Q
    def get_queries(self, rows):
        row_arr = np.array(rows)
        return self.Q[row_arr, :]

class QueryManagerTorch(QueryManager):
    """
    Query manager implementation in Pytorch, containing several optimizations.
        - Lazy generation of workload query vectors
        - Support for slicing to learn subsections of the query
        - Sparse Pytorch storage of query vectors (using COO)
    """
    def __init__(self, rel_dataset: RelationalDataset, k, df1_synth, df2_synth, device="cpu", verbose=False) -> None:
        super().__init__(rel_dataset, k, df1_synth, df2_synth, verbose)
        self.workload_query_answers = {}
        for workload in self.workload_names:
            self.workload_query_answers[workload] = None
        self.true_ans_tensor = torch.from_numpy(self.true_ans).to(device=device)
    def get_query_mat_full_table(self, workload):
        if self.workload_query_answers[workload] is not None:
            print("cached")
            return self.workload_query_answers[workload]
        else:
            # generate the matrix
            print("uncached")
            range_low = self.workload_dict[workload]["range_low"]
            range_high = self.workload_dict[workload]["range_high"] + 1
            num_queries = range_high - range_low
            vec_len = self.n_syn1 * self.n_syn2
            true_vals = self.true_ans_tensor[range_low:range_high]
            
            indices_row = np.zeros((vec_len, ))
            indices_col = np.arange(0, vec_len)
            
            pbar = tqdm(total=vec_len, disable=not self.verbose)
            
            table_t1 = self.df1_synth[list(workload[0])]
            table_t1 = tuple(table_t1[x].iloc[:].values for x in workload[0])
            
            dim1 = self.workload_dict[workload]['dim_1']
            dim2 = self.workload_dict[workload]['dim_2']
            dim = dim1 + dim2
            total_dim = 1
            for i in dim:
                total_dim *= i
                
            t1_dimsizes = []
            for subdim_size in dim1:
                total_dim = total_dim/subdim_size
                t1_dimsizes.append(total_dim)
            t2_dimsizes = []
            for subdim_size in dim2:
                total_dim = total_dim/subdim_size
                t2_dimsizes.append(total_dim)

            offsets = np.zeros(shape=table_t1[0].shape)
            for t1_col in range(len(t1_dimsizes)):
                offsets += table_t1[t1_col] * t1_dimsizes[t1_col]
            
            for y in range(self.n_syn2):
                row_t2 = self.df2_synth[list(workload[1])].iloc[y]
                row_t2 = tuple(row_t2[x] for x in workload[1]) # get the integer values
                
                val_index = np.copy(offsets)
                for t2_col in range(len(t2_dimsizes)):
                    val_index += row_t2[t2_col] * t2_dimsizes[t2_col]
                indices_row[(y * self.n_syn1):((y + 1) * self.n_syn1)] = val_index
            pbar.close()
            indices = np.stack((indices_row, indices_col))
            query_mat = torch.sparse_coo_tensor(indices, np.ones(shape=(vec_len, )), size=(num_queries, vec_len))
            # print(torch.sparse.sum(query_mat, dim=1))
            # print(vec_len)
            self.workload_query_answers[workload] = (query_mat, true_vals)
            return self.get_query_mat_full_table(workload)
# TODO: rename
def learn_relationship_vector_basic(qm: QueryManager, epsilon_relationship=1.0, T=100,
                                   delta_relationship = 1e-5, verbose=False):
    Q = qm.Q
    num_relationship = qm.num_relationship
    n_relationship_synt = int(num_relationship / 5) # TODO: fix!
    
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
            return 2 * Q.T @ (np.matmul(Q, b) - a)

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
def learn_relationship_vector_torch(qm: QueryManagerTorch, epsilon_relationship=1.0, T=100,
                                   delta_relationship = 1e-5, verbose=False, device="cpu"):
    num_relationship = qm.num_relationship
    n_relationship_synt = int(num_relationship / 5) # TODO: fix!
    
    def mirror_descent(Q, b, a, step_size = 0.01, T_mirror = 50):
        # b is a vector whose sum is 1
        assert isinstance(Q, torch.Tensor)
        assert isinstance(b, torch.Tensor)
        assert isinstance(a, torch.Tensor)
        
        print(Q.shape)
        print(b.shape)
        print(a.shape)

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
    def GM(inp, rho):
        rand = torch.normal(0, np.sqrt(2)/(num_relationship * np.sqrt(rho)), size=inp.size)
        return inp + rand
    def expround(b):
        m = torch.sum(b).numpy(force=True)
        X_dist = torch.distributions.exponential.Exponential(1 / b)
        X = X_dist.sample()
        # finding the index of top m elements
        values, indices = torch.topk(X, int(m))
        
        bu = torch.sparse_coo_tensor(indices, torch.ones_like(indices), b.size)
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
    Q_set = torch.empty((0, qm.n_syn1 * qm.n_syn2))
    noisy_ans = torch.empty((0,))
    b = np.ones(qm.n_syn1 * qm.n_syn2) / (qm.n_syn1 * qm.n_syn2)

    for t in tqdm(range(T)):

        # TODO: use exponential mechanism!!!
        # Here I randomly choose 2 workloads
        select_workload = random.choices(unselected_workload, k=num_workload_ite)
        unselected_workload = [i for i in unselected_workload if i not in select_workload]

        for i in select_workload:

            curr_workload = qm.workload_names[i]

            curr_Qmat, curr_true_answer = qm.get_query_mat_full_table(curr_workload)
            
            noisy_curr_ans = GM(curr_true_answer, per_round_rho_rel)

            noisy_ans = torch.cat([noisy_ans, noisy_curr_ans])

            Q_set = torch.cat([Q_set, curr_Qmat])

        b = mirror_descent(Q_set, b, noisy_ans, step_size = 0.01, T_mirror = 50)

    b = b * n_relationship_synt
    b_round = expround(b)
    b_round = b_round.reshape(qm.n_syn1, qm.n_syn2)
    
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
