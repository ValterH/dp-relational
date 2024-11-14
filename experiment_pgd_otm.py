import dp_relational.data.ipums_otm
from dp_relational.lib.runner import ModelRunner

import dp_relational.lib.synth_data
import dp_relational.data.movies

import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("cuda_available", torch.cuda.is_available())
print("using device: ", device)

# EPSILON_PYTORCH_EXPERIMENT_DATASET = uuid.UUID('53b3b6a5-fe75-11ee-9cb7-a059507978f3')

def print_iter_eval(qm, b_round, T):
    relationship_syn = dp_relational.lib.synth_data.make_synthetic_rel_table_sparse(qm, b_round)
    ave_error, answers = dp_relational.lib.synth_data.evaluate_synthetic_rel_table(qm, relationship_syn)
    print(ave_error, T)

def qm_generator_torch(rel_dataset, k, df1_synth, df2_synth):
    return dp_relational.lib.synth_data.QueryManagerTorch(rel_dataset, otm=True, k=k, df1_synth=df1_synth, df2_synth=df2_synth, device=device)

""" Medium size IPUMS tables """

fraction = 0.05
table_size = 10000
Tconst = 15
alpha = 0.2
k_new = 3
worst = True
q_reuse = 8
g_rels = 0.08
synth_strat = 'mst'

def make_summary_dict():
    return {
        "fraction": fraction,
        "table_size": table_size,
        "Tconst": Tconst,
        "alpha": alpha,
        "k_new": k_new,
        "worst": worst,
        "q_reuse": q_reuse,
        "g_rels": g_rels,
        "synth_strat": synth_strat
    }

def cross_generator_torch(qm, eps_rel, T):
    b_round = dp_relational.lib.synth_data.learn_relationship_vector_torch_pgd_otm(qm, eps_rel, T=Tconst,
                subtable_size=1000000, verbose=True, device=device, queries_to_reuse=q_reuse,
                exp_mech_alpha=alpha, k_new_queries=k_new, choose_worst=worst, slices_per_iter=3, expansion_ratio=2
            )
    print(make_summary_dict())
    relationship_syn = dp_relational.lib.synth_data.make_synthetic_rel_table_sparse(qm, b_round)
    return relationship_syn

runner = ModelRunner(self_relation=False)

runner.update(dataset_generator=lambda dmax: dp_relational.data.ipums_otm.dataset(dmax, frac=fraction), n_syn1=table_size, n_syn2=table_size,
              synth='mst', epsilon=4.0, eps1=1.0, eps2=1.0, k=3, dmax=8, T=Tconst,
              qm_generator=qm_generator_torch, cross_generation_strategy=cross_generator_torch)
runner.load_artifacts('5ec5706e-7fff-11ef-b42f-bae7b799ac02')

epsilons = [2.01, 2.1, 2.25, 2.5, 2.75, 3.0]
#alphas = [0.00001, 0.2, 0.5, 0.8, 0.99999]
#k_news = [1, 2, 4, 8]
#q_reuses = [1, 2, 4, 8]
#g_rel_opts = [] # [0.12, 0.15, 0.18] # [0, 0.03, 0.05, 0.08]
run_count = 0
#synth_strats = ['mst', 'aim']
#worsts = [True, False]
#Ts = [0, 1, 5, 10, 15, 25, 50]
for loops in range(10):
    #reset_runner()
    for eps in epsilons:
        # synth_strat = sn
        runner.update(epsilon=eps)
        runner.regenerate_qm = True
        runner.regenerate_cross_answers = True
        results = runner.run(extra_params={ "info": make_summary_dict(), "run_set": "0" })
        run_count += 1
        print(runner.rel_dataset_runid, runner.synth_tables_runid, runner.relationship_syn_runid)
        print(f"eps: {runner.epsilon}, error_ave: {results['error_ave']}")
        print(f"###### COMPLETED {run_count} RUNS ######")
