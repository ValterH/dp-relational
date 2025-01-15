import dp_relational.data.ipums
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
    return dp_relational.lib.synth_data.QueryManagerTorch(rel_dataset, k=k, df1_synth=df1_synth, df2_synth=df2_synth, device=device)

""" Medium size IPUMS tables """

fraction = 0.05
table_size = 10000
Tconst = 15
alpha = 0.2
k_new = 3
worst = True
q_reuse = 8
g_rels = 0.08

def make_summary_dict():
    return {
        "fraction": fraction,
        "table_size": table_size,
        "Tconst": Tconst,
        "alpha": alpha,
        "k_new": k_new,
        "worst": worst,
        "q_reuse": q_reuse,
        "g_rels": g_rels
    }

def cross_generator_torch(qm, eps_rel, T):
    b_round = dp_relational.lib.synth_data.learn_relationship_vector_torch_pgd(qm, eps_rel, T=Tconst,
                subtable_size=1000000, verbose=True, device=device, queries_to_reuse=q_reuse,
                exp_mech_alpha=alpha, k_new_queries=k_new, choose_worst=worst, slices_per_iter=3, guaranteed_rels=g_rels
            )
    print(make_summary_dict())
    relationship_syn = dp_relational.lib.synth_data.make_synthetic_rel_table_sparse(qm, b_round)
    return relationship_syn

runner = ModelRunner(self_relation=True)

runner.update(dataset_generator=lambda dmax: dp_relational.data.ipums.dataset(dmax, frac=fraction), n_syn1=table_size, n_syn2=table_size,
              synth='mst', epsilon=4.0, eps1=1.0, eps2=1.0, k=3, dmax=4, T=Tconst,
              qm_generator=qm_generator_torch, cross_generation_strategy=cross_generator_torch)
# runner.load_artifacts('6214898c-6464-11ef-a981-4e3d7b9b1ba8')

def reset_runner():
    global fraction
    global table_size
    global Tconst
    global alpha
    global k_new
    global worst
    global q_reuse
    global g_rels
    runner.update(epsilon=4.0, eps1=1.0, eps2=1.0, dmax=4, T=Tconst)

    fraction = 0.05
    table_size = 10000
    Tconst = 15
    alpha = 0.2
    k_new = 3
    worst = True
    q_reuse = 8
    g_rels = 0.08

NUM_LOOPS = 7

epsilons = [2.01, 2.1, 2.25, 2.5, 2.75, 3.0]
alphas = [0.00001, 0.2, 0.5, 0.8, 0.99999]
k_news = [1, 2, 4, 8]
q_reuses = [1, 2, 4, 8]
g_rel_opts = [0, 0.03, 0.05, 0.08] # [] # [0.12, 0.15, 0.18] # 
run_count = 0
worsts = [True, False]
Ts = [0, 1, 5, 10, 15, 25]
for loops in range(NUM_LOOPS):
    reset_runner()
    for T_in in Ts:
        Tconst = T_in
        runner.regenerate_qm = True
        runner.regenerate_cross_answers = True
        results = runner.run(extra_params={ "info": make_summary_dict(), "run_set": "AM2IPUMS_MediumPGD_Tfinal" })
        run_count += 1
        print(f"eps: {runner.epsilon}, error_ave: {results['error_ave']}")
        print(f"###### COMPLETED {run_count} RUNS ######")
    reset_runner()
    for g_in in g_rel_opts:
        g_rels = g_in
        runner.regenerate_qm = True
        runner.regenerate_cross_answers = True
        results = runner.run(extra_params={ "info": make_summary_dict(), "run_set": "AM2IPUMS_MediumPGD_g_in" })
        run_count += 1
        print(f"eps: {runner.epsilon}, error_ave: {results['error_ave']}")
        print(f"###### COMPLETED {run_count} RUNS ######")
    reset_runner()
    for q_in in q_reuses:
        q_reuse = q_in
        runner.regenerate_qm = True
        runner.regenerate_cross_answers = True
        results = runner.run(extra_params={ "info": make_summary_dict(), "run_set": "AM2IPUMS_MediumPGD_q_reuse" })
        run_count += 1
        print(f"eps: {runner.epsilon}, error_ave: {results['error_ave']}")
        print(f"###### COMPLETED {run_count} RUNS ######")
    reset_runner()
    reset_runner()
    for k_in in k_news:
        k_new = k_in
        runner.regenerate_qm = True
        runner.regenerate_cross_answers = True
        results = runner.run(extra_params={ "info": make_summary_dict(), "run_set": "AM2IPUMS_MediumPGD_knew" })
        run_count += 1
        print(f"eps: {runner.epsilon}, error_ave: {results['error_ave']}")
        print(f"###### COMPLETED {run_count} RUNS ######")
    reset_runner()
    for epsilon in epsilons:
        runner.update(epsilon=epsilon)
        runner.regenerate_qm = True
        runner.regenerate_cross_answers = True
        results = runner.run(extra_params={ "info": make_summary_dict(), "run_set": "AM2IPUMS_MediumPGD_eps" })
        run_count += 1
        print(f"eps: {runner.epsilon}, error_ave: {results['error_ave']}")
        print(f"###### COMPLETED {run_count} RUNS ######")
    reset_runner()
    for a_in in alphas:
        alpha = a_in
        runner.regenerate_qm = True
        runner.regenerate_cross_answers = True
        results = runner.run(extra_params={ "info": make_summary_dict(), "run_set": "AMIPUMS_MediumPGD_alpha" })
        run_count += 1
        print(f"eps: {runner.epsilon}, error_ave: {results['error_ave']}")
        print(f"###### COMPLETED {run_count} RUNS ######")
    reset_runner()
    # for worst_in in worsts:
    #     worst = worst_in
    #     runner.regenerate_qm = True
    #     runner.regenerate_cross_answers = True
    #     results = runner.run(extra_params={ "info": make_summary_dict(), "run_set": "AMIPUMS_MediumPGD_worst" })
    #     run_count += 1
    #     print(f"eps: {runner.epsilon}, error_ave: {results['error_ave']}")
    #     print(f"###### COMPLETED {run_count} RUNS ######")
    # reset_runner()