from dp_relational.lib.runner import ModelRunner

import dp_relational.lib.synth_data
import dp_relational.data.airbnb

import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("cuda_available", torch.cuda.is_available())
print("using device: ", device)

# EPSILON_PYTORCH_EXPERIMENT_DATASET = uuid.UUID('53b3b6a5-fe75-11ee-9cb7-a059507978f3')


def print_iter_eval(qm, b_round, T):
    relationship_syn = dp_relational.lib.synth_data.make_synthetic_rel_table_sparse(
        qm, b_round
    )
    ave_error, answers = dp_relational.lib.synth_data.evaluate_synthetic_rel_table(
        qm, relationship_syn
    )
    print(ave_error, T)


def qm_generator_torch(rel_dataset, k, df1_synth, df2_synth):
    return dp_relational.lib.synth_data.QueryManagerTorch(
        rel_dataset, k=k, df1_synth=df1_synth, df2_synth=df2_synth, device=device
    )


""" Medium size IPUMS tables """

table_size = 10000
Tconst = 25
alpha = 0.2
k_new = 15
worst = False
q_reuse = 15
g_rels = 0.5


def make_summary_dict():
    return {
        "table_size": table_size,
        "Tconst": Tconst,
        "alpha": alpha,
        "k_new": k_new,
        "worst": worst,
        "q_reuse": q_reuse,
        "g_rels": g_rels,
    }


def cross_generator_torch(qm, eps_rel, T):
    b_round = dp_relational.lib.synth_data.learn_relationship_vector_torch_pgd(
        qm,
        eps_rel,
        T=Tconst,
        subtable_size=1000000,
        verbose=True,
        device=device,
        queries_to_reuse=q_reuse,
        exp_mech_alpha=alpha,
        k_new_queries=k_new,
        choose_worst=worst,
        slices_per_iter=8,
        guaranteed_rels=g_rels,
        iter_cb=print_iter_eval,
    )
    print(make_summary_dict())
    relationship_syn = dp_relational.lib.synth_data.make_synthetic_rel_table_sparse(
        qm, b_round
    )
    return relationship_syn


runner = ModelRunner(self_relation=False)

runner.update(
    dataset_generator=dp_relational.data.airbnb.dataset,
    n_syn1=10_000,
    n_syn2=47217,
    synth="aim",
    epsilon=6.0,
    eps1=2.0,
    eps2=2.0,
    k=3,
    dmax=10,
    T=Tconst,
    qm_generator=qm_generator_torch,
    cross_generation_strategy=cross_generator_torch,
)


def reset_runner():
    global table_size
    global Tconst
    global alpha
    global k_new
    global worst
    global q_reuse
    global g_rels
    runner.update(epsilon=6.0, eps1=2.0, eps2=2.0, dmax=10, T=Tconst)

    table_size = 10000
    Tconst = 15
    alpha = 0.2
    k_new = 15
    worst = False
    q_reuse = 15
    g_rels = 0.0


NUM_LOOPS = 7

epsilons = [4.1, 4.5] + [
    x + 4.0 + 1.0 for x in range(10)
]  # [5, 2.01, 2.1, 2.25, 2.5, 2.75, 3.0]
alphas = [0.00001, 0.2, 0.5, 0.8, 0.99999]
k_news = [1, 2, 4, 8]
q_reuses = [1, 2, 4, 8, 15]
g_rel_opts = [0, 0.1, 0.3, 0.5]  # [] # [0.12, 0.15, 0.18] #
run_count = 0
worsts = [True, False]
Ts = [0, 1, 5, 10, 15, 25, 40]

for loops in range(NUM_LOOPS):
    for synth in ["aim"]:  # "mst",
        runner.update(synth=synth)
        reset_runner()
        for eps in epsilons:
            runner.update(epsilon=eps)
            runner.regenerate_qm = True
            runner.regenerate_cross_answers = True
            results = runner.run(
                extra_params={
                    "info": make_summary_dict(),
                    "run_set": "Airbnb-TEST_eps",
                }
            )
            run_count += 1
            print(
                f"eps: {runner.epsilon}, error_ave: {results['error_ave']}, error_max: {100 * np.max(np.array([np.sum(x) for x in results['errors']]))}"
            )
            print(f"###### COMPLETED {run_count} RUNS ######")
        reset_runner()
        for T_in in Ts:
            Tconst = T_in
            runner.regenerate_qm = True
            runner.regenerate_cross_answers = True
            results = runner.run(
                extra_params={
                    "info": make_summary_dict(),
                    "run_set": "Airbnb-TEST_Tfinal",
                },
                save_to="data/synthetic",
            )
            run_count += 1
            print(
                f"eps: {runner.epsilon}, error_ave: {results['error_ave']}, error_max: {100 * np.max(np.array([np.sum(x) for x in results['errors']]))}"
            )
            print(f"###### COMPLETED {run_count} RUNS ######")
        reset_runner()
        for g_in in g_rel_opts:
            g_rels = g_in
            runner.regenerate_qm = True
            runner.regenerate_cross_answers = True
            results = runner.run(
                extra_params={
                    "info": make_summary_dict(),
                    "run_set": "Airbnb-TEST_g_in",
                }
            )
            run_count += 1
            print(
                f"eps: {runner.epsilon}, error_ave: {results['error_ave']}, error_max: {100 * np.max(np.array([np.sum(x) for x in results['errors']]))}"
            )
            print(f"###### COMPLETED {run_count} RUNS ######")
        reset_runner()
        for q_in in q_reuses:
            q_reuse = q_in
            runner.regenerate_qm = True
            runner.regenerate_cross_answers = True
            results = runner.run(
                extra_params={
                    "info": make_summary_dict(),
                    "run_set": "Airbnb-TEST_q_reuse",
                }
            )
            run_count += 1
            print(
                f"eps: {runner.epsilon}, error_ave: {results['error_ave']}, error_max: {100 * np.max(np.array([np.sum(x) for x in results['errors']]))}"
            )
            print(f"###### COMPLETED {run_count} RUNS ######")
        reset_runner()
        for k_in in k_news:
            k_new = k_in
            runner.regenerate_qm = True
            runner.regenerate_cross_answers = True
            results = runner.run(
                extra_params={
                    "info": make_summary_dict(),
                    "run_set": "Airbnb-TEST_knew",
                }
            )
            run_count += 1
            print(
                f"eps: {runner.epsilon}, error_ave: {results['error_ave']}, error_max: {100 * np.max(np.array([np.sum(x) for x in results['errors']]))}"
            )
            print(f"###### COMPLETED {run_count} RUNS ######")
        reset_runner()
        for a_in in alphas:
            alpha = a_in
            runner.regenerate_qm = True
            runner.regenerate_cross_answers = True
            results = runner.run(
                extra_params={
                    "info": make_summary_dict(),
                    "run_set": "Airbnb-TEST_alpha",
                }
            )
            run_count += 1
            print(
                f"eps: {runner.epsilon}, error_ave: {results['error_ave']}, error_max: {100 * np.max(np.array([np.sum(x) for x in results['errors']]))}"
            )
            print(f"###### COMPLETED {run_count} RUNS ######")
        reset_runner()
