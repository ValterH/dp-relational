from dp_relational.lib.runner import ModelRunner

import matplotlib.pyplot as plt
import numpy as np

import uuid

def get_errors_by_eps_rel(runner, filter_exp=lambda x: True):
    experiments = runner.get_experiments(save_to="./runs")
    
    results = {}
    for experiment in experiments:
        if filter_exp(experiment):
            eps_rel = experiment['parameters']['epsilon'] - experiment['parameters']['eps1'] - experiment['parameters']['eps2']
            if not (eps_rel in results):
                results[eps_rel] = []
            results[eps_rel].append(experiment['error_ave'])

    epsilons = []
    ave_errors = []
    error_bars = []

    for epsilon, res_list in results.items():
        epsilons.append(epsilon)
        res_list_np = np.array(res_list)
        ave_errors.append(np.mean(res_list_np))
        error_bars.append(np.std(res_list_np, ddof=1))

    epsilons = np.array(epsilons)
    ave_errors = np.array(ave_errors)
    error_bars = np.array(error_bars)

    sorted_idxes = np.argsort(epsilons)
    epsilons = epsilons[sorted_idxes]
    ave_errors = ave_errors[sorted_idxes]
    error_bars = error_bars[sorted_idxes]

    return (epsilons, ave_errors, error_bars)

EPSILON_PYTORCH_EXPERIMENT_DATASET = uuid.UUID('037e4226-fd37-11ee-9e28-a059507978f3')
EPSILON_PYTORCH_SPARSE_EXPERIMENT_DATASET = uuid.UUID('8e8d44c1-fe1e-11ee-88ce-a059507978f3')
EPSILON_PYTORCH_SPARSE_LARGE_EXPERIMENT_DATASET = uuid.UUID('53b3b6a5-fe75-11ee-9cb7-a059507978f3')
EPSILON_PYTORCH_SPARSE_VLRGE_EXPERIMENT_DATASET = uuid.UUID('607ae43c-fece-11ee-bd8b-a059507978f3')

filter_exp_orig = lambda experiment: experiment['artifacts']['rel_dataset'] == EPSILON_PYTORCH_EXPERIMENT_DATASET
filter_exp_ojas = lambda experiment: experiment['artifacts']['rel_dataset'] == EPSILON_PYTORCH_SPARSE_EXPERIMENT_DATASET \
    and experiment['extra_params']['run_set'] == 2
filter_exp_masked_large = lambda experiment: experiment['artifacts']['rel_dataset'] == EPSILON_PYTORCH_SPARSE_LARGE_EXPERIMENT_DATASET \
    and experiment['extra_params']['run_set'] == 11
filter_exp_masked_vlrge = lambda experiment: experiment['artifacts']['rel_dataset'] == EPSILON_PYTORCH_SPARSE_VLRGE_EXPERIMENT_DATASET \
    and experiment['extra_params']['run_set'] == 20

runner = ModelRunner(save_to='./runs')

epsilons, ave_errors, error_bars = get_errors_by_eps_rel(runner, filter_exp_orig)
epsilons_o, ave_errors_o, error_bars_o = get_errors_by_eps_rel(runner, filter_exp_masked_large)
epsilons_l, ave_errors_l, error_bars_l = get_errors_by_eps_rel(runner, filter_exp_masked_vlrge)

plt.figure()
plt.errorbar(epsilons, ave_errors, error_bars, fmt='-o', capsize=5, label="Original generation")
plt.errorbar(epsilons_o, ave_errors_o, error_bars_o, fmt='-o', capsize=5, label="Ojas masking method")
plt.errorbar(epsilons_l, ave_errors_l, error_bars_l, fmt='-o', capsize=5, label="Masking large dataset")
plt.title("Average workload error vs privacy budget")
plt.xlabel("Relational privacy budget")
plt.ylabel("Average % error")
plt.legend()
plt.grid()
plt.show()