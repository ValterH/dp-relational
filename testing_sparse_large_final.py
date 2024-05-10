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

def cross_generator_torch(qm, eps_rel, T):
    b_round = dp_relational.lib.synth_data.learn_relationship_vector_torch_paper_algo(qm, eps_rel, T=T,
                subtable_size=100000, verbose=True, device=device, queries_to_reuse=8)
    relationship_syn = dp_relational.lib.synth_data.make_synthetic_rel_table_sparse(qm, b_round)
    return relationship_syn

runner = ModelRunner()
runner.update(dataset_generator=dp_relational.data.movies.dataset, n_syn1=3880, n_syn2=6040,
              synth='mst', epsilon=4.0, eps1=1.0, eps2=1.0, k=3, dmax=10,
              qm_generator=qm_generator_torch, cross_generation_strategy=cross_generator_torch,
              T=60)
runner.load_artifacts('53b3b6a5-fe75-11ee-9cb7-a059507978f3')

epsilons = [4.0, 6.0]
run_count = 0
while True:
    for epsilon in epsilons:
        runner.update(epsilon=epsilon)
        runner.regenerate_qm = True
        results = runner.run(extra_params={ "run_set": -23 })
        print(runner.relationship_syn.shape[0])
        run_count += 1
        print(f"epsilon: {epsilon}, error_ave: {results['error_ave']}")
        print(f"###### COMPLETED {run_count} RUNS ######")