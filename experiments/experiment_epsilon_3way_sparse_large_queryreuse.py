from dp_relational.lib.runner import ModelRunner

import dp_relational.lib.synth_data
import dp_relational.data.movies

import torch

if not torch.cuda.is_available():
    print("Cuda not found!")
    raise RuntimeError("No cuda found.")

device = torch.device('cuda')
print("cuda_available", torch.cuda.is_available())
print("using device: ", device)

# EXPERIMENT_DATASET = uuid.UUID('not-run-yet')

def qm_generator_torch(rel_dataset, k, df1_synth, df2_synth):
    return dp_relational.lib.synth_data.QueryManagerTorch(rel_dataset, k=k, df1_synth=df1_synth, df2_synth=df2_synth, device=device)

def cross_generator_torch(qm, eps_rel, T):
    b_round = dp_relational.lib.synth_data.learn_relationship_vector_torch_masked_query_reuse(qm, eps_rel, T=T, queries_to_reuse=8, T_mirror=150, subtable_size=200000, verbose=True, device=device)
    relationship_syn = dp_relational.lib.synth_data.make_synthetic_rel_table_sparse(qm, b_round)
    return relationship_syn
    
def cross_generator_torch2(qm, eps_rel, T):
    b_round = dp_relational.lib.synth_data.learn_relationship_vector_torch_masked_query_reuse(qm, eps_rel, T=T, queries_to_reuse=1, T_mirror=150, subtable_size=200000, verbose=True, device=device)
    relationship_syn = dp_relational.lib.synth_data.make_synthetic_rel_table_sparse(qm, b_round)
    return relationship_syn

runner = ModelRunner()
runner.update(dataset_generator=dp_relational.data.movies.dataset, n_syn1=int(7760 / 2), n_syn2=int(12080 / 2),
              synth='mst', epsilon=4.0, eps1=1.0, eps2=1.0, k=3, dmax=10,
              qm_generator=qm_generator_torch, cross_generation_strategy=cross_generator_torch,
              T=40)
runner.load_artifacts('0c96cc62-014b-11ef-9020-d21cd07a44f3')

epsilons = [2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0]
run_count = 0
generators = [cross_generator_torch, cross_generator_torch2]
while True:
    for epsilon in epsilons:
        for x in range(2):
            runner.update(epsilon=epsilon, cross_generation_strategy=generators[x])
            runner.regenerate_qm = True
            results = runner.run(extra_params={ "T_mirror": 150, "run_set": 320 + x, "q_reused": [8, 1][x] })
            print(runner.relationship_syn.shape[0])
            run_count += 1
            print(f"epsilon: {epsilon}, error_ave: {results['error_ave']}")
            print(f"###### COMPLETED {run_count} RUNS ######")
