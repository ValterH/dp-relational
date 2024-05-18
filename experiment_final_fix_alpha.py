from dp_relational.lib.runner import ModelRunner

import dp_relational.lib.synth_data
import dp_relational.data.movies

import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("cuda_available", torch.cuda.is_available())
print("using device: ", device)

# EPSILON_PYTORCH_EXPERIMENT_DATASET = uuid.UUID('53b3b6a5-fe75-11ee-9cb7-a059507978f3')

def qm_generator_torch(rel_dataset, k, df1_synth, df2_synth):
    return dp_relational.lib.synth_data.QueryManagerTorch(rel_dataset, k=k, df1_synth=df1_synth, df2_synth=df2_synth, device=device)

alpha_value = -1
def cross_generator_torch(qm, eps_rel, T):
    print("in_gen", alpha_value)
    b_round = dp_relational.lib.synth_data.learn_relationship_vector_torch_paper_algo(qm, eps_rel, T=T, k_new_queries=2,
                subtable_size=100000, verbose=True, device=device, queries_to_reuse=2, exp_mech_alpha=alpha_value)
    relationship_syn = dp_relational.lib.synth_data.make_synthetic_rel_table_sparse(qm, b_round)
    return relationship_syn

runner = ModelRunner()
runner.update(dataset_generator=dp_relational.data.movies.dataset, n_syn1=3880, n_syn2=6040,
              synth='mst', epsilon=4.0, eps1=1.0, eps2=1.0, k=3, dmax=10,
              qm_generator=qm_generator_torch, cross_generation_strategy=cross_generator_torch,
              T=110)
runner.load_artifacts('0c96cc62-014b-11ef-9020-d21cd07a44f3')
print("hiii!")
alphas = [0.000001, 0.1, 0.2, 0.3, 0.4]
run_count = 0
while True:
    for alpha in alphas:
        alpha_value = alpha
        runner.regenerate_qm = True
        runner.regenerate_cross_answers = True
        results = runner.run(extra_params={ "run_set": "final_fix_reusing2_with_expmech_3", "alpha": alpha_value })
        print(runner.relationship_syn.shape[0])
        run_count += 1
        print(f"alpha: {alpha_value}, error_ave: {results['error_ave']}")
        print(f"###### COMPLETED {run_count} RUNS ######")