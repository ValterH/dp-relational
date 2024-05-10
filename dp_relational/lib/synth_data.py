from snsynth import Synthesizer
from .dataset import RelationalDataset
import itertools
import random
import numpy as np
import pandas as pd

import torch

from .helpers import cdp_delta, cdp_eps, cdp_rho, get_per_round_privacy_budget, torch_cat_sparse_coo, get_relationships_from_sparse

from tqdm import tqdm

from .qm import QueryManager, QueryManagerBasic, QueryManagerTorch

import gc

# import linecache
# import os
# import tracemalloc

# for debugging purposes
# tracemalloc.start()


# debugging function

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
from .synth_strategies.basic_algo import learn_relationship_vector_basic

from .synth_strategies.torch_algo import learn_relationship_vector_torch

from .synth_strategies.torch_masked_algo import learn_relationship_vector_torch_masked

from .synth_strategies.torch_masked_query_reuse import learn_relationship_vector_torch_masked_query_reuse

from .synth_strategies.torch_paper_algo import learn_relationship_vector_torch_paper_algo

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

def make_synthetic_rel_table_sparse(qm: QueryManager, b_round: torch.Tensor):
    ID_1 = qm.rel_dataset.rel_id1_col
    ID_2 = qm.rel_dataset.rel_id2_col
    
    table1_nums, table2_nums = get_relationships_from_sparse(qm, b_round)
    
    relationship_syn = pd.DataFrame(data={
        ID_1: table1_nums,
        ID_2: table2_nums
    })

    return relationship_syn

def evaluate_synthetic_rel_table(qm: QueryManager, relationship_syn):
    ans_syn = qm.calculate_ans_from_rel_dataset(relationship_syn, is_synth=True)
    
    errors = np.abs(ans_syn - qm.true_ans)
    ave_error =100 * np.sum(errors) / len(qm.true_ans)
    
    return (ave_error, errors)