import dp_relational
import dp_relational.data.movies
import dp_relational.lib.qm
import dp_relational.lib.synth_data

import numpy as np
import time

# parameters

class ModelRunner:
    def __init__(self, *args, **kwargs) -> None:
        self.dataset_generator = None
        self.n_syn1 = None
        self.n_syn2 = None
        self.synth = None
        self.epsilon = None
        self.eps1 = None
        self.eps2 = None
        self.k = None
        self.dmax = None
        self.qm_generator = None
        self.cross_generation_strategy = None
        
        self.regenerate_dataset = False
        self.regenerate_syn_tables = False
        self.regenerate_qm = False
        self.regenerate_cross_answers = False
        
        self.rel_dataset = None
        self.df1_synth = None
        self.df2_synth = None
        self.qm = None
        self.relationship_syn = None
        
        self.T = None
        
        self.update(*args, **kwargs)
    
    def update(self, dataset_generator=None, n_syn1=None, n_syn2=None, synth=None,
                 epsilon=None, eps1=None, eps2=None, k=None,
                 dmax=None, qm_generator=None, cross_generation_strategy=None, T=None):
        if dataset_generator != self.dataset_generator or dmax != self.dmax:
            self.regenerate_dataset = True
        
        if n_syn1 != self.n_syn1 or n_syn2 != self.n_syn2 \
            or eps1 != self.eps1 or eps2 != self.eps2 or synth != self.synth:
            
            self.regenerate_syn_tables = True
        
        if qm_generator != self.qm_generator or k != self.k:
            self.regenerate_qm = True
        
        if epsilon != self.epsilon or cross_generation_strategy != self.cross_generation_strategy or T != self.T:
            self.regenerate_cross_answers = True
            
        # ensure that all downstream stages are forced to regenerate
        self.regenerate_syn_tables |= self.regenerate_dataset
        self.regenerate_qm |= self.regenerate_syn_tables
        self.regenerate_cross_answers |= self.regenerate_qm
        
        # actually copy in the values
        self.dataset_generator = self.dataset_generator if dataset_generator is None else dataset_generator
        self.n_syn1 = self.n_syn1 if n_syn1 is None else n_syn1
        self.n_syn2 = self.n_syn2 if n_syn2 is None else n_syn2
        self.synth = self.synth if synth is None else synth
        self.epsilon = self.epsilon if epsilon is None else epsilon
        self.eps1 = self.eps1 if eps1 is None else eps1
        self.eps2 = self.eps2 if eps2 is None else eps2
        self.k = self.k if k is None else k
        self.dmax = self.dmax if dmax is None else dmax
        self.qm_generator = self.qm_generator if qm_generator is None else qm_generator
        self.cross_generation_strategy = self.cross_generation_strategy if cross_generation_strategy is None else cross_generation_strategy
        self.T = self.T if T is None else T
    def run(self):
        self.times = {}
        
        class FuncTimer(object):
            def __init__(self, objin, name):
                self.objin = objin
                self.name = name
            def __enter__(self):
                self.time_start = time.perf_counter()
            def __exit__(self, exception_type, exception_value, traceback):
                time_end = time.perf_counter()
                self.objin[self.name] = time_end - self.time_start
        
        if self.regenerate_dataset:
            with FuncTimer(self.times, "dataset_generation"):
                self.rel_dataset = self.dataset_generator(self.dmax)
        
        if self.regenerate_syn_tables:
            with FuncTimer(self.times, "synth_table_generation"):
                self.df1_synth = dp_relational.lib.synth_data.compute_single_table_synth_data(
                    self.rel_dataset.table1.df, self.n_syn1, self.synth, epsilon=self.eps1)
                self.df2_synth = dp_relational.lib.synth_data.compute_single_table_synth_data(
                    self.rel_dataset.table2.df, self.n_syn2, self.synth, epsilon=self.eps2)
        
        if self.regenerate_qm:
            with FuncTimer(self.times, "qm_init"):
                self.qm = self.qm_generator(self.rel_dataset, k=self.k, df1_synth=self.df1_synth, df2_synth=self.df2_synth)
        
        if self.regenerate_cross_answers:
            with FuncTimer(self.times, "cross_answers_gen"):
                self.relationship_syn = self.cross_generation_strategy(self.qm, self.epsilon - self.eps1 - self.eps2, T=self.T)
        
        ave_error, errors = dp_relational.lib.synth_data.evaluate_synthetic_rel_table(self.qm, self.relationship_syn)
        
        return (self.times, (ave_error, errors), self.relationship_syn)
        
def qm_generator_basic(rel_dataset, k, df1_synth, df2_synth):
    return dp_relational.lib.synth_data.QueryManagerBasic(rel_dataset, k=k, df1_synth=df1_synth, df2_synth=df2_synth)

def qm_generator_torch(rel_dataset, k, df1_synth, df2_synth):
    return dp_relational.lib.synth_data.QueryManagerTorch(rel_dataset, k=k, df1_synth=df1_synth, df2_synth=df2_synth)

def cross_generator_basic(qm, eps_rel, T):
    b_round = dp_relational.lib.synth_data.learn_relationship_vector_basic(qm, eps_rel, T=T, verbose=True)
    relationship_syn = dp_relational.lib.synth_data.make_synthetic_rel_table(qm, b_round)
    return relationship_syn

def cross_generator_torch(qm, eps_rel, T):
    b_round = dp_relational.lib.synth_data.learn_relationship_vector_torch(qm, eps_rel, T=T, T_mirror=50, verbose=True)
    relationship_syn = dp_relational.lib.synth_data.make_synthetic_rel_table_sparse(qm, b_round)
    return relationship_syn

runner = ModelRunner()
runner.update(dataset_generator=dp_relational.data.movies.dataset, n_syn1=776, n_syn2=1208,
              synth='mst', epsilon=3.0, eps1=1.0, eps2=1.0, k=2, dmax=10,
              qm_generator=qm_generator_basic, cross_generation_strategy=cross_generator_basic)

epsilons = [3, 5, 7, 9, 12, 16]
results = []
for epsilon in epsilons:
    runner.update(epsilon=epsilon)
    results.append(runner.run())