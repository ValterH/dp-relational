import dp_relational
import dp_relational.data.movies
import dp_relational.lib.qm
import dp_relational.lib.synth_data

import numpy as np
import time
import uuid
import pickle

import os
from pathlib import Path

# parameters

DATASET_FOLDER = "datasets"
SYNTABLES_FOLDER = "syntables"
RELATIONSHIPS_FOLDER = "relationships"
RUNS_FOLDER = "runs"

class ModelRunner:
    def __init__(self, save_to="./runs", self_relation=False, *args, **kwargs) -> None:
        self.save_to = save_to
        self.self_relation = self_relation
        
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
        
        self.rel_dataset_runid = -1
        self.rel_dataset = None
        
        self.synth_tables_runid = -1
        self.df1_synth = None
        self.df2_synth = None
        self.qm = None
        
        self.relationship_syn_runid = -1
        self.relationship_syn = None
        
        self.T = None
        
        self.update(*args, **kwargs)
    
    def update(self, dataset_generator=None, n_syn1=None, n_syn2=None, synth=None,
                 epsilon=None, eps1=None, eps2=None, k=None,
                 dmax=None, qm_generator=None, cross_generation_strategy=None, T=None):
        dataset_generator = self.dataset_generator if dataset_generator is None else dataset_generator
        n_syn1 = self.n_syn1 if n_syn1 is None else n_syn1
        n_syn2 = self.n_syn2 if n_syn2 is None else n_syn2
        synth = self.synth if synth is None else synth
        epsilon = self.epsilon if epsilon is None else epsilon
        eps1 = self.eps1 if eps1 is None else eps1
        eps2 = self.eps2 if eps2 is None else eps2
        k = self.k if k is None else k
        dmax = self.dmax if dmax is None else dmax
        qm_generator = self.qm_generator if qm_generator is None else qm_generator
        cross_generation_strategy = self.cross_generation_strategy if cross_generation_strategy is None else cross_generation_strategy
        T = self.T if T is None else T
        
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
        self.dataset_generator = dataset_generator
        self.n_syn1 = n_syn1
        self.n_syn2 = n_syn2
        self.synth = synth
        self.epsilon = epsilon
        self.eps1 = eps1
        self.eps2 = eps2
        self.k = k
        self.dmax = dmax
        self.qm_generator = qm_generator
        self.cross_generation_strategy = cross_generation_strategy
        self.T = T
    
    def dump_parameters(self):
        return {
            "dataset_generator": self.dataset_generator.__name__,
            "n_syn1": self.n_syn1,
            "n_syn2": self.n_syn2,
            "synth": self.synth,
            "epsilon": self.epsilon,
            "eps1": self.eps1,
            "eps2": self.eps2,
            "k": self.k,
            "dmax": self.dmax,
            "qm_generator": self.qm_generator.__name__,
            "cross_generation_strategy": self.cross_generation_strategy.__name__,
            "T": self.T
        }
    
    def get_experiments(self, save_to=None):
        if save_to is None:
            save_to = self.save_to
        fpath = os.path.join(save_to, "runs")
        files = os.listdir(fpath)
        run_data = []
        for file in files:
            with open(os.path.join(fpath, file), "rb") as file_reader:
                run_data.append(pickle.load(file_reader))
        return run_data
    
    def load_artifacts(self, experiment_id, save_to=None):
        # try to find an artifact for each section
        dataset_path = os.path.join(self.save_to, DATASET_FOLDER, experiment_id + ".pkl")
        if os.path.isfile(dataset_path):
            print("Loaded dataset!")
            with open(dataset_path, "rb") as file_in:
                self.rel_dataset = pickle.load(file_in)
            self.regenerate_dataset = False
            self.rel_dataset_runid = uuid.UUID(experiment_id)
            
        syntable_path = os.path.join(self.save_to, SYNTABLES_FOLDER, experiment_id + ".pkl")
        if os.path.isfile(syntable_path):
            print("Loaded syntables!")
            with open(syntable_path, "rb") as file_in:
                dfs = pickle.load(file_in)
                self.df1_synth = dfs[0]
                self.df2_synth = dfs[1]
            self.regenerate_syn_tables = False
            self.synth_tables_runid = uuid.UUID(experiment_id)
        
        # TODO: load other artifacts...
    
    def run(self, extra_params={}, save_to=None):
        if save_to is None:
            save_to = self.save_to
        curr_run_id = uuid.uuid1()
        
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
            self.regenerate_dataset = False
            self.rel_dataset_runid = curr_run_id
            with FuncTimer(self.times, "dataset_generation"):
                self.rel_dataset = self.dataset_generator(self.dmax)
            # save it
            fpath = os.path.join(save_to, DATASET_FOLDER)
            os.makedirs(fpath, exist_ok=True)
            with open(os.path.join(fpath, f"{curr_run_id}.pkl"), 'wb') as f:
                pickle.dump(self.rel_dataset, f)
        
        if self.regenerate_syn_tables:
            self.regenerate_syn_tables = False
            self.synth_tables_runid = curr_run_id
            print(curr_run_id)
            with FuncTimer(self.times, "synth_table_generation"):
                if not self.self_relation:
                    self.df1_synth = dp_relational.lib.synth_data.compute_single_table_synth_data(
                        self.rel_dataset.table1.df, self.n_syn1, self.synth, epsilon=self.eps1)
                    self.df2_synth = dp_relational.lib.synth_data.compute_single_table_synth_data(
                        self.rel_dataset.table2.df, self.n_syn2, self.synth, epsilon=self.eps2)
                else:
                    self.df1_synth = dp_relational.lib.synth_data.compute_single_table_synth_data(
                        self.rel_dataset.table1.df, self.n_syn1, self.synth, epsilon=self.eps1)
                    self.df2_synth = self.df1_synth.copy()
            # save it
            fpath = os.path.join(save_to, SYNTABLES_FOLDER)
            os.makedirs(fpath, exist_ok=True)
            with open(os.path.join(fpath, f"{curr_run_id}.pkl"), 'wb') as f:
                pickle.dump([self.df1_synth, self.df2_synth], f)
        
        if self.regenerate_qm:
            self.regenerate_qm = False
            with FuncTimer(self.times, "qm_init"):
                self.qm = self.qm_generator(self.rel_dataset, k=self.k, df1_synth=self.df1_synth, df2_synth=self.df2_synth)
        
        if self.regenerate_cross_answers:
            self.regenerate_cross_answers = False
            self.relationship_syn_runid = curr_run_id
            with FuncTimer(self.times, "cross_answers_gen"):
                self.relationship_syn = self.cross_generation_strategy(self.qm, self.epsilon - self.eps1 - self.eps2, T=self.T)
            # save it
            fpath = os.path.join(save_to, RELATIONSHIPS_FOLDER)
            os.makedirs(fpath, exist_ok=True)
            with open(os.path.join(fpath, f"{curr_run_id}.pkl"), 'wb') as f:
                pickle.dump(self.relationship_syn, f)
        
        ave_error, errors = dp_relational.lib.synth_data.evaluate_synthetic_rel_table(self.qm, self.relationship_syn)
        to_return = {
            "run_id": curr_run_id,
            "parameters": self.dump_parameters(),
            "extra_params": extra_params,
            "times": self.times,
            "error_ave": ave_error,
            "errors": errors,
            "artifacts": {
                "rel_dataset": self.rel_dataset_runid,
                "syn_tables": self.synth_tables_runid,
                "relationship_syn": self.relationship_syn_runid
            }
        }
        fpath = os.path.join(save_to, RUNS_FOLDER)
        os.makedirs(fpath, exist_ok=True)
        with open(os.path.join(fpath, f"{curr_run_id}.pkl"), 'wb') as f:
            pickle.dump(to_return, f)
        
        return to_return