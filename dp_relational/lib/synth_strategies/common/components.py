class LearningEnvironment:
    def __init__(self) -> None:
        self.n_relationship_synt = None
        self.m = None
        self.table1_slice_size = None
        self.table2_slice_size = None
        self.cross_slice_size = None
        
        self.exp_mech_factor = None
        self.gm_stddev = None
        
        self.unselected_workload = None # Numbers of all workloads that have not yet been selected
        self.selected_workloads = None # Numbers of all workloads that have been chosen
        self.noisy_ans_list = None # Noisy answers to all currently selected workloads
        
        # per slice variables
        self.table1_idxes = None # Locations of relationships in both tables 1 and 2
        self.table2_idxes = None # in the full b_round vector
        
        self.slice_table1 = None # Which indices were chosen from each table
        self.slice_table2 = None # in the randomly selected pair?
        self.offsets = None      # Locations of the relations between the slices in the b vector

def common_synth_strategy()