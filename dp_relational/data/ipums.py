import pandas as pd
from dp_relational.lib.dataset import Table, RelationalDataset

from ipumspy import readers, ddi

import os

# Shorthand for joining a list of paths
p_join = os.path.join
root_path = p_join(os.path.dirname(os.path.realpath(__file__)), "datasets")
PICKLE_PATH = p_join(root_path, "ipums/usa_00001.pkl")

def preprocess():
    # unzip the dat file
    DAT_PATH = p_join(root_path, "ipums/usa_00001.dat")
    import gzip
    import shutil
    with gzip.open(p_join(root_path, "ipums/usa_00001.dat.gz"), 'rb') as f_in:
        with open(DAT_PATH, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        # Saves the thingy as a dataframe
    ddi_codebook = readers.read_ipums_ddi(p_join(root_path, "ipums/usa_00001.dat.xml"))
    ipums_df = readers.read_microdata(ddi_codebook, DAT_PATH)
    
    ipums_df.to_pickle(PICKLE_PATH)
    os.remove(DAT_PATH)
def dataset(dmax):
    # todo: doublecheck dmax
    # we should be good...
    df = pd.read_pickle(PICKLE_PATH)
    return df # just for debugging for now