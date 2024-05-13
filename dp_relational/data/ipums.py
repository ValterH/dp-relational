import pandas as pd
import numpy as np
from tqdm import tqdm
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
def dataset(dmax=10):
    # todo: doublecheck dmax
    # we should be good...
    df = pd.read_pickle(PICKLE_PATH)
    all_cols = ['YEAR', 'SAMPLE', 'SERIAL', 'CBSERIAL', 'HHWT', 'CLUSTER', 'METRO', 'STRATA',\
        'GQ', 'FARM', 'OWNERSHP', 'OWNERSHPD', 'FRIDGE', 'PHONE', 'CINETHH', 'CILAPTOP',\
            'CIHISPEED', 'VEHICLES', 'SSMC', 'PERNUM', 'PERWT', 'FAMUNIT', 'MOMLOC', 'POPLOC',\
                'SEX', 'MARST', 'MARRINYR', 'LANGUAGE', 'LANGUAGED', 'RACAMIND', 'RACASIAN',\
                    'RACBLK', 'RACPACIS', 'RACWHT', 'RACOTHER', 'HCOVANY', 'HCOVPRIV', 'HINSEMP',\
                        'EMPSTAT', 'EMPSTATD', 'PERNUM_HEAD', 'PERNUM_MOM', 'PERNUM_POP',\
                            'PERNUM_SP', 'PERNUM_MOM2', 'PERNUM_POP2']
    person_cols = ['SAMPLE', 'SERIAL', 'PERNUM', 'PERWT', 'FAMUNIT', 'MOMLOC', 'POPLOC', \
        'SEX', 'MARST', 'MARRINYR', 'LANGUAGE', 'LANGUAGED', 'RACAMIND', 'RACASIAN',\
            'RACBLK', 'RACPACIS', 'RACWHT', 'RACOTHER', 'HCOVANY', 'HCOVPRIV',\
                'HINSEMP', 'EMPSTAT', 'EMPSTATD', 'PERNUM_HEAD', 'PERNUM_MOM', 'PERNUM_POP',\
                    'PERNUM_SP', 'PERNUM_MOM2', 'PERNUM_POP2']
    
    max_pernum = df['PERNUM'].max()
    max_serial = df['SERIAL'].max()
    # max_sample = df['SAMPLE'].max()

    pernum_mul = 10 ** (int(np.log10(max_pernum)) + 1)
    serial_mul = 10 ** (int(np.log10(max_serial)) + 1)

    def sample_serial_pernum_conv(sample, serial, pernum):
        return (sample * (pernum_mul * serial_mul)) + (serial * pernum_mul) + pernum

    df['PK'] = sample_serial_pernum_conv(df['SAMPLE'], df['SERIAL'], df['PERNUM'])
    df.set_index(['PK'])
    df = df[['PK', 'SAMPLE', 'SERIAL', 'PERNUM', 'SEX', 'MARST', 'MARRINYR', \
        'RACAMIND', 'RACASIAN', 'RACBLK', 'RACPACIS', 'RACWHT', 'RACOTHER', \
            'HINSEMP', 'EMPSTAT', 'PERNUM_MOM', 'PERNUM_POP', 'PERNUM_MOM2', 'PERNUM_POP2']]
    
    records = len(df.index)
    rel_np = np.zeros((2, records * 4), dtype=np)
    curr_rel_idx = 0
    
    def _add_rel_np(a, b):
        nonlocal rel_np
        nonlocal curr_rel_idx
        rel_np[0, curr_rel_idx] = a
        rel_np[1, curr_rel_idx] = b
        curr_rel_idx += 1
    
    parent_vars = ['PERNUM_MOM', 'PERNUM_POP', 'PERNUM_MOM2', 'PERNUM_POP2']
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        for var in parent_vars:
            if not pd.isna(row[var]):
                _add_rel_np(row['PK'], sample_serial_pernum_conv(row['SAMPLE'], row['SERIAL'], row[var]))
    
    df.drop(columns=parent_vars)
    
    df_rel = pd.DataFrame({'CHILDKEY': rel_np[0, :curr_rel_idx], 'PARENTKEY': rel_np[1:curr_rel_idx]})
    
    people_table = Table(df, 'PK')
    
    return RelationalDataset(people_table, people_table, df_rel, 'CHILDKEY', 'PARENTKEY', dmax=dmax) # just for debugging for now