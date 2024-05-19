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
def dataset(dmax=4, frac=1,
            parent_vars=['PERNUM_MOM', 'PERNUM_POP', 'PERNUM_MOM2', 'PERNUM_POP2'],
            table_cols=['SEX', 'MARST', 'MARRINYR', 'RACAMIND', 'RACASIAN', 'RACBLK', \
                'RACPACIS', 'RACWHT', 'RACOTHER']):
    # todo: doublecheck dmax
    # we should be good...
    df = pd.read_pickle(PICKLE_PATH)
    
    print("Original size:", int(len(df.index)), ", using size:", int(len(df.index) * frac))
    if frac != 1:
        df = df.head(int(len(df.index) * frac))
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
    
    df.set_index('PK')
    col_slice = ['PK', 'SAMPLE', 'SERIAL', 'PERNUM'] + table_cols + parent_vars
    df = df[col_slice]
    
    # print("df:::")
    # print(df[df['MOMLOC'] != 0][['PK', 'SAMPLE', 'SERIAL', 'PERNUM', 'MOMLOC', 'POPLOC', 'PERNUM_MOM', 'PERNUM_POP', 'PERNUM_MOM2', 'PERNUM_POP2']])
    # print(202201314602 in df)
    # print(202201314602 in df['PK'].values)
    
    records = len(df.index)
    rel_np = np.zeros((2, records * len(parent_vars)), dtype=np)
    missing_rels = 0
    curr_rel_idx = 0
    
    # ojas cursed magic (TM)
    # i will be exclusively using "Cursed Magic" as a term for convoluted vectorization
    # this turned out to be very uncursed, thank you pandas!
    for parent_var in parent_vars:
        slice_df = df[df[parent_var].notna()]
        slice_df[parent_var] = sample_serial_pernum_conv(slice_df['SAMPLE'], slice_df['SERIAL'], slice_df[parent_var])
        slice_df = slice_df[slice_df[parent_var].isin(df['PK'])]
        slice_size = len(slice_df.index)
        rel_np[0, curr_rel_idx:curr_rel_idx + slice_size] = slice_df['PK'].to_numpy()
        rel_np[1, curr_rel_idx:curr_rel_idx + slice_size] = slice_df[parent_var].to_numpy()
        curr_rel_idx = curr_rel_idx + slice_size
    
    # for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    #     for var in parent_vars:
    #         if not pd.isna(row[var]):
    #             curr_rel_pk = sample_serial_pernum_conv(row['SAMPLE'], row['SERIAL'], row[var])
    #             if curr_rel_pk in df['PK'].values:
    #                 _add_rel_np(row['PK'], curr_rel_pk)
    #             else:
    #                 missing_rels += 1
    
    df = df.drop(columns=parent_vars)
    df = df.drop(columns=['SAMPLE', 'SERIAL', 'PERNUM'])
    # print(df.columns)
    print(f"Missing rels: {missing_rels}. Found rels: {curr_rel_idx}")
    
    df_rel = pd.DataFrame({'CHILDKEY': rel_np[0, :curr_rel_idx], 'PARENTKEY': rel_np[1, :curr_rel_idx]})
    
    df_new = df.copy()
    # print(df_new)
    people_table = Table(df, 'PK')
    people_table2 = Table(df_new, 'PK')
    # print(people_table.df)
    
    return RelationalDataset(people_table, people_table2, df_rel, 'CHILDKEY', 'PARENTKEY', dmax=dmax) # just for debugging for now