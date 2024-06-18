"""
Contains classes for data management
"""
import numpy as np
from tqdm import tqdm

def make_unique_ints(df, col):
    unique_ids = set(i[col]
                        for i in df.to_dict('records'))
    ids_to_ints = {j: i for i, j in enumerate(unique_ids)}
    new_col = [ids_to_ints[i] for i in df[col]]
    return new_col, ids_to_ints

def map_unique_ints(df, col, ids_to_ints):
    new_col = [ids_to_ints[i] for i in df[col]]
    return new_col

def remove_excess_rows(df, column, k):
    counts = df.groupby(column).cumcount()
    return df[counts < k].reset_index()

class Table:
    def __init__(self, df, id_col, do_onehot_encode=None):
        self.df = df
        self.id_col = id_col
        self.col_dict = None # created by make_column_dict
        
        columns_to_encode = do_onehot_encode
        if columns_to_encode is None:
            # encode all columns
            columns_to_encode = []
            for col in self.df.columns:
                if col is not id_col:
                    columns_to_encode.append(col)
        if id_col is not None:
            columns_to_encode.append(id_col)
        
        self.column_lookups = {}
        for column in tqdm(columns_to_encode):
            self.df[column], self.column_lookups[column] = make_unique_ints(self.df, column)
        
    def make_column_dict(self):
        """Calculates a column dict for the columns in the table, storing the number of
        unique values and their identities"""
        self.col_dict = {col: 
                {'unique_values': list(np.sort(self.df[col].unique())), 'count': len(self.df[col].unique())} 
            for col in self.df.columns}

class RelationalDataset:
    def __init__(self, table1: Table, table2: Table, df_rel, rel_id1_col, rel_id2_col, dmax=10) -> None:
        # prepare the cross table
        df_rel[rel_id1_col] = map_unique_ints(df_rel, rel_id1_col, table1.column_lookups[table1.id_col])
        df_rel[rel_id2_col] = map_unique_ints(df_rel, rel_id2_col, table2.column_lookups[table2.id_col])
        df_rel = df_rel[[rel_id1_col, rel_id2_col]]
        # drop id columns from tables, they are no longer needed
        table1.df = table1.df.drop(table1.id_col, axis=1)
        table2.df = table2.df.drop(table2.id_col, axis=1)
        
        self.table1 = table1
        self.table2 = table2
        
        self.df_rel = df_rel
        self.rel_id1_col = rel_id1_col
        self.rel_id2_col = rel_id2_col
        
        self.dmax = dmax
        
        self.df_rel = remove_excess_rows(self.df_rel, self.rel_id1_col, dmax)
        self.df_rel = remove_excess_rows(self.df_rel, self.rel_id2_col, dmax) # limit relations
        
        #self.B = np.zeros((len(table1.df), len(table2.df)))
        #self.B[df_rel[rel_id1_col], df_rel[rel_id2_col]] = 1
        
        # make column dicts for the first two tables
        table1.make_column_dict()
        table2.make_column_dict()
        