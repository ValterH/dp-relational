import json

import numpy as np

from pathlib import Path
from syntherela.data import load_tables
from syntherela.metadata import Metadata
from dp_relational.lib.dataset import Table, RelationalDataset


data_path = "data/original/airbnb-simplified_subsampled"


def fill_nans(df, col):
    if df[col].isnull().any():
        col_mean = df[col].mean()
        df[col + "_missing"] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(col_mean)
    return df


def dataset(dmax):
    metadata = Metadata().load_from_json(Path(data_path) / "metadata.json")
    tables = load_tables(Path(data_path), metadata)

    users_df = tables["users"]
    sessions_df = tables["sessions"]

    categorical_columns_users = metadata.get_column_names("users", sdtype="categorical")
    categorical_columns_sessions = metadata.get_column_names(
        "sessions", sdtype="categorical"
    )
    datetime_columns_users = metadata.get_column_names("users", sdtype="datetime")

    # convert datetime columns to integer timestamps
    for col in datetime_columns_users:
        users_df[col] = users_df[col].astype("int64")

    # convert categorical columns to strings
    for col in categorical_columns_users:
        users_df[col] = users_df[col].astype(str)

    # convert categorical columns to strings
    for col in categorical_columns_sessions:
        sessions_df[col] = sessions_df[col].astype(str)

    numeric_columns_users = metadata.get_column_names("users", sdtype="numerical")
    numeric_columns_sessions = metadata.get_column_names("sessions", sdtype="numerical")

    numeric_columns_users += datetime_columns_users

    bins = {
        "users": {},
        "sessions": {},
    }

    # fill nan values and create boolean columns for missing values
    for col in numeric_columns_users:
        if users_df[col].isnull().any():
            users_df = fill_nans(users_df, col)

        if len(users_df[col].unique()) > 5:
            # discretize the data into 5 bins
            bins["users"][col] = np.linspace(
                users_df[col].min(), users_df[col].max(), 6
            ).tolist()
            users_df[col] = (
                np.digitize(
                    users_df[col],
                    bins=bins["users"][col],
                )
                - 1
            )

    for col in numeric_columns_sessions:
        if sessions_df[col].isnull().any():
            sessions_df = fill_nans(sessions_df, col)
        if sessions_df[col].isnull().any():
            sessions_df = fill_nans(sessions_df, col)

        if len(sessions_df[col].unique()) > 5:
            # discretize the data into 5 bins
            bins["sessions"][col] = np.linspace(
                sessions_df[col].min(), sessions_df[col].max(), 6
            ).tolist()
            sessions_df[col] = (
                np.digitize(
                    sessions_df[col],
                    bins=bins["sessions"][col],
                )
                - 1
            )
        else:
            pass

    with open(Path(data_path) / "bins.json", "w") as f:
        json.dump(bins, f, indent=4)

    pk_users = metadata.get_primary_key("users")
    # sessions does not have a primary key, so we create one
    pk_sessions = "SessionID"
    sessions_df[pk_sessions] = sessions_df.index

    users_table = Table(users_df, pk_users, do_onehot_encode=categorical_columns_users)

    sessions_table = Table(
        sessions_df,
        pk_sessions,
        do_onehot_encode=categorical_columns_sessions,
    )

    fk_users = metadata.get_foreign_keys("users", "sessions")[0]

    df_rel = sessions_df[[pk_sessions, fk_users]]

    sessions_df.drop(columns=[fk_users], inplace=True, errors="ignore")

    return RelationalDataset(
        users_table,
        sessions_table,
        df_rel,
        rel_id1_col=fk_users,
        rel_id2_col=pk_sessions,
        dmax=dmax,
    )
