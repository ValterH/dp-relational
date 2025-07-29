from pathlib import Path
from syntherela.data import load_tables
from syntherela.metadata import Metadata
from dp_relational.lib.dataset import Table, RelationalDataset


data_path = "data/original/airbnb-simplified_subsampled"


def dataset(dmax):
    metadata = Metadata().load_from_json(Path(data_path) / "metadata.json")
    tables = load_tables(Path(data_path), metadata)

    users_df = tables["users"]
    sessions_df = tables["sessions"]

    categorical_columns_users = metadata.get_column_names("users", sdtype="categorical")

    # convert categorical columns to strings
    for col in categorical_columns_users:
        users_df[col] = users_df[col].astype(str)
        # # fill NaN values with "unknown"
        # users_df[col] = users_df[col].fillna("unknown")

    categorical_columns_sessions = metadata.get_column_names(
        "sessions", sdtype="categorical"
    )

    # convert categorical columns to strings
    for col in categorical_columns_sessions:
        sessions_df[col] = sessions_df[col].astype(str)
        # # fill NaN values with "unknown"
        # sessions_df[col] = sessions_df[col].fillna("unknown")

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
