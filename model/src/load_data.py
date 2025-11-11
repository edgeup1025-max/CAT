import duckdb
from pathlib import Path
from enum import Enum
import pandas as pd
import numpy as np
from configparser import ConfigParser


class InputFormat(Enum):
    CSV = ".csv"
    PARQUET = ".parquet"
    JSON = ".json"
    JSONL = ".jsonl"
    TSV = ".tsv"


def config_reader(Path: str):
    try:
        return ConfigParser().read(Path)
    except Exception as e:
        print(e)


def local_duckdb(file_path: str, limit: int = None):
    con = duckdb.connect()
    ext = Path(file_path).suffix.lower()
    if ext in (InputFormat.CSV.value, InputFormat.TSV.value):
        query = f"SELECT * FROM read_csv_auto('{file_path}', nullstr='NULL')"
    elif ext == InputFormat.PARQUET.value:
        query = f"SELECT * FROM read_parquet('{file_path}')"
    elif ext in (InputFormat.JSON.value, InputFormat.JSONL.value):
        query = f"SELECT * FROM read_json_auto('{file_path}')"
    else:
        con.close()
        raise ValueError(f"Unsupported file format: {ext}")
    if limit:
        query += f" LIMIT {limit}"
    df = con.execute(query).df()
    con.close()
    return df


def irt_data_processor(thetas, alpha, beta):
    thetas = pd.DataFrame({'Theta': thetas})
    Alpha = pd.DataFrame({'Alpha': np.array(alpha.detach().numpy())})
    betas = pd.DataFrame({'b1': [], 'b2': [], 'b3': [], 'b4': []})
    for i, j in enumerate(beta.unbind()):
        betas.loc[i] = j.detach().numpy()
    return thetas, Alpha, betas
