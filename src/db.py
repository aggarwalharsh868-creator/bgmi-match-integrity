import sqlite3, math
import pandas as pd
from . import config as C


def connect(db_path=C.DB_PATH):
    con = sqlite3.connect(db_path)
    con.create_function("SQRT", 1, lambda x: math.sqrt(x) if x is not None and x >= 0 else None)
    return con


def query(con, sql_file, params=None):
    return pd.read_sql_query((C.SQL_DIR / sql_file).read_text(), con, params=params or {})
