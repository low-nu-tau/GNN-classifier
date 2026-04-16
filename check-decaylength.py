# Quick check — what are the actual decay_length values in your data?
import sqlite3
import numpy as np
import pandas as pd

def check_decay_length(db_file, label=""):
    conn = sqlite3.connect(db_file)
    df   = pd.read_sql_query(
        "SELECT dbang_decay_length, energy, pid FROM truth LIMIT 1000", conn
    )
    conn.close()
    print(f"\n{label}")
    print(f"  decay_length min/max/mean : {df.dbang_decay_length.min():.2f} / "
          f"{df.dbang_decay_length.max():.2f} / {df.dbang_decay_length.mean():.2f}")
    print(f"  decay_length value counts (top 5):\n"
          f"{df.dbang_decay_length.value_counts().head()}")
    print(f"  n unique decay_length values: {df.dbang_decay_length.nunique()}")
    print(f"  PIDs: {df.pid.value_counts().to_dict()}")
    print(f"  energy min/max: {df.energy.min():.1f} / {df.energy.max():.1f} GeV")

check_decay_length(
    "/mnt/scratch/baburish/doublepulse/gnn/files/output/combined_nutau_5TeV.db",
    "tau 5TeV"
)
check_decay_length(
    "/mnt/scratch/baburish/doublepulse/gnn/files/output/combined_nue_5TeV.db",
    "nue 5TeV"
)