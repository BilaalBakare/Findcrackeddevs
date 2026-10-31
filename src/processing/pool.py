from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

df = pd.read_parquet(ROOT / "data" / "raw" / "archive.parquet")

active = df[(df['active_days'] >= 3) & (df['n_events'] >= 5) & (df['n_events'] <= 400)]


active["hash_val"] = pd.util.hash_pandas_object(active["actor_id"], index=False)
active["bucket"] = active["hash_val"] % 1000

conditions = [
    active["bucket"] <= 3,
    active["bucket"] <= 99,
    active["bucket"] < 900,
    active["bucket"] >= 900,
]

choices = ["pilot", "main", "reserve", "control"]
active["bucket_cond"] = np.select(conditions, choices, default="unknown")

pool = active[active["bucket_cond"].isin(["pilot", "main"])]
(ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
pool.to_parquet(ROOT / "data" / "processed" / "pool.parquet", index=False)