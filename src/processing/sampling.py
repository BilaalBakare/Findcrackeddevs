from pathlib import Path

import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / ".git").exists())
profiles = pd.read_parquet(ROOT / "data" / "processed" / "profiles.parquet")

QUOTA = 150
EXCLUDE = {"unclassified", "unavailable", "learning"}
MIN_REPOS_USED = 2
SALT = "corpus-v1"

# 1) eligibility
eligible = profiles[
    ~profiles["label"].isin(EXCLUDE) & (profiles["n_repos_used"] >= MIN_REPOS_USED)
].copy()

# 2) deterministic, independent ordering inside each label
eligible["sample_key"] = pd.util.hash_pandas_object(
    eligible["actor_id"].astype(str) + "|" + SALT, index=False
)
eligible = eligible.sort_values("sample_key")

# 3) take the first QUOTA of each label (everyone, if the label is smaller)
corpus = eligible.groupby("label").head(QUOTA)

# 4) report shortfalls: a finding about your frame, not a bug
available = eligible["label"].value_counts()
taken = corpus["label"].value_counts()
report = pd.DataFrame({"available": available, "taken": taken})
report["shortfall"] = (QUOTA - report["available"]).clip(lower=0)
print(report.sort_values("available", ascending=False))
print("corpus size:", len(corpus))

corpus.to_parquet(ROOT / "data" / "processed" / "corpus.parquet", index=False)