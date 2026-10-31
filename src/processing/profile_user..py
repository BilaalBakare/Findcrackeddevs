import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# ---------- paths and settings ----------
ROOT = Path(__file__).resolve().parents[2]
POOL_FILE = ROOT / "data" / "processed" / "pool.parquet"
RAW_DIR = ROOT / "data" / "raw" / "repos"
OUT_FILE = ROOT / "data" / "processed" / "profiles.parquet"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TOKEN = os.getenv("GITHUB_TOKEN")
URL = "https://api.github.com/users/{login}/repos"
PARAMS = {"per_page": 100, "sort": "pushed", "type": "owner"}
MAX_RETRIES = 4
LOW_REMAINING = 20
MIN_SCORE = 3.0

# ---------- the dictionary ----------
DOMAINS = {
    "ml_ai": {
        "strong": ["machine learning", "deep learning", "pytorch", "tensorflow",
                   "neural network", "transformers", "llm", "nlp", "computer vision",
                   "reinforcement learning", "scikit learn", "huggingface", "diffusion"],
        "weak": ["ai", "model", "training", "dataset", "classification", "embedding"],
        "langs": ["jupyter notebook"],
    },
    "data_eng": {
        "strong": ["data pipeline", "etl", "airflow", "spark", "kafka", "dbt",
                   "data warehouse", "data engineering", "bigquery", "snowflake"],
        "weak": ["pipeline", "analytics", "pandas", "sql", "dashboard"],
        "langs": [],
    },
    "frontend": {
        "strong": ["react", "vue", "angular", "svelte", "nextjs", "tailwind",
                   "frontend", "css", "webpack", "vite", "web components"],
        "weak": ["ui", "website", "portfolio", "component", "landing page"],
        "langs": ["typescript", "css", "html", "vue", "svelte"],
    },
    "backend": {
        "strong": ["backend", "rest api", "graphql", "django", "fastapi", "flask",
                   "spring boot", "express", "microservice", "grpc", "postgresql"],
        "weak": ["api", "server", "database", "auth", "redis", "mongodb"],
        "langs": ["java", "go", "php", "ruby", "c#"],
    },
    "devops_cloud": {
        "strong": ["kubernetes", "terraform", "docker", "devops", "ansible", "helm",
                   "ci cd", "github actions", "aws", "gcp", "azure",
                   "infrastructure as code", "prometheus", "grafana"],
        "weak": ["deploy", "container", "cloud", "monitoring", "pipeline"],
        "langs": ["hcl", "dockerfile", "shell"],
    },
    "mobile": {
        "strong": ["android", "ios", "flutter", "react native", "swiftui", "kotlin",
                   "mobile app", "jetpack compose", "xcode"],
        "weak": ["mobile", "app"],
        "langs": ["swift", "kotlin", "dart", "objective c"],
    },
    "systems": {
        "strong": ["compiler", "operating system", "kernel", "embedded", "rtos",
                   "memory allocator", "systems programming", "firmware",
                   "interpreter", "distributed systems", "concurrency"],
        "weak": ["performance", "low level", "cli", "parser"],
        "langs": ["c", "c++", "rust", "zig", "assembly"],
    },
    "security": {
        "strong": ["security", "pentest", "penetration testing", "malware",
                   "cryptography", "vulnerability", "exploit", "ctf",
                   "reverse engineering", "osint"],
        "weak": ["hacking", "encryption", "scanner"],
        "langs": [],
    },
}

JUNK_NAMES = {"dotfiles", "test", "tests", "hello world", "helloworld", "demo",
              "playground", "sandbox", "learning", "practice", "untitled"}


# ---------- text matching ----------
def normalise(text):
    return re.sub(r"[-_/]+", " ", (text or "").lower())


def compile_term(term):
    return re.compile(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])")


PATTERNS = {
    domain: {
        "strong": [compile_term(t) for t in spec["strong"]],
        "weak": [compile_term(t) for t in spec["weak"]],
        "langs": {l.lower() for l in spec["langs"]},
    }
    for domain, spec in DOMAINS.items()
}


# ---------- stage 1: fetching ----------
def make_session():
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    return s


def sleep_until_reset(resp):
    reset = int(resp.headers.get("X-RateLimit-Reset", 0))
    wait = max(reset - time.time(), 0) + 2
    if "Retry-After" in resp.headers:
        wait = max(wait, int(resp.headers["Retry-After"]))
    print(f"  rate limit hit, sleeping {wait / 60:.1f} min")
    time.sleep(wait)


def save_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def fetch_repos(session, actor_id, login):
    path = RAW_DIR / f"{actor_id}.json"
    if path.exists():
        return "skipped"

    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(URL.format(login=login), params=PARAMS, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue

        remaining = r.headers.get("X-RateLimit-Remaining")
        status = r.status_code

        if status == 200:
            repos = r.json()
            save_json(path, {"login": login, "repos": repos})
            if remaining is not None and int(remaining) < LOW_REMAINING:
                sleep_until_reset(r)
            return "fetched" if repos else "empty"

        if status in (404, 451):
            save_json(path, {"login": login, "error": status, "repos": []})
            return "gone"

        if status in (403, 429):
            if remaining == "0" or "Retry-After" in r.headers:
                sleep_until_reset(r)
                continue
            return "failed"

        if status >= 500:
            time.sleep(2 ** attempt)
            continue

        return "failed"

    return "failed"


# ---------- stage 2: scoring ----------
def usable(repo):
    if repo.get("fork"):
        return False
    if normalise(repo.get("name")) in JUNK_NAMES:
        return False
    if not (repo.get("description") or repo.get("topics") or repo.get("language")):
        return False
    return True


def score_repo(repo):
    topics = normalise(" ".join(repo.get("topics") or []))
    text = normalise(f"{repo.get('name', '')} {repo.get('description') or ''}")
    lang = (repo.get("language") or "").lower()

    points = {}
    for domain, p in PATTERNS.items():
        best = 0.0
        if any(rx.search(topics) for rx in p["strong"]):
            best = 3.0
        elif any(rx.search(text) for rx in p["strong"]):
            best = 2.0
        elif any(rx.search(topics) or rx.search(text) for rx in p["weak"]):
            best = 1.0
        if lang in p["langs"]:
            best += 0.5
        points[domain] = best
    return points


def score_developer(repos):
    totals = Counter({d: 0.0 for d in DOMAINS})
    used = [r for r in repos if usable(r)]
    for repo in used:
        totals.update(score_repo(repo))
    return totals, len(used)


def assign_label(totals):
    top_domain, top_score = max(totals.items(), key=lambda kv: kv[1])
    return top_domain if top_score >= MIN_SCORE else "unclassified"


# ---------- main ----------
def main(mode):
    pool = pd.read_parquet(POOL_FILE)
    if mode == "pilot":
        pool = pool[pool["bucket_cond"] == "pilot"]
    print(f"Users to process: {len(pool):,}")

    session = make_session()
    counts = Counter()
    for i, row in enumerate(pool.itertuples(index=False), 1):
        counts[fetch_repos(session, row.actor_id, row.login)] += 1
        if i % 100 == 0:
            print(f"  {i:,}/{len(pool):,}  {dict(counts)}")
    print("Fetch summary:", dict(counts))

    rows = []
    for row in pool.itertuples(index=False):
        path = RAW_DIR / f"{row.actor_id}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        totals, n_used = score_developer(data["repos"])
        label = "unavailable" if "error" in data else assign_label(totals)
        rows.append({
            "actor_id": row.actor_id,
            "login": row.login,
            "label": label,
            "n_repos_used": n_used,
            **{f"score_{d}": totals[d] for d in DOMAINS},
        })

    out = pd.DataFrame(rows)
    out.to_parquet(OUT_FILE, index=False)
    print(f"\nSaved {len(out):,} profiles to {OUT_FILE}")
    print(out["label"].value_counts())
    print(f"\nUnclassified: {(out['label'] == 'unclassified').mean():.1%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pilot")