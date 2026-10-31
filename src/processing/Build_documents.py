"""
Build one embeddable text document per developer in the corpus.

Run from anywhere:
    python src/processing/build_documents.py
"""
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CORPUS_FILE = ROOT / "data" / "processed" / "corpus.parquet"
RAW_DIR = ROOT / "data" / "raw" / "repos"
OUT_FILE = ROOT / "data" / "processed" / "documents.parquet"

# ---------------------------------------------------------------- settings
MAX_REPOS = 8            # never include more than this many repos
WORD_BUDGET = 300        # bge-small reads ~512 tokens; ~300 words leaves a safety margin
MAX_DESC_WORDS = 30      # long descriptions are trimmed so one repo can't eat the budget
JUNK_NAMES = {"dotfiles", "test", "tests", "hello world", "helloworld", "demo",
              "playground", "sandbox", "learning", "practice", "untitled"}

URL_RE = re.compile(r"https?://\S+")
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF]+"
)
SPACE_RE = re.compile(r"\s+")


# ----------------------------------------------------------------- cleaning
def clean(text):
    """Remove URLs and emojis, collapse whitespace. Returns '' for None."""
    text = text or ""
    text = URL_RE.sub("", text)
    text = EMOJI_RE.sub("", text)
    return SPACE_RE.sub(" ", text).strip()


def trim_words(text, limit):
    words = text.split()
    return text if len(words) <= limit else " ".join(words[:limit]) + "..."


def pretty_name(name):
    """'react-dashboard_v2' -> 'react dashboard v2' (reads more like language)."""
    return SPACE_RE.sub(" ", re.sub(r"[-_]+", " ", name or "")).strip()


# --------------------------------------------------------- choosing repos
def usable(repo):
    if repo.get("fork") or repo.get("archived"):
        return False
    if pretty_name(repo.get("name")).lower() in JUNK_NAMES:
        return False
    return bool(repo.get("description") or repo.get("topics") or repo.get("language"))


def days_since(iso_time):
    if not iso_time:
        return 10_000
    pushed = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - pushed).days


def repo_rank(repo):
    """
    Higher = more worth showing.
    log1p(stars): the difference between 0 and 10 stars matters more than 1000 vs 1010.
    recency: a repo touched recently says more about what someone does today.
    richness: repos with a description and topics carry more searchable meaning.
    """
    stars = math.log1p(repo.get("stargazers_count", 0))
    age = days_since(repo.get("pushed_at"))
    recency = 2.0 if age <= 180 else 1.0 if age <= 365 else 0.0
    richness = (1.0 if repo.get("description") else 0.0) + (0.5 if repo.get("topics") else 0.0)
    return stars + recency + richness


# ------------------------------------------------------- writing the text
def repo_line(repo):
    name = pretty_name(repo.get("name"))
    lang = repo.get("language")
    stars = repo.get("stargazers_count", 0)

    meta = [lang] if lang else []
    if stars:
        meta.append(f"{stars} stars")
    head = f"- {name}" + (f" ({', '.join(meta)})" if meta else "")

    desc = trim_words(clean(repo.get("description")), MAX_DESC_WORDS)
    topics = [pretty_name(t) for t in (repo.get("topics") or [])[:6]]

    line = head
    if desc:
        line += f": {desc}"
    if topics:
        line += f" Topics: {', '.join(topics)}."
    return line


def build_document(repos):
    """Returns (text, n_repos_included). Text is '' if nothing usable."""
    good = [r for r in repos if usable(r)]
    if not good:
        return "", 0

    # the developer's overall language profile, counted over ALL usable repos
    langs = Counter(r["language"] for r in good if r.get("language"))
    top_langs = [l for l, _ in langs.most_common(4)]

    lines = []
    header = "Main languages: " + ", ".join(top_langs) + "." if top_langs else ""
    words_used = len(header.split())

    ranked = sorted(good, key=repo_rank, reverse=True)[:MAX_REPOS]
    included = 0
    for repo in ranked:
        line = repo_line(repo)
        n = len(line.split())
        if included > 0 and words_used + n > WORD_BUDGET:
            break              # stop BEFORE overflowing, so no repo is cut in half
        lines.append(line)
        words_used += n
        included += 1

    text = "\n".join(([header] if header else []) + ["Repositories:"] + lines)
    return text, included


# ------------------------------------------------------------------- main
def main():
    corpus = pd.read_parquet(CORPUS_FILE)
    rows, missing = [], 0

    for row in corpus.itertuples(index=False):
        path = RAW_DIR / f"{row.actor_id}.json"
        if not path.exists():
            missing += 1
            continue
        data = json.loads(path.read_text())
        text, n_included = build_document(data["repos"])
        if not text:
            continue
        rows.append({
            "actor_id": int(row.actor_id),
            "login": row.login,
            "label": row.label,
            "n_repos_used": int(row.n_repos_used),
            "n_repos_in_doc": n_included,
            "n_words": len(text.split()),
            "github_url": f"https://github.com/{row.login}",
            "text": text,
        })

    docs = pd.DataFrame(rows)
    docs.to_parquet(OUT_FILE, index=False)

    print(f"Corpus: {len(corpus):,} | documents built: {len(docs):,} | missing raw files: {missing}")
    print(docs["n_words"].describe())
    print(f"Over 350 words: {(docs['n_words'] > 350).sum()}")
    print("\n--- 3 sample documents ---")
    for r in docs.sample(3, random_state=1).itertuples():
        print(f"\n[{r.login} | {r.label} | {r.n_words} words]\n{r.text}")


if __name__ == "__main__":
    main()