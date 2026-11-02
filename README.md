# FindCrackedDevs

A retrieval-augmented generation (RAG) search engine over a sample of public GitHub developers. A recruiter describes the developer they need in plain English, and the app returns the closest matches with a short, evidence-cited report and a link to each GitHub profile.

**Live demo:** https://findcrackeddevs.bilaalbakare.workers.dev
**Status:** early build and learning project. Read [Limitations](#limitations) before relying on any result.

> Example request: *"a developer experienced in machine learning and large-scale applications"*

## How it works

```
OFFLINE (Python, run on a laptop)
GH Archive (BigQuery) -> activity filter -> hash-bucket pool -> GitHub API repo lists
   -> keyword domain scoring -> stratified sample (~1,800) -> one text document per developer
   -> embeddings (BGE-small, 384 dims) -> exported vectors + documents

ONLINE (Cloudflare Worker, JavaScript)
query -> embed (Workers AI, same model) -> cosine similarity over all vectors -> top 5
      -> LLM (Groq) writes a JSON report that must cite repositories -> frontend
```

### Design decisions

| Decision | Why |
|---|---|
| **GH Archive via BigQuery as the sampling frame** | Gives a free list of recently active accounts without spending GitHub API calls. |
| **Hash-bucket sampling on `actor_id`** | A person's bucket never changes, so pilot, pool, reserve and control sets stay reproducible and non-overlapping when filters change. |
| **Stratified quotas per domain (150 each)** | Recruiters ask about rare skills. Equal quotas guarantee candidates exist in small domains, at the cost of not mirroring the real population. |
| **Fetch and score as separate stages** | Raw API responses are saved, so the keyword dictionary can be changed and rescored without new API calls. |
| **Label kept out of the embedded text** | Labels come from a keyword dictionary. Embedding them would make search echo the dictionary instead of finding meaning. |
| **Brute-force cosine search instead of a vector database** | With ~1,800 vectors it is exact and takes about a millisecond. A vector database pays off at much larger scale. |
| **LLM never writes links** | Names and URLs come from stored metadata, so the model cannot invent them. |
| **Strict JSON output with graceful fallback** | If the model returns bad JSON, the retrieved developers are still shown. |

## Numbers from this build

| Stage | Count |
|---|---|
| Active-account rows analysed (8-day window, bots removed) | 1,031,403 |
| After activity filter (3+ active days, 5 to 400 events) | 90,810 |
| Pool profiled through the GitHub API (hash buckets 0 to 99) | 9,053 |
| Labelled `unclassified` in the pool | about 46% |
| Final corpus (up to 150 per domain, 17 domains) | about 1,800 |

## Limitations

This is a demo built on free tiers and one laptop. Please read this section.

- **Small, biased corpus.** About 1,800 developers drawn from accounts that were active in a single 8-day window. The best match for your query may not be indexed.
- **Keyword-based labels.** Domains come from a hand-written dictionary that has only been lightly audited. Some developers are mislabelled, and about 46% of the pool matched no domain.
- **Relevant is not best.** Search measures similarity to your words. There is no quality or seniority ranking, so a query like "best developer" returns people whose repositories sound relevant.
- **Dense retrieval only.** No keyword (BM25) hybrid search and no reranker yet, so exact technology names can be blurred. The relevance label and result order can disagree.
- **Thin profiles.** Documents are built from repository names, descriptions, topics and languages. READMEs, bios and commit history are not used.
- **No evaluation yet.** There is no measured recall or precision. Do not treat quality claims as verified.
- **Public data is a partial picture.** Private work and offline skills are invisible. Stars reflect visibility, not skill.
- **LLM summaries can be wrong.** The model is told to use only retrieved text and cite repositories, but it can still misread. Always check the linked profile.
- **Free-tier infrastructure.** Daily request limits, rate limits and model changes can make the demo slow or unavailable.

### Hardware it was built on

Ubuntu laptop, 7.6 GB RAM (often under 2 GB free), 4 GB swap, CPU only, Python 3.12. The first bulk-embedding attempt was killed twice by the out-of-memory killer, which is why the offline pipeline works in small, resumable chunks and why serving moved to Cloudflare.

## Privacy and removal

The app indexes only public information from public GitHub profiles: login, profile link, and public repository names, descriptions, topics and languages. It does not collect emails or private data.

**To be removed:** open an issue titled "Removal request" with your GitHub username. Requests are honoured by removing your entry from the index and redeploying. This tool should never be the sole basis for a hiring decision.

## Run it yourself

### Offline pipeline (Python 3.12)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # add GITHUB_TOKEN, GCP_PROJECT, GROQ_API_KEY, LLM_MODEL
```

Run the stages from the project root, in order:

```bash
python src/ingestion/population.py          # BigQuery: active accounts (check the dry-run cost first)
python src/processing/pool.py               # activity filter and hash buckets
python src/processing/profile_users.py pilot   # then: all  (fetch repos, score domains)
python src/processing/sampling.py           # stratified corpus
python src/processing/build_documents.py    # one text document per developer
python src/processing/build_vector_database.py build
python scripts/export_cloudflare.py         # vectors.bin and one JSON file per developer
```

Notes:
- BigQuery bills by bytes scanned. Always dry-run first and set `maximum_bytes_billed`.
- `profile_users.py` skips users already fetched, so you can stop and resume.
- GitHub allows 5,000 authenticated requests per hour.

### Local Python API (optional)

```bash
uvicorn src.api.main:app --port 8000
```

### Cloudflare Worker (production)

```bash
npm install
cp frontend/index.html public/index.html
npx wrangler login
npx wrangler secret put GROQ_API_KEY
npx wrangler deploy
```

For local Worker testing, put `GROQ_API_KEY` in `.dev.vars` and run `npx wrangler dev`.

## Project structure

```
.
├── src/
│   ├── ingestion/      # BigQuery population query
│   ├── processing/     # pool, repo fetching, scoring, sampling, documents, vector build
│   ├── retrieval/
│   ├── generation/     # prompt and LLM report (CLI)
│   └── api/            # FastAPI app for local use
├── worker/index.js     # Cloudflare Worker: the production backend
├── public/             # static site and exported index (served by the Worker)
├── frontend/           # frontend source
├── scripts/            # export script
├── wrangler.jsonc      # Cloudflare configuration
├── data/               # gitignored: raw, processed, vectors, Chroma
├── eval/  notebooks/  tests/
└── .env.example
```

Secrets (`.env`, `.dev.vars`, `env.yaml`) are gitignored. Never commit tokens.

## Roadmap

- [ ] Evaluation set with recall@k and MRR, so changes can be measured
- [ ] Hybrid search (BM25 plus vectors) with reciprocal rank fusion
- [ ] Cross-encoder reranking
- [ ] Quality signal from merged pull requests to other projects, shown with evidence
- [ ] README and bio ingestion
- [ ] Per-visitor rate limiting
- [ ] Sort results by fit, then similarity
- [ ] Tests and CI
- [ ] Query understanding with structured filters (language, location, seniority)

## Contributing

Pull requests are welcome from anyone. For anything larger than a small fix, open an issue first so effort isn't wasted.

1. Fork the repository and create a branch.
2. Make one focused change.
3. Run what exists: the pipeline stage you touched, and a few queries before and after.
4. Open a pull request explaining what changed and why. Include before-and-after results where retrieval is affected.

**Help especially wanted:** an evaluation set, hybrid search and reranking, a better domain dictionary, GPU or high-memory compute for re-embedding, and rate limiting.

## Maintainer

**Bilaal Bakare**, AI/ML engineer · RAG, embeddings and data pipelines. Open to collaboration.

- GitHub: [@BilaalBakare](https://github.com/BilaalBakare)
- LinkedIn: [bilaalbakare](https://www.linkedin.com/in/bilaalbakare/)

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgements

[GH Archive](https://www.gharchive.org/) for the public activity data, [BAAI](https://huggingface.co/BAAI/bge-small-en-v1.5) for the BGE embedding model, Cloudflare Workers and Workers AI for hosting and inference, and Groq for LLM inference.
