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
    # ---- order matters: ties go to the domain listed first, so specific domains
    # ---- come before broad ones like backend and frontend.
    "ml_ai": {
        "strong": ["machine learning", "deep learning", "pytorch", "tensorflow", "keras",
                   "neural network", "neural networks", "transformers", "llm", "llms",
                   "large language model", "nlp", "natural language processing",
                   "computer vision", "reinforcement learning", "scikit learn", "sklearn",
                   "huggingface", "hugging face", "diffusion", "stable diffusion", "langchain",
                   "llamaindex", "rag", "retrieval augmented generation", "fine tuning",
                   "finetuning", "lora", "gpt", "openai", "chatgpt", "claude", "gemini",
                   "ollama", "vllm", "onnx", "jax", "xgboost", "lightgbm", "yolo", "opencv",
                   "object detection", "image classification", "speech recognition",
                   "text to speech", "whisper", "embeddings", "vector database", "agents",
                   "ai agent", "generative ai", "genai", "prompt engineering", "mlops",
                   "pytorch lightning", "cnn", "rnn", "lstm", "gan", "bert"],
        "weak": ["ai", "model", "models", "training", "dataset", "classification",
                 "embedding", "prediction", "inference", "chatbot", "recommendation",
                 "recommender", "artificial intelligence", "kaggle", "neural"],
        "langs": ["jupyter notebook"],
    },
    "data_eng": {
        "strong": ["data pipeline", "data pipelines", "etl", "elt", "airflow", "spark",
                   "pyspark", "kafka", "dbt", "data warehouse", "data lake", "lakehouse",
                   "data engineering", "bigquery", "snowflake", "redshift", "databricks",
                   "flink", "beam", "hadoop", "hive", "presto", "trino", "clickhouse",
                   "delta lake", "iceberg", "dagster", "prefect", "luigi", "data ingestion",
                   "stream processing", "batch processing", "olap", "data modeling",
                   "data catalog", "great expectations", "debezium", "cdc"],
        "weak": ["pipeline", "pipelines", "warehouse", "streaming", "sql", "parquet",
                 "ingestion", "scraper", "scraping", "crawler", "orchestration"],
        "langs": [],
    },
    "data_science": {
        "strong": ["data science", "data analysis", "data analytics", "data visualization",
                   "pandas", "numpy", "matplotlib", "seaborn", "plotly", "statistics",
                   "statistical", "time series", "forecasting", "eda", "exploratory data analysis",
                   "tableau", "power bi", "jupyter", "notebook", "bayesian", "a b testing",
                   "regression", "streamlit", "dash"],
        "weak": ["analytics", "dashboard", "analysis", "visualization", "kaggle", "data",
                 "excel", "report"],
        "langs": ["r", "jupyter notebook"],
    },
    "security": {
        "strong": ["security", "cybersecurity", "pentest", "pentesting", "penetration testing",
                   "malware", "cryptography", "vulnerability", "vulnerabilities", "exploit",
                   "ctf", "reverse engineering", "osint", "red team", "blue team", "infosec",
                   "forensics", "xss", "sql injection", "burp suite", "metasploit", "nmap",
                   "fuzzing", "fuzzer", "bug bounty", "cve", "siem", "zero day", "ransomware",
                   "threat intelligence", "ghidra", "ida pro", "shellcode", "privilege escalation",
                   "oauth", "jwt", "authentication", "zero trust", "sast", "dast"],
        "weak": ["hacking", "hacker", "encryption", "scanner", "password", "bruteforce",
                 "brute force", "payload", "obfuscation", "tls", "ssl", "firewall"],
        "langs": [],
    },
    "systems": {
        "strong": ["compiler", "compilers", "operating system", "kernel", "linux kernel",
                   "systems programming", "memory allocator", "interpreter", "virtual machine",
                   "distributed systems", "distributed system", "concurrency", "consensus",
                   "raft", "paxos", "file system", "filesystem", "bootloader", "hypervisor",
                   "llvm", "assembler", "disassembler", "garbage collector", "runtime",
                   "network stack", "tcp", "udp", "ebpf", "syscall", "drivers", "device driver",
                   "lock free", "simd", "cuda", "gpu programming", "database engine",
                   "storage engine", "key value store", "b tree", "lsm tree", "no std"],
        "weak": ["performance", "low level", "parser", "lexer", "emulator", "benchmark",
                 "allocator", "threading", "multithreading", "async", "protocol"],
        "langs": ["c", "c++", "rust", "zig", "assembly", "nim", "d", "ocaml", "haskell"],
    },
    "embedded_robotics": {
        "strong": ["embedded", "rtos", "firmware", "arduino", "esp32", "esp8266",
                   "raspberry pi", "stm32", "microcontroller", "iot", "internet of things",
                   "robotics", "ros", "ros2", "fpga", "verilog", "vhdl", "pcb", "sensor",
                   "drone", "uav", "mqtt", "can bus", "plc", "freertos", "zephyr",
                   "autonomous", "slam", "kicad", "platformio", "home assistant", "smart home"],
        "weak": ["hardware", "raspberry", "circuit", "robot", "motor", "arm", "gpio"],
        "langs": ["verilog", "vhdl", "systemverilog", "arduino"],
    },
    "blockchain_web3": {
        "strong": ["blockchain", "web3", "ethereum", "solidity", "smart contract",
                   "smart contracts", "bitcoin", "defi", "nft", "solana", "dapp", "hardhat",
                   "foundry", "evm", "polygon", "cosmos", "polkadot", "substrate", "ipfs",
                   "zero knowledge", "zk", "zksnark", "layer 2", "crypto", "cryptocurrency",
                   "wallet", "token", "dao", "move language", "anchor"],
        "weak": ["trading bot", "exchange", "mining", "ledger"],
        "langs": ["solidity", "move"],
    },
    "gamedev": {
        "strong": ["game", "gamedev", "game development", "game engine", "unity", "unreal",
                   "unreal engine", "godot", "pygame", "phaser", "opengl", "vulkan", "directx",
                   "shader", "shaders", "raylib", "libgdx", "roblox", "minecraft", "mod",
                   "mods", "modding", "sdl", "bevy", "love2d", "2d game", "3d game", "voxel",
                   "ray tracing", "raytracer", "procedural generation", "game jam", "ludum dare"],
        "weak": ["player", "level", "sprite", "rpg", "platformer", "multiplayer", "3d",
                 "graphics", "rendering"],
        "langs": ["gdscript", "shaderlab", "hlsl", "glsl"],
    },
    "mobile": {
        "strong": ["android", "ios", "flutter", "react native", "swiftui", "swift ui",
                   "kotlin", "mobile app", "jetpack compose", "xcode", "expo", "ionic",
                   "kotlin multiplatform", "kmp", "cocoapods", "uikit", "android studio",
                   "gradle", "capacitor", "xamarin", "maui", "apple watch", "wearos",
                   "firebase", "app store", "play store", "mobile development"],
        "weak": ["mobile", "app", "apk", "iphone", "ipad", "tablet"],
        "langs": ["swift", "kotlin", "dart", "objective c", "objective c++"],
    },
    "devops_cloud": {
        "strong": ["kubernetes", "k8s", "terraform", "docker", "dockerfile", "devops", "ansible",
                   "helm", "ci cd", "cicd", "github actions", "gitlab ci", "jenkins", "aws",
                   "gcp", "azure", "google cloud", "infrastructure as code", "iac",
                   "prometheus", "grafana", "sre", "site reliability", "observability",
                   "opentelemetry", "cloudformation", "pulumi", "serverless", "lambda",
                   "cloudflare", "nginx", "traefik", "istio", "service mesh", "argocd",
                   "gitops", "docker compose", "vagrant", "packer", "consul", "vault",
                   "kustomize", "eks", "gke", "aks", "ec2", "s3", "cdk", "linux server",
                   "load balancer", "reverse proxy", "self hosted", "selfhosted", "homelab"],
        "weak": ["deploy", "deployment", "container", "containers", "cloud", "monitoring",
                 "pipeline", "infrastructure", "automation", "provisioning", "logging",
                 "ci", "cd", "server"],
        "langs": ["hcl", "dockerfile", "shell", "bash", "powershell", "nix"],
    },
    "backend": {
        "strong": ["backend", "back end", "rest api", "restful", "graphql", "django",
                   "fastapi", "flask", "spring boot", "spring", "express", "expressjs",
                   "nestjs", "microservice", "microservices", "grpc", "postgresql", "postgres",
                   "mysql", "mongodb", "redis", "laravel", "rails", "ruby on rails",
                   "asp.net", "aspnet", "dotnet", "node.js", "nodejs", "gin", "fiber",
                   "actix", "axum", "rocket", "sqlalchemy", "prisma", "orm", "rabbitmq",
                   "celery", "websocket", "websockets", "openapi", "swagger", "sqlite",
                   "elasticsearch", "api gateway", "message queue", "event driven",
                   "domain driven design", "ddd", "cqrs", "hibernate", "jpa", "strapi",
                   "supabase", "trpc", "deno", "bun"],
        "weak": ["api", "apis", "server", "database", "auth", "crud", "sql", "rest",
                 "middleware", "cache", "caching", "queue", "service", "endpoint", "saas"],
        "langs": ["java", "go", "php", "ruby", "c#", "elixir", "scala", "kotlin", "erlang"],
    },
    "frontend": {
        "strong": ["react", "reactjs", "vue", "vuejs", "angular", "svelte", "sveltekit",
                   "nextjs", "next.js", "nuxt", "nuxtjs", "tailwind", "tailwindcss",
                   "frontend", "front end", "css", "sass", "scss", "webpack", "vite",
                   "web components", "redux", "zustand", "astro", "remix", "gatsby",
                   "storybook", "material ui", "mui", "chakra ui", "shadcn", "bootstrap",
                   "jquery", "htmx", "pwa", "progressive web app", "three.js", "threejs",
                   "d3", "d3.js", "framer motion", "responsive design", "web design",
                   "design system", "ui library", "component library", "solidjs", "qwik",
                   "alpine.js", "browser extension", "chrome extension", "webassembly",
                   "wasm", "typescript"],
        "weak": ["ui", "ux", "website", "web", "portfolio", "component", "components",
                 "landing page", "webpage", "web app", "webapp", "dashboard", "template",
                 "javascript", "html", "animation"],
        "langs": ["typescript", "vue", "svelte", "css", "scss", "less", "astro"],
    },
    "devtools_oss": {
        "strong": ["cli", "command line", "command line tool", "developer tools", "devtools",
                   "developer experience", "dx", "linter", "formatter", "static analysis",
                   "vscode extension", "vim plugin", "neovim", "neovim plugin", "emacs",
                   "language server", "lsp", "build tool", "package manager", "monorepo",
                   "sdk", "library", "framework", "open source", "code generation", "codegen",
                   "boilerplate", "scaffolding", "ast", "transpiler", "bundler", "git",
                   "terminal", "tui", "shell script", "dotfiles", "productivity tool",
                   "api client", "wrapper", "plugin"],
        "weak": ["tool", "tools", "utility", "utilities", "helper", "toolkit", "starter"],
        "langs": ["vim script", "vim", "emacs lisp", "lua"],
    },
    "testing_qa": {
        "strong": ["testing", "test automation", "unit testing", "integration testing",
                   "e2e", "end to end testing", "selenium", "cypress", "playwright",
                   "pytest", "jest", "junit", "mocha", "qa", "quality assurance",
                   "load testing", "k6", "locust", "mutation testing", "tdd", "bdd",
                   "cucumber", "appium", "postman", "test framework", "mocking"],
        "weak": ["test", "tests", "coverage", "benchmark", "assertion"],
        "langs": [],
    },
    "automation_bots": {
        "strong": ["discord bot", "telegram bot", "slack bot", "twitch bot", "whatsapp bot",
                   "bot", "automation", "web scraping", "scraper", "crawler", "selenium bot",
                   "rpa", "workflow automation", "n8n", "zapier", "userscript", "tampermonkey",
                   "discord.py", "discord.js", "telegraf", "puppeteer", "webhook", "cron",
                   "macro", "autoclicker", "script"],
        "weak": ["discord", "telegram", "slack", "twitter", "reddit", "youtube", "scripts"],
        "langs": ["autohotkey"],
    },
    "scientific_computing": {
        "strong": ["bioinformatics", "genomics", "computational biology", "scientific computing",
                   "numerical", "simulation", "physics", "quantum", "quantum computing",
                   "qiskit", "computational", "finite element", "cfd", "molecular dynamics",
                   "astronomy", "astrophysics", "climate", "geospatial", "gis", "remote sensing",
                   "hpc", "mpi", "openmp", "linear algebra", "optimization", "solver",
                   "differential equations", "monte carlo", "matlab", "julia", "fortran",
                   "cheminformatics", "neuroscience", "medical imaging", "healthcare", "fhir"],
        "weak": ["research", "paper", "thesis", "scientific", "math", "mathematics",
                 "algorithm", "algorithms"],
        "langs": ["julia", "fortran", "matlab", "r", "mathematica"],
    },
    "desktop_apps": {
        "strong": ["desktop app", "desktop application", "electron", "tauri", "qt", "pyqt",
                   "tkinter", "wpf", "winforms", "gtk", "wxwidgets", "javafx", "swing",
                   "cross platform", "gui", "avalonia", "imgui", "macos app", "menubar",
                   "windows app", "linux app", "system tray", "overlay"],
        "weak": ["desktop", "window", "widget", "installer"],
        "langs": ["qml"],
    },
    # Deliberate sink: tags beginners and coursework so they stop polluting real domains.
    # Decide later whether this label belongs in the corpus at all.
    "learning": {
        "strong": ["homework", "assignment", "coursework", "tutorial", "bootcamp", "cs50",
                   "leetcode", "hackerrank", "codewars", "exercism", "learning", "practice",
                   "exercises", "course project", "university project", "college project",
                   "student", "100 days of code", "freecodecamp", "the odin project",
                   "udemy", "coursera", "my first", "hello world", "todo app", "todo list",
                   "calculator", "tic tac toe", "snake game", "to do app", "cheat sheet",
                   "notes", "lab"],
        "weak": ["learn", "basic", "basics", "beginner", "demo", "example", "examples",
                 "sample", "clone", "simple", "test project"],
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