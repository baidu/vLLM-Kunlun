# GitHub CI/CD Data Collection Tool

This tool is used to pull activity data from specified GitHub repositories within a given time window (e.g., PRs, Issues, Contributors, Stars, etc.), which can then be used for further analysis and visualization.

Suitable for:
- Engineering productivity metrics (PR/Issue/Contributors/Stars within a period)
- CI/CD change tracking (by time range)
- Batch data collection for a single repository

---

## Directory Structure

```text
community/
  collect_github_data.py
```

---

## Requirements

- Python 3.8+ (recommended 3.10+)
- Network access to the GitHub API
- GitHub Personal Access Token (PAT)

---

## Quick Start

### 1) Create a virtual environment (recommended)

```bash
python3 -m venv venv/dev-cicd
source venv/dev-cicd/bin/activate
```

(Optional) Upgrade pip:

```bash
pip install -U pip
```

---

### 2) Configure a GitHub Token (required)

This tool fetches data through the GitHub API. To avoid API rate limits and ensure access to private repositories, it is recommended to configure a personal token.

1. Open the token creation page:  
   https://github.com/settings/tokens
2. Create a token (read-only permissions are sufficient)
3. Export the token as an environment variable:

```bash
export GH_PAT=xxx
```

> ⚠️ Do NOT hardcode your token or commit it into the repository.

---

### 3) Set collection parameters and run

```bash
export START_DATE=2025-12-08
export END_DATE=2026-01-24
export TARGET_REPOS=baidu/vLLM-Kunlun

python community/collect_github_data.py
```

---

## Parameters

The tool is configured via environment variables:

| Parameter | Required | Example | Description |
|----------|----------|---------|-------------|
| `GH_PAT` | ✅ | `ghp_xxx` | GitHub Personal Access Token |
| `START_DATE` | ✅ | `2025-12-08` | Collection start date (format: YYYY-MM-DD) |
| `END_DATE` | ✅ | `2026-01-24` | Collection end date (format: YYYY-MM-DD) |
| `TARGET_REPOS` | ✅ | `baidu/vLLM-Kunlun` | Target repositories (supports multiple) |
