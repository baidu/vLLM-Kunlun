#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub open-source repository statistics script.

This tool collects activity metrics (PR/Issue/Contributors/Stars/Forks) for one or more
GitHub repositories within a specified time window, and outputs a summary report.
It also generates a visualization chart and saves it as 'github_stats.png'.
"""

import os
import requests
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import matplotlib.pyplot as plt


"""
# Initialization & configuration
"""

# Load configuration from environment variables
GH_PAT = os.getenv("GH_PAT")
TARGET_REPOS = os.getenv("TARGET_REPOS")
HEADERS = {"Authorization": f"token {GH_PAT}"}

# Configure request retry strategy (to handle transient GitHub API instability)
session = requests.Session()
retry = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)


def validate_config():
    """
    Validate required configuration and return TARGET_REPOS_LIST.

    Returns:
        List[str]: Repositories list in format ["owner/repo1", "owner/repo2", ...]
    """
    if not GH_PAT:
        raise ValueError("ERROR: GH_PAT is not set. Please add it in Secrets/environment variables.")
    if not TARGET_REPOS or TARGET_REPOS.strip() == "":
        raise ValueError(
            "ERROR: TARGET_REPOS is not set. Please provide a comma-separated repo list "
            "(e.g. 'owner/repo1,owner/repo2')."
        )

    target_repos_list = TARGET_REPOS.split(",")
    if not all(repo.strip() for repo in target_repos_list):
        raise ValueError("ERROR: Invalid TARGET_REPOS format. Repo list cannot contain empty items.")
    return [repo.strip() for repo in target_repos_list]


def get_stat_period():
    """
    Get the statistics time range.

    Returns:
        tuple[str, str]: (since, until)
            - since: 00:00:00Z of START_DATE or (now - 14 days)
            - until: 23:59:59Z of END_DATE or today
    """
    # Prefer manually specified window via env vars
    # Format: START_DATE=2025-10-31, END_DATE=2026-01-31
    start_str = os.getenv("START_DATE")
    end_str = os.getenv("END_DATE")

    if start_str and end_str:
        try:
            datetime.strptime(start_str, "%Y-%m-%d")
            datetime.strptime(end_str, "%Y-%m-%d")
            return f"{start_str}T00:00:00Z", f"{end_str}T23:59:59Z"
        except ValueError:
            raise ValueError("ERROR: Invalid START_DATE/END_DATE format. Expected 'YYYY-MM-DD'.")

    # Auto-calculate biweekly window: last 14 days to today
    end_date = datetime.now()
    start_date = end_date - timedelta(days=14)
    return start_date.strftime("%Y-%m-%dT00:00:00Z"), end_date.strftime("%Y-%m-%dT23:59:59Z")


def get_github_contributor_stats(
    repo,
    since,
    until,
    token=None,
    per_page=1000,
):
    """
    Collect contributor statistics for a repository.

    Returns:
        (int, int):
            - total_contributors_count: total historical contributors count
            - period_contributors_count: contributors within the given time window
    """
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # -------------------------
    # 1) Total historical contributors
    # -------------------------
    total_contributors = set()
    page = 1

    while True:
        url = (
            f"https://api.github.com/repos/{repo}/contributors"
            f"?per_page={per_page}&page={page}"
        )
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()

        data = resp.json()
        if not data:
            break

        for c in data:
            if c.get("login"):
                total_contributors.add(c["login"])

        page += 1

    # -------------------------
    # 2) Contributors in time window (from commits)
    # -------------------------
    period_contributors = set()
    page = 1

    while True:
        url = (
            f"https://api.github.com/repos/{repo}/commits"
            f"?since={since}&until={until}"
            f"&per_page={per_page}&page={page}"
        )
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()

        commits = resp.json()
        if not commits:
            break

        for c in commits:
            if c.get("author") and c["author"]:
                period_contributors.add(c["author"]["login"])
            else:
                # Fallback to email (anonymous commits)
                email = c["commit"]["author"].get("email")
                if email:
                    period_contributors.add(email)

        page += 1

    return len(total_contributors), len(period_contributors)


def fetch_prs(repo, start_str, end_str, is_main_repo=False):
    """
    Fetch PR statistics within the time window.
    Also supports counting PRs merged into a main/community repo from target repos.

    Returns:
        (int, int, int, int, int):
            - total PRs
            - merged PRs (within time window)
            - closed PRs (closed but not merged)
            - open PRs
            - prs_from_target (only for main repo scenario)
    """
    try:
        url = f"https://api.github.com/repos/{repo}/pulls"
        params = {"state": "all", "since": start_str, "until": end_str, "per_page": 1000}
        response = session.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        prs = response.json()

        open_prs = [pr for pr in prs if pr["state"] == "open"]
        merged_prs = [pr for pr in prs if pr.get("merged_at") and start_str <= pr["merged_at"] <= end_str]
        closed_prs = [pr for pr in prs if pr["state"] == "closed" and not pr.get("merged_at")]

        if is_main_repo:
            target_repos = validate_config()
            prs_from_target = 0
            for pr in merged_prs:
                # Determine whether PR head repo is from a target repo list
                if pr.get("head") and pr["head"].get("repo") and pr["head"]["repo"].get("full_name") in target_repos:
                    prs_from_target += 1
            return len(prs), len(merged_prs), len(closed_prs), len(open_prs), prs_from_target

        return len(prs), len(merged_prs), len(closed_prs), len(open_prs), 0

    except requests.exceptions.HTTPError as e:
        raise Exception(f"Failed to fetch PR data for {repo}: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        raise Exception(f"Exception while fetching PR data for {repo}: {str(e)}")


def fetch_issues(repo, start_str, end_str):
    """
    Fetch issue statistics within the time window (Open/Closed).
    Note: GitHub issues API also returns PRs; we filter PR objects out.

    Returns:
        (int, int): open_issues_count, closed_issues_count
    """
    try:
        url = f"https://api.github.com/repos/{repo}/issues"
        params = {"state": "all", "since": start_str, "until": end_str, "per_page": 1000}
        response = session.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        issues = response.json()

        # Filter out PR objects
        issues = [issue for issue in issues if "pull_request" not in issue]

        open_issues = [issue for issue in issues if issue["state"] == "open"]
        close_issues = [issue for issue in issues if issue["state"] == "closed"]

        return len(open_issues), len(close_issues)

    except requests.exceptions.HTTPError as e:
        raise Exception(f"Failed to fetch Issue data for {repo}: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        raise Exception(f"Exception while fetching Issue data for {repo}: {str(e)}")


def fetch_stars_forks(repo, start_str):
    """
    Fetch stars/forks totals and estimate new stars/forks since start date.
    Note: This uses /events which is limited and only covers a short history.

    Returns:
        (int, int, int, int):
            - new_stars
            - current_stars_total
            - new_forks
            - current_forks_total
    """
    try:
        # Current totals
        url = f"https://api.github.com/repos/{repo}"
        response = session.get(url, headers=HEADERS)
        response.raise_for_status()
        repo_info = response.json()
        current_stars = repo_info["stargazers_count"]
        current_forks = repo_info["forks_count"]

        # Count events since start date
        start_date = start_str.split("T")[0]
        url = f"https://api.github.com/repos/{repo}/events?per_page=1000"
        response = session.get(url, headers=HEADERS)
        response.raise_for_status()
        events = response.json()

        new_stars = 0
        new_forks = 0
        for event in events:
            event_date = event["created_at"].split("T")[0]
            if event_date >= start_date:
                if event["type"] == "WatchEvent":
                    new_stars += 1
                elif event["type"] == "ForkEvent":
                    new_forks += 1

        return new_stars, current_stars, new_forks, current_forks

    except requests.exceptions.HTTPError as e:
        raise Exception(f"Failed to fetch Stars/Forks data for {repo}: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        raise Exception(f"Exception while fetching Stars/Forks data for {repo}: {str(e)}")


def plot_github_stats(stats: dict, start_time: str, end_time: str):
    """
    Visualize GitHub community metrics (PR/Issue/Contributors/Stars/Forks).

    Args:
        stats (dict): aggregated statistics
        start_time (str): start time string
        end_time (str): end time string
    """
    labels = ["PR Total", "Issue Total", "Contributors", "Stars", "Forks"]
    values = [
        stats["prs_total"],
        stats["issues_total"],
        stats["total_contributors"],
        stats["stars_total"],
        stats["forks_total"],
    ]

    colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F']

    def extract_date(date_str):
        # Support multiple time formats
        for fmt in ('%Y-%m-%d', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d'):
            try:
                return datetime.strptime(date_str.strip(), fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        return date_str[:10] if len(date_str) >= 10 else date_str

    start_date = extract_date(start_time)
    end_date = extract_date(end_time)

    plt.figure(figsize=(9, 6))
    bars = plt.bar(labels, values, color=colors)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(values) * 0.01,
            f'{int(height):,}',
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=10
        )

    plt.title(f'GitHub Community Statistics\n({start_date} ~ {end_date})', fontsize=14, fontweight='bold')
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(__file__), "github_stats.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """
    Main entry:
    - Validate configuration
    - Compute time window
    - Fetch statistics for target repos
    - Print summary
    - Generate chart
    """
    try:
        # 1) Validate config
        target_repos_list = validate_config()

        # 2) Get time window
        start_str, end_str = get_stat_period()
        print(f"Statistics period: {start_str} ~ {end_str}")

        # 3) Aggregation container
        total_data = {
            "total_contributors": 0,
            "period_contributors": 0,
            "prs_total": 0,
            "prs_merged": 0,
            "prs_closed": 0,
            "prs_open": 0,
            "stars_new": 0,
            "stars_total": 0,
            "forks_new": 0,
            "forks_total": 0,
            "open_issues": 0,
            "close_issues": 0,
            "issues_total": 0
        }

        # 4) Fetch data per repo
        for repo in target_repos_list:
            total_contributors, period_contributors = get_github_contributor_stats(repo, start_str, end_str, GH_PAT)
            prs_t, prs_m, prs_c, prs_o, _ = fetch_prs(repo, start_str, end_str)
            open_issues, close_issues = fetch_issues(repo, start_str, end_str)
            stars_n, stars_t, forks_n, forks_t = fetch_stars_forks(repo, start_str)

            # Aggregate
            total_data["total_contributors"] += total_contributors
            total_data["period_contributors"] += period_contributors
            total_data["prs_total"] += prs_t
            total_data["prs_merged"] += prs_m
            total_data["prs_closed"] += prs_c
            total_data["prs_open"] += prs_o
            total_data["open_issues"] += open_issues
            total_data["close_issues"] += close_issues
            total_data["issues_total"] += open_issues + close_issues
            total_data["stars_new"] += stars_n
            total_data["stars_total"] = stars_t  # simplified: last repo's total
            total_data["forks_new"] += forks_n
            total_data["forks_total"] = forks_t

        prs_closed_total = total_data["prs_merged"] + total_data["prs_closed"]

        print("\n================ GitHub Community Summary ================\n")

        print("[Pull Request]")
        print(f"  - Total PRs: {total_data['prs_total']}")
        print(f"  - Closed PRs (Merged + Closed): {prs_closed_total}")
        print(f"  - Current Open PRs: {total_data['prs_open']}\n")

        print("[Issues]")
        print(f"  - Total Issues: {total_data['issues_total']}")
        print(f"  - Closed Issues: {total_data['close_issues']}")
        print(f"  - Current Open Issues: {total_data['open_issues']}\n")

        print("[Community Metrics]")
        print(f"  - Total Stars: {total_data['stars_total']}")
        print(f"  - Total Forks: {total_data['forks_total']}")
        print(f"  - Total Contributors: {total_data['total_contributors']}")

        plot_github_stats(total_data, start_str, end_str)
        print("\nChart generated successfully: github_stats.png\n")

    except Exception as e:
        print(f"Data collection failed: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
