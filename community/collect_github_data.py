#!/usr/bin/env python3
"""github 开源仓库统计脚本"""
# -*- coding: utf-8 -*-

import os
import requests
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import matplotlib.pyplot as plt


"""
# 初始化配置
"""

# 初始化配置
GH_PAT = os.getenv("GH_PAT")
TARGET_REPOS = os.getenv("TARGET_REPOS")
HEADERS = {"Authorization": f"token {GH_PAT}"}

# 配置请求重试机制（解决API调用不稳定问题）
session = requests.Session()
retry = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)

# 校验核心配置（缺失则直接报错，便于定位问题）
def validate_config():
    """
    返回：TARGET_REPOS_LIST
    """
    if not GH_PAT:
        raise ValueError("错误：GH_PAT 未配置，请在 Secrets 中添加")
    if not TARGET_REPOS or TARGET_REPOS.strip() == "":
        raise ValueError("错误：TARGET_REPOS 未配置，请在 Secrets 中添加需统计的仓库列表（逗号分隔）")
    TARGET_REPOS_LIST = TARGET_REPOS.split(",")
    if not all(repo.strip() for repo in TARGET_REPOS_LIST):
        raise ValueError("错误：TARGET_REPOS 格式错误，仓库列表不能为空（如 'owner/repo1,owner/repo2'）")
    return [repo.strip() for repo in TARGET_REPOS_LIST]


# 计算统计周期（双周：前14天到当天，支持手动指定周期用于测试）
def get_stat_period():
    """
    返回：双周统计周期，格式为 (since, until)
    - since: 当地时间前14天的00:00:00Z
    - until: 当地时间的23:59:59Z
    """
    # 优先从环境变量获取手动指定的周期（格式：START_DATE=2025-10-31,END_DATE=2026-01-31）
    start_str = os.getenv("START_DATE")
    end_str = os.getenv("END_DATE")
    if start_str and end_str:
        try:
            datetime.strptime(start_str, "%Y-%m-%d")
            datetime.strptime(end_str, "%Y-%m-%d")
            return f"{start_str}T00:00:00Z", f"{end_str}T23:59:59Z"
        except ValueError:
            raise ValueError("错误：手动指定的 START_DATE/END_DATE 格式错误，应为 'YYYY-MM-DD'")
    # 自动计算双周周期
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
    返回：
    - total_contributors_count: 历史总贡献者数
    - period_contributors_count: 指定时间段贡献者数
    """

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # -------------------------
    # 1. 历史总 contributors
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
    # 2. 时间区间 contributors
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
                # fallback 到 email（匿名提交）
                email = c["commit"]["author"].get("email")
                if email:
                    period_contributors.add(email)

        page += 1

    return len(total_contributors), len(period_contributors)

# def fetch_commits(repo, start_str, end_str):
#     """拉取统计周期内的提交数据，添加异常捕获（用于推导Contributors）"""
#     try:
#         url = f"https://api.github.com/repos/{repo}/commits"
#         params = {"since": start_str, "until": end_str, "per_page": 1000}  # 增加分页参数，默认100条
#         response = session.get(url, headers=HEADERS, params=params)
#         response.raise_for_status()  # 触发HTTP错误（如403、404）
#         commits = response.json()
#         # 处理提交者可能为None的情况
#         authors = set()
#         for commit in commits:
#             if commit.get("author") and commit["author"].get("login"):
#                 authors.add(commit["author"]["login"])
#         return len(commits), authors
#     except requests.exceptions.HTTPError as e:
#         raise Exception(f"拉取 {repo} 提交数据失败：{e.response.status_code} - {e.response.text}")
#     except Exception as e:
#         raise Exception(f"拉取 {repo} 提交数据异常：{str(e)}")

def fetch_prs(repo, start_str, end_str, is_main_repo=False):
    """拉取统计周期内的PR数据，新增合入社区主干统计，添加异常捕获"""
    try:
        url = f"https://api.github.com/repos/{repo}/pulls"
        params = {"state": "all", "since": start_str, "until": end_str, "per_page": 1000}
        response = session.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        prs = response.json()
        open_prs = [pr for pr in prs if pr["state"] == "open"]
        merged_prs = [pr for pr in prs if pr.get("merged_at") and start_str <= pr["merged_at"] <= end_str]
        closed_prs = [pr for pr in prs if pr["state"] == "closed" and not pr.get("merged_at")]

        # 若为社区主干仓库，统计目标仓库合入的PR数
        if is_main_repo:
            target_repos = validate_config()
            prs_from_target = 0
            for pr in merged_prs:
                # 检查PR提交者是否来自目标仓库（或PR源仓库为目标仓库）
                if pr.get("head") and pr["head"].get("repo") and pr["head"]["repo"].get("full_name") in target_repos:
                    prs_from_target += 1
            return len(prs), len(merged_prs), len(closed_prs), len(open_prs), prs_from_target
        return len(prs), len(merged_prs), len(closed_prs), len(open_prs), 0
    except requests.exceptions.HTTPError as e:
        raise Exception(f"拉取 {repo} PR数据失败：{e.response.status_code} - {e.response.text}")
    except Exception as e:
        raise Exception(f"拉取 {repo} PR数据异常：{str(e)}")


def fetch_issues(repo, start_str, end_str):
    """拉取统计周期内的Issue数据，包含Open和Close数量"""
    try:
        url = f"https://api.github.com/repos/{repo}/issues"
        params = {"state": "all", "since": start_str, "until": end_str, "per_page": 1000}
        response = session.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        issues = response.json()
        # 排除PR（Issue API会返回PR，需过滤）
        issues = [issue for issue in issues if "pull_request" not in issue]
        open_issues = [issue for issue in issues if issue["state"] == "open"]
        close_issues = [issue for issue in issues if issue["state"] == "closed"]
        # close_issues = [issue for issue in issues if start_str <= issue["closed_at"] <= end_str]
        return len(open_issues), len(close_issues)
    except requests.exceptions.HTTPError as e:
        raise Exception(f"拉取 {repo} Issue数据失败：{e.response.status_code} - {e.response.text}")
    except Exception as e:
        raise Exception(f"拉取 {repo} Issue数据异常：{str(e)}")


def fetch_stars_forks(repo, start_str):
    """拉取Stars和Forks增长数据，优化计算逻辑，添加异常捕获"""
    try:
        # 获取当前累计Stars和Forks
        url = f"https://api.github.com/repos/{repo}"
        response = session.get(url, headers=HEADERS)
        response.raise_for_status()
        repo_info = response.json()
        current_stars = repo_info["stargazers_count"]
        current_forks = repo_info["forks_count"]

        # 优化：通过时间范围筛选事件，更准确计算新增数量
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
        raise Exception(f"拉取 {repo} Stars/Forks数据失败：{e.response.status_code} - {e.response.text}")
    except Exception as e:
        raise Exception(f"拉取 {repo} Stars/Forks数据异常：{str(e)}")

def plot_github_stats(stats: dict, start_time: str, end_time: str):
    """
    可视化 GitHub 社区统计数据（PR/Issue/Contributors/Star/Fork）
    :param stats: 统计数据字典
    :param start_time: 统计开始时间（字符串，支持常见日期/时间格式）
    :param end_time: 统计结束时间（字符串）
    """

    labels = [
        "PR Total", "Issue Total", "Contributors", "Stars", "Forks"
    ]

    values = [
        stats["prs_total"],
        stats["issues_total"],
        stats["total_contributors"],
        stats["stars_total"],
        stats["forks_total"],
    ]

    colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F']

    # 标准化时间为 YYYY-MM-DD
    def extract_date(date_str):
        # 支持 ISO 格式（如 '2025-01-15T10:30:00Z'）或简单日期（如 '2025-01-15'）
        for fmt in ('%Y-%m-%d', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d'):
            try:
                return datetime.strptime(date_str.strip(), fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        # 如果都不匹配，直接截取前10个字符（保守策略）
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
    # plt.savefig("github_stats.png", dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

def main():
    """
    主函数，执行统计流程：
    """
    try:
        # 1. 校验配置
        TARGET_REPOS_LIST = validate_config()
        # 2. 获取统计周期
        start_str, end_str = get_stat_period()
        print(f"统计周期：{start_str} ~ {end_str}")

        # 3. 初始化汇总数据
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
            "open_issues": 0,  # Open Issue数
            "close_issues": 0,  # Close Issue数
            "issues_total": 0  # Issue总数
        }

        # 4. 遍历目标仓库拉取基础数据
        for repo in TARGET_REPOS_LIST:
            # print(f"正在拉取 {repo} 数据...")
            # commits_cnt, authors = fetch_commits(repo, start_str, end_str)
            total_contributors, period_contributors = get_github_contributor_stats(repo, start_str, end_str, GH_PAT)
            prs_t, prs_m, prs_c, prs_o, _ = fetch_prs(repo, start_str, end_str)
            open_issues, close_issues = fetch_issues(repo, start_str, end_str)
            stars_n, stars_t, forks_n, forks_t = fetch_stars_forks(repo, start_str)

            # 汇总数据
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
            total_data["stars_total"] = stars_t  # 多仓库场景可改为列表存储，此处简化取最后一个
            total_data["forks_new"] += forks_n
            total_data["forks_total"] = forks_t

        # 7. 输出数据（供Action后续生成报告使用，新增commit_authors用于计算Contributors）
        # print(f"PRS_TOTAL={total_data['prs_total']}")
        # print(f"PRS_MERGED={total_data['prs_merged']}")
        # print(f"PRS_CLOSED={total_data['prs_closed']}")
        # print(f"PRS_OPEN={total_data['prs_open']}")
        # print(f"STARS_NEW={total_data['stars_new']}")
        # print(f"STARS_TOTAL={total_data['stars_total']}")
        # print(f"FORKS_NEW={total_data['forks_new']}")
        # print(f"FORKS_TOTAL={total_data['forks_total']}")
        # print(f"OPEN_ISSUES={total_data['open_issues']}")
        # print(f"CLOSE_ISSUES={total_data['close_issues']}")
        prs_closed_total = total_data["prs_merged"] + total_data["prs_closed"]

        print("\n================ GitHub 社区统计汇总 ================\n")

        print("【Pull Request】")
        print(f"  - PR 总数：{total_data['prs_total']}")
        print(f"  - 已关闭 PR（Merged + Closed）：{prs_closed_total}")
        print(f"  - 当前 Open PR：{total_data['prs_open']}\n")

        print("【Issue】")
        print(f"  - Issue 总数：{total_data['issues_total']}")
        print(f"  - 已关闭 Issue：{total_data['close_issues']}")
        print(f"  - 当前 Open Issue：{total_data['open_issues']}\n")

        print("【社区指标】")
        # print(f"  - Star 总数：{total_data['stars_total']}（本周期新增 {total_data['stars_new']}）")
        # print(f"  - Fork 总数：{total_data['forks_total']}（本周期新增 {total_data['forks_new']}）")
        print(f"  - Star 总数：{total_data['stars_total']}")
        print(f"  - Fork 总数：{total_data['forks_total']}")
        print(f"  - 参与贡献者：{total_data['total_contributors']}")

        plot_github_stats(total_data, start_str, end_str)
        print("\n可视化数据展示写入成功\n")

    except Exception as e:
        print(f"数据拉取失败：{str(e)}")
        exit(1)  # 退出码1表示失败，便于Action日志定位

if __name__ == "__main__":
    """
    主函数，程序入口
    """
    main()