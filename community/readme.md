# GitHub CI/CD 数据采集工具（collect_github_data）

该工具用于从 GitHub 拉取指定仓库在某个时间窗口内的活动数据（如PR、Issue、Contributors、Stars 等），用于后续统计分析/可视化。

适用于：
- 团队研发效能统计（某段时间内PR/Issue/Contributors/Stars）
- CI/CD 变更追踪（按时间区间）
- 单仓库/多仓库数据批量采集

---

## 目录结构

```
community/
  collect_github_data.py
```

---

## 运行环境要求

- Python 3.8+（建议 3.10+）
- 可访问 GitHub API 的网络环境
- GitHub Personal Access Token（PAT）

---

## 快速开始

### 1）创建虚拟环境（推荐）

```bash
python3 -m venv venv/dev-cicd
source venv/dev-cicd/bin/activate
```

（可选）升级 pip：

```bash
pip install -U pip
```

---

### 2）配置 GitHub Token（必须）

该工具通过 GitHub API 拉取数据。为避免触发 API 限速、以及确保能访问私有仓库，建议配置个人 Token。

1. 打开 Token 创建页面：  
   https://github.com/settings/tokens
2. 创建 Token（只读权限即可）
3. 将 Token 配置为环境变量：

```bash
export GH_PAT=xxx
```

> ⚠️ 请勿将 token 写入代码或提交到仓库。

---

### 3）配置采集参数并启动

```bash
export START_DATE=2025-12-08
export END_DATE=2026-01-24
export TARGET_REPOS=baidu/vLLM-Kunlun

python AI-Tools/cicd/collect_github_data.py
```

---

## 参数说明

工具通过环境变量传参：

| 参数 | 必填 | 示例 | 说明 |
|------|------|------|------|
| `GH_PAT` | ✅ | `ghp_xxx` | GitHub Personal Access Token |
| `START_DATE` | ✅ | `2025-12-08` | 采集开始日期（格式：YYYY-MM-DD） |
| `END_DATE` | ✅ | `2026-01-24` | 采集结束日期（格式：YYYY-MM-DD） |
| `TARGET_REPOS` | ✅ | `baidu/vLLM-Kunlun` | 目标仓库，支持多个 |

---

## 多仓库采集

支持多个仓库用英文逗号分隔：

```bash
export TARGET_REPOS=baidu/vLLM-Kunlun,baidu/another-repo,openai/some-repo
python AI-Tools/cicd/collect_github_data.py
```

---

## 输出说明

脚本执行完成后会生成采集数据文件（如 JSON/CSV 等），用于后续统计分析。

> 输出目录与文件命名规则以 `collect_github_data.py` 实现为准。  
> 建议运行后查看控制台日志，或检查当前目录下的 `output/`、`data/` 等文件夹。

---

## 常见问题（FAQ）

### 1）提示 401 / Bad credentials
- `GH_PAT` 不正确或已过期
- Token 无访问目标仓库权限（尤其私有仓库）

解决方式：
- 重新创建 token 并导出 `GH_PAT`
- 检查仓库访问权限

---

### 2）提示 rate limit exceeded（触发限流）
- 未配置 token 或 token 权限不足导致限流阈值较低
- 采集仓库太多、时间跨度过长导致请求量大

解决方式：
- 确保 `GH_PAT` 已设置
- 缩小日期范围或减少仓库数

---

### 3）日期格式错误
日期必须是：

```
YYYY-MM-DD
```

例如：

```
2026-01-01
```

---

## 安全建议

- ✅ 使用环境变量或 `.env` 文件管理 token
- ⛔ 不要把 token 写到代码里
- ⛔ 不要把 token 写到 README / 截图 / 日志里
- ✅ 推荐把 `.env` 加入 `.gitignore`

---

## 贡献方式

欢迎通过 Issue 或 PR 贡献：

- 新增采集字段
- 丰富输出格式（CSV/JSON/Markdown 汇总）
- 增加更友好的日志和错误提示

---

## License

Apache License 2.0, as found in the LICENSE file.
