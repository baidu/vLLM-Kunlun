# 说明

## 配置环境

```
python3 -m venv venv/dev-cicd
source venv/dev-cicd/bin/activate 
```

## 启动服务

```bash
export START_DATE=2025-12-08
export END_DATE=2026-01-24
export TARGET_REPOS=baidu/vLLM-Kunlun
# 配置 github 个人 token，见 https://github.com/settings/tokens
export GH_PAT=xxx

python AI-Tools/cicd/collect_github_data.py 
```

