#!/usr/bin/env bash
# HF Spaces 一键发版：main 已提交内容 + data/index（LFS）→ 推到 hf remote 的 main。
#
# 为什么用临时 worktree：在主工作区反复 checkout 部署分支，会在切回 main 时把
# 被 gitignore 的 data/index 从磁盘清掉（tracked→untracked 切换的 git 行为，
# 2026-07-04 实际踩过两次）。改在一次性 worktree 里打包提交，主工作区零接触。
#
# 用法：
#   bash scripts/deploy_hf.sh            # 发版（push 时提示输入 HF 用户名 + write token）
#   bash scripts/deploy_hf.sh --dry-run  # 演练：构建部署提交并校验，不 push
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# ── 前置检查 ──────────────────────────────────────────────────────────
git lfs version >/dev/null 2>&1 \
  || { echo "✗ 需要 git-lfs：brew install git-lfs && git lfs install"; exit 1; }
git remote get-url hf >/dev/null 2>&1 \
  || { echo "✗ 未配置 hf remote：git remote add hf https://huggingface.co/spaces/<用户>/<space>"; exit 1; }
.venv/bin/python scripts/preflight_hf.py || { echo "✗ preflight 未通过，中止"; exit 1; }

MAIN_SHA=$(git rev-parse --short main)
echo "→ 部署内容：main@${MAIN_SHA} + data/index（LFS）"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "⚠ 工作区有未提交改动：本次发版只含已提交到 main 的内容"
fi

# ── 临时 worktree 里打包部署提交 ─────────────────────────────────────
WT=$(mktemp -d /tmp/hf-deploy.XXXXXX)
cleanup() {
  git worktree remove --force "$WT" >/dev/null 2>&1 || true
  rm -rf "$WT" 2>/dev/null || true
}
trap cleanup EXIT

git worktree add --detach "$WT" main >/dev/null 2>&1
mkdir -p "$WT/data"
cp -R "$ROOT/data/index" "$WT/data/index"
git -C "$WT" add -f data/index
git -C "$WT" commit -q -m "chore: bundle index for HF deploy (main@${MAIN_SHA})"
DEPLOY_SHA=$(git -C "$WT" rev-parse HEAD)

# LFS 校验：索引二进制必须是指针，否则 HF pre-receive 会拒（踩过）
N_LFS=$(git -C "$WT" lfs ls-files | wc -l | tr -d ' ')
echo "→ 部署提交 ${DEPLOY_SHA:0:7}，LFS 追踪 ${N_LFS} 个索引文件"
if [[ "$N_LFS" -lt 10 ]]; then
  echo "✗ LFS 追踪数异常（${N_LFS} < 10）：检查 .gitattributes 的 data/index/** 规则"
  exit 1
fi

git branch -f hf-deploy "$DEPLOY_SHA"  # 本地留痕便于排查；不 checkout，主工作区不动

if $DRY_RUN; then
  echo "✓ dry-run 通过（未 push）。正式发版：bash scripts/deploy_hf.sh"
  exit 0
fi

echo "→ 推送到 HF（LFS 上传 + git push，可能提示两次凭证，均用 write token）..."
git push hf "${DEPLOY_SHA}:refs/heads/main" --force

cat <<'DONE'
✓ 已推送，Space 将自动重建。验证：
  1. Space 页面徽章 Building → Running（绿）
  2. Logs → Container 出现 "index ready at startup" 与 "Uvicorn running ... 7860"
  3. 页面里问「等候期是多少天」，应返回带 [产品-条号/章节-页码] 引用的回答
DONE
