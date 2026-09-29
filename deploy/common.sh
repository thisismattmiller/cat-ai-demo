# Shared settings for the deploy scripts. Override any of these in the environment.
set -euo pipefail
export AWS_PAGER=""
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
FUNCTION_NAME="${FUNCTION_NAME:-cat-ai-demo}"
ECR_REPO="${ECR_REPO:-cat-ai-demo}"
ROLE_NAME="${ROLE_NAME:-cat-ai-demo-lambda}"
API_NAME="${API_NAME:-cat-ai-demo-ws}"
STAGE="${STAGE:-prod}"
ASSET_BUCKET="${ASSET_BUCKET:-cat-ai-demo-assets-$ACCOUNT}"     # holds lcc.sqlite for builds on other machines
MEMORY_MB="${MEMORY_MB:-2048}"
TIMEOUT_S="${TIMEOUT_S:-900}"
LCC_SQLITE_SOURCE="${LCC_SQLITE_SOURCE:-$HOME/git/shelflisting_manual/.cache/lcc.sqlite}"
SUBJECT_SUGGEST_URL="${SUBJECT_SUGGEST_URL:-https://abeniabvmaysz2npcr3sr47fxq0xgoes.lambda-url.us-east-1.on.aws/}"
IMAGE_URI="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log() { printf '\033[1;34m==> %s\033[0m\n' "$*"; }
