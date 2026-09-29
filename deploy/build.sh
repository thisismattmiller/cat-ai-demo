#!/usr/bin/env bash
# Build the Lambda container image (arm64) and push it to ECR; update the function's code if it exists.
#
#   deploy/build.sh              # build + push (+ update-function-code when the function exists)
#   deploy/build.sh --no-push    # local build only
#
# Needs lambda/build/lcc.sqlite (the parsed LC schedules). It is copied from $LCC_SQLITE_SOURCE
# (default ~/git/shelflisting_manual/.cache/lcc.sqlite) or, failing that, downloaded from s3://$ASSET_BUCKET.
source "$(dirname "$0")/common.sh"
PUSH=true
[[ "${1:-}" == "--no-push" ]] && PUSH=false

DB="$ROOT/lambda/build/lcc.sqlite"
mkdir -p "$ROOT/lambda/build"
if [[ -L "$DB" ]]; then
    log "replacing symlink with a real copy (docker COPY does not follow links)"
    cp -L "$DB" "$DB.tmp" && rm "$DB" && mv "$DB.tmp" "$DB"
fi
if [[ ! -f "$DB" ]]; then
    if [[ -f "$LCC_SQLITE_SOURCE" ]]; then
        log "copying $LCC_SQLITE_SOURCE"
        cp "$LCC_SQLITE_SOURCE" "$DB"
    else
        log "downloading s3://$ASSET_BUCKET/lcc.sqlite"
        aws s3 cp "s3://$ASSET_BUCKET/lcc.sqlite" "$DB"
    fi
fi
log "schedules DB: $(du -h "$DB" | cut -f1)"

log "exporting requirements.txt from uv.lock"
(cd "$ROOT" && uv export --no-dev --no-hashes --no-emit-project -o lambda/requirements.txt -q)

log "docker build (linux/arm64)"
docker build --platform linux/arm64 --provenance=false -t "$ECR_REPO:latest" "$ROOT/lambda"

$PUSH || exit 0

aws ecr describe-repositories --repository-names "$ECR_REPO" >/dev/null 2>&1 \
    || { log "creating ECR repository $ECR_REPO"; aws ecr create-repository --repository-name "$ECR_REPO" --image-scanning-configuration scanOnPush=false >/dev/null; }
log "pushing to $IMAGE_URI:latest"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com" >/dev/null
docker tag "$ECR_REPO:latest" "$IMAGE_URI:latest"
docker push "$IMAGE_URI:latest" | tail -1
# the pushed digest, from the local image (no ecr:DescribeImages needed)
DIGEST="$(docker image inspect "$IMAGE_URI:latest" --format '{{range .RepoDigests}}{{println .}}{{end}}' | grep "^$IMAGE_URI@" | head -1 | cut -d@ -f2)"
[[ -n "$DIGEST" ]] || { echo "could not read the pushed digest"; exit 1; }
echo "$IMAGE_URI@$DIGEST" > "$ROOT/lambda/build/image_uri"
log "image: $IMAGE_URI@$DIGEST"

if aws lambda get-function --function-name "$FUNCTION_NAME" >/dev/null 2>&1; then
    log "updating $FUNCTION_NAME code"
    aws lambda update-function-code --function-name "$FUNCTION_NAME" --image-uri "$IMAGE_URI@$DIGEST" --query 'LastUpdateStatus' --output text
    aws lambda wait function-updated --function-name "$FUNCTION_NAME"
    log "done"
else
    log "function $FUNCTION_NAME does not exist yet: run deploy/setup_aws.sh"
fi
