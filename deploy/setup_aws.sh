#!/usr/bin/env bash
# Create (or update) everything on AWS: IAM role, Lambda function (from the image build.sh pushed),
# the WebSocket API with its routes and stage, and the function's environment (API keys from your shell).
# Idempotent: rerun after changing keys or settings. Writes the wss:// URL into docs/config.js.
#
#   deploy/build.sh && deploy/setup_aws.sh
#   deploy/setup_aws.sh --upload-db      # also put lambda/build/lcc.sqlite in s3://$ASSET_BUCKET
source "$(dirname "$0")/common.sh"
IMAGE="$(cat "$ROOT/lambda/build/image_uri" 2>/dev/null || true)"
[[ -n "$IMAGE" ]] || { echo "no lambda/build/image_uri: run deploy/build.sh first"; exit 1; }
for k in CLAUDE_PAID_API GOOGLE_AI ISBNDB_API_KEY; do
    [[ -n "${!k:-}" ]] || echo "warning: $k is not set in this shell; the function will not get it"
done

# ---------------------------------------------------------------- IAM role
# The function needs CloudWatch logs and execute-api:ManageConnections (deploy/iam/role-policy.json).
# If $ROLE_NAME does not exist the script tries to create it; when the CLI user may not (iam:CreateRole
# denied), create it in the console from deploy/iam/*.json or point ROLE_NAME at an existing role that
# already has those permissions (e.g. ROLE_NAME=tsundoku-scan-ws-lambda-role in this account).
ROLE_ARN="arn:aws:iam::$ACCOUNT:role/$ROLE_NAME"
NEW_ROLE=false
if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
    log "creating role $ROLE_NAME"
    if aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document "file://$ROOT/deploy/iam/trust.json" >/dev/null; then
        aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
        sed "s/REGION/$REGION/g; s/ACCOUNT/$ACCOUNT/g; s/FUNCTION_NAME/$FUNCTION_NAME/g" "$ROOT/deploy/iam/role-policy.json" > /tmp/cat-ai-demo-policy.json
        aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name cat-ai-demo-inline --policy-document file:///tmp/cat-ai-demo-policy.json
        NEW_ROLE=true
    else
        echo "cannot create $ROLE_NAME: create it in the console from deploy/iam/trust.json + role-policy.json,"
        echo "or rerun with ROLE_NAME=<an existing role with logs + execute-api:ManageConnections>"
        exit 1
    fi
fi
log "role: $ROLE_ARN"

# ---------------------------------------------------------------- environment
ENV_FILE="$(mktemp)"
python3 - "$ENV_FILE" <<PY
import json, os, sys
v = {k: os.environ[k] for k in ("CLAUDE_PAID_API", "GOOGLE_AI", "ISBNDB_API_KEY", "SHELFLISTER_MODEL", "NAR_PROVIDER", "NAR_MODEL", "NAR_USER_AGENT") if os.environ.get(k)}
v["SUBJECT_SUGGEST_URL"] = "$SUBJECT_SUGGEST_URL"
v["SHELFLISTER_CACHE_DIR"] = "/tmp/cache"
v["SHELFLISTER_LCC_DB"] = "/var/task/build/lcc.sqlite"
v["TASK_FANOUT"] = os.environ.get("TASK_FANOUT", "threads")
json.dump({"Variables": v}, open(sys.argv[1], "w"))
PY

# ---------------------------------------------------------------- Lambda function
if ! aws lambda get-function --function-name "$FUNCTION_NAME" >/dev/null 2>&1; then
    log "creating function $FUNCTION_NAME from $IMAGE"
    $NEW_ROLE && sleep 10     # IAM propagation
    for attempt in 1 2 3 4 5 6; do
        if aws lambda create-function --function-name "$FUNCTION_NAME" --package-type Image --code "ImageUri=$IMAGE" \
              --role "$ROLE_ARN" --architectures arm64 --memory-size "$MEMORY_MB" --timeout "$TIMEOUT_S" \
              --environment "file://$ENV_FILE" --description "LCCN -> subject suggest, LC call number, name reconciliation (WebSocket)" \
              --query FunctionArn --output text; then break; fi
        echo "retrying in 10s (role propagation)"; sleep 10
    done
    aws lambda wait function-active --function-name "$FUNCTION_NAME"
else
    log "updating $FUNCTION_NAME configuration"
    aws lambda wait function-updated --function-name "$FUNCTION_NAME"
    aws lambda update-function-configuration --function-name "$FUNCTION_NAME" --memory-size "$MEMORY_MB" --timeout "$TIMEOUT_S" \
        --environment "file://$ENV_FILE" --query 'LastUpdateStatus' --output text
    aws lambda wait function-updated --function-name "$FUNCTION_NAME"
fi
rm -f "$ENV_FILE"
FUNCTION_ARN="arn:aws:lambda:$REGION:$ACCOUNT:function:$FUNCTION_NAME"

# ---------------------------------------------------------------- WebSocket API
API_ID="$(aws apigatewayv2 get-apis --query "Items[?Name=='$API_NAME'].ApiId | [0]" --output text)"
if [[ -z "$API_ID" || "$API_ID" == "None" ]]; then
    log "creating WebSocket API $API_NAME"
    API_ID="$(aws apigatewayv2 create-api --name "$API_NAME" --protocol-type WEBSOCKET \
        --route-selection-expression '$request.body.action' --query ApiId --output text)"
fi
INT_ID="$(aws apigatewayv2 get-integrations --api-id "$API_ID" --query "Items[?IntegrationUri=='$FUNCTION_ARN'].IntegrationId | [0]" --output text)"
if [[ -z "$INT_ID" || "$INT_ID" == "None" ]]; then
    log "creating Lambda proxy integration"
    INT_ID="$(aws apigatewayv2 create-integration --api-id "$API_ID" --integration-type AWS_PROXY \
        --integration-uri "$FUNCTION_ARN" --query IntegrationId --output text)"
fi
EXISTING_ROUTES="$(aws apigatewayv2 get-routes --api-id "$API_ID" --query 'Items[].RouteKey' --output text)"
for r in '$connect' '$disconnect' '$default'; do
    if ! grep -qF -- "$r" <<<"$EXISTING_ROUTES"; then
        log "route $r"
        aws apigatewayv2 create-route --api-id "$API_ID" --route-key "$r" --target "integrations/$INT_ID" >/dev/null
    fi
done
if ! aws apigatewayv2 get-stage --api-id "$API_ID" --stage-name "$STAGE" >/dev/null 2>&1; then
    log "stage $STAGE (auto-deploy)"
    aws apigatewayv2 create-stage --api-id "$API_ID" --stage-name "$STAGE" --auto-deploy >/dev/null
fi
aws lambda add-permission --function-name "$FUNCTION_NAME" --statement-id "apigw-ws-$API_ID" --action lambda:InvokeFunction \
    --principal apigateway.amazonaws.com --source-arn "arn:aws:execute-api:$REGION:$ACCOUNT:$API_ID/*" >/dev/null 2>&1 || true

WSS="wss://$API_ID.execute-api.$REGION.amazonaws.com/$STAGE"
log "WebSocket: $WSS"
cat > "$ROOT/docs/config.js" <<JS
// generated by deploy/setup_aws.sh
window.CAT_AI_DEMO_WS = "$WSS";
JS
log "wrote docs/config.js"

# ---------------------------------------------------------------- schedules DB to S3 (optional)
if [[ "${1:-}" == "--upload-db" ]]; then
    aws s3api head-bucket --bucket "$ASSET_BUCKET" 2>/dev/null || { log "creating bucket $ASSET_BUCKET"; aws s3 mb "s3://$ASSET_BUCKET" --region "$REGION"; }
    log "uploading lcc.sqlite to s3://$ASSET_BUCKET/"
    aws s3 cp "$ROOT/lambda/build/lcc.sqlite" "s3://$ASSET_BUCKET/lcc.sqlite"
fi
log "all set. try: uv run deploy/ws_client.py 2025947561"
