# Lambda container image (arm64) for the cat-ai-demo function.
# Build with deploy/build.sh: it stages build/lcc.sqlite (the parsed LC schedules, 256 MB) and pushes to ECR.
FROM public.ecr.aws/lambda/python:3.14

# dependencies first (rarely change)
COPY requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt --target ${LAMBDA_TASK_ROOT}

# the schedules index (big, changes least)
COPY build/lcc.sqlite ${LAMBDA_TASK_ROOT}/build/lcc.sqlite

# the CSM rule cards and the two tools
COPY rules ${LAMBDA_TASK_ROOT}/rules
COPY shelflister ${LAMBDA_TASK_ROOT}/shelflister
COPY nar_reconcile ${LAMBDA_TASK_ROOT}/nar_reconcile

# the handler
COPY ws.py tasks.py app.py ${LAMBDA_TASK_ROOT}/

ENV SHELFLISTER_LCC_DB=/var/task/build/lcc.sqlite \
    SHELFLISTER_CACHE_DIR=/tmp/cache \
    PYTHONUNBUFFERED=1

CMD ["app.handler"]
