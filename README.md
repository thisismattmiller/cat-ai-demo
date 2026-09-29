# cat-ai-demo

Give it an LCCN and three AI cataloging assistants run at once, streaming their work to the browser over a
WebSocket as each one finishes:

| task | what | source |
|---|---|---|
| `record` | the LC record from id.loc.gov (BIBFRAME instance + work), ISBNdb-enriched when LC's record is thin | `lambda/shelflister/idloc.py` |
| `subjects` | LCSH + LCC candidates from the nearest records in LC's catalog (vector search), an LLM picks the fit | the embedded-catalog **subject-suggest lambda** (`SUBJECT_SUGGEST_URL`) |
| `shelflist` | a full LC call number: class number from the 2024 schedules, then Cutter + date by the *Classification and Shelflisting Manual*, fitted among the shelflist neighbours | [shelflisting_manual](https://github.com/thisismattmiller/shelflisting_manual), vendored as `lambda/shelflister` + `lambda/rules` |
| `names` | LCNAF links (Wikidata as fallback) for the record's contributors that have no authority URI | [nar-auto-reconcile](https://github.com/thisismattmiller/nar-auto-reconcile), vendored as `lambda/nar_reconcile` |

The two vendored tools are copied verbatim apart from small patches: file locations come from environment
variables (schedule DB, caches), the subject-suggest URL is `SUBJECT_SUGGEST_URL`, and each pipeline got a
progress hook (`Provider.on_event` for the shelflister's tool calls, `on_book` / `on_contributor` for the name
reconciler) so the demo can show the work as it happens.

```
docs/            the demo page (GitHub Pages): docs/index.html + docs/config.js (wss:// URL, generated)
lambda/          the function: app.py (handler) · tasks.py (the four tasks) · ws.py (events) · Dockerfile
deploy/          build.sh (image -> ECR) · setup_aws.sh (role, function, WebSocket API) · ws_client.py · mock_ws.py
local_run.py     run the tasks locally, no AWS
```

## Run locally

```sh
uv sync --all-groups
export CLAUDE_PAID_API=...  GOOGLE_AI=...  ISBNDB_API_KEY=...
ln -s ~/git/shelflisting_manual/.cache/lcc.sqlite lambda/build/lcc.sqlite    # the parsed LC schedules (256 MB)
uv run local_run.py 2025947561                          # all four tasks, events as JSON lines
uv run local_run.py 2025947561 --tasks shelflist --options '{"hide_class": true}'
```

The schedule index comes from the shelflister repo (built there from `~/git/lcc_pdfs_2024/json`); `deploy/build.sh`
copies it, or downloads it from `s3://$ASSET_BUCKET/lcc.sqlite` once it has been uploaded (`setup_aws.sh --upload-db`).

To work on the page without AWS, replay recorded events:

```sh
uv run deploy/mock_ws.py deploy/sample_events.jsonl --speed 10 &
(cd docs && python3 -m http.server 8766)
open 'http://localhost:8766/?ws=ws://localhost:8765&lccn=2025947561'
```

## Deploy

One Lambda function (container image, arm64, 2 GB, 15 min) behind an API Gateway **WebSocket** API.

```sh
deploy/build.sh                    # docker build (linux/arm64) -> push to ECR cat-ai-demo -> update the function's code if it exists
deploy/setup_aws.sh                # IAM role, function, environment (keys from your shell), WebSocket API + routes + stage; writes docs/config.js
deploy/setup_aws.sh --upload-db    # also put lcc.sqlite in S3 for builds elsewhere
uv run deploy/ws_client.py 2025947561       # end-to-end test against the deployed socket
```

`setup_aws.sh` tries to create the role `cat-ai-demo-lambda` (trust + policy in `deploy/iam/`). If the CLI user
may not create roles, create it in the console from those files or run with `ROLE_NAME=<existing role>` that has
CloudWatch logs and `execute-api:ManageConnections`.

Function environment: `CLAUDE_PAID_API` (or `ANTHROPIC_API_KEY`), `GOOGLE_AI`, `ISBNDB_API_KEY`,
`SUBJECT_SUGGEST_URL`, `SHELFLISTER_LCC_DB`, `SHELFLISTER_CACHE_DIR=/tmp/cache`, `TASK_FANOUT`; optional
`SHELFLISTER_MODEL` (default gemini-3.8-flash), `NAR_PROVIDER` / `NAR_MODEL` (default claude / claude-sonnet-5-5).

### How a run flows

```
browser ──(wss)──▶ API Gateway $default ──▶ Lambda app.handler
   {"action":"run","lccn":"2025947561"}          │ validates, posts {"type":"started"}
                                                 │ runs record / subjects / shelflist / names as threads
   ◀── progress / partial / result / error ──────┘ each task posts to the connection as it goes
```

`TASK_FANOUT=threads` (default): the tasks are threads of one invocation, which lasts until the slowest
finishes. API Gateway stops waiting for the integration response after 29 s, but the invocation keeps running and
keeps posting. `TASK_FANOUT=lambda` invokes the function asynchronously once per task instead (needs
`lambda:InvokeFunction` on itself).

### Message protocol

Client → server: `{"action": "run", "lccn": "...", "tasks": ["record","subjects","shelflist","names"], "options": {...}}`
or `{"action": "ping"}`. Options may be flat or per task, e.g. `{"shelflist": {"hide_class": true, "hide_subjects": false,
"model": "sonnet"}, "names": {"provider": "gemini"}, "subjects": {"top_k": 10}}`.

Server → client, every event carries `task`, `lccn` and `t` (seconds since the task started):

| type | payload |
|---|---|
| `started` | `tasks` that will run |
| `progress` | `message`; for shelflister tool calls also `tool`, `input`, `output` (first 300 chars), `phase` |
| `partial` | `data` with a `stage`: `classified` (class number before the book number is built), `book` (contributors on the record), `contributor` (one name's result) |
| `result` | `data`: the task's full result (see `lambda/tasks.py`) |
| `error` | `error` text |

By default the shelflister does **not** see LC's own call number and the record is hidden from the shelflist
(`hide_class: true`), so a record LC has already classified is a real test; the result carries a `comparison` with
LC's number. The subject task likewise reports which candidates are already on LC's record.
.
## GitHub Pages

The page is `docs/index.html`; publish `main` / `docs` under the repository's Pages settings
(`gh api repos/thisismattmiller/cat-ai-demo/pages -X POST -f 'source[branch]=main' -f 'source[path]=/docs'`).
`docs/config.js` holds the WebSocket URL (`?ws=wss://…` on the URL overrides it).
