#!/usr/bin/env bash
# Demo: onboard 4 factories, inspect the model, select the highest usable
# gear under a pressure interval and power cap, then replace one factory's
# zero-point correction by version number and watch correction + model +
# per-factory errors take effect together (and a stale gear be rejected
# with a suggestion, a stale-version update conflict leave the current
# version untouched).
set -euo pipefail

BASE="${BASE_URL:-http://localhost:8000}"
cd "$(dirname "$0")"

fmt() { if command -v jq >/dev/null 2>&1; then jq .; else cat; echo; fi; }
field() { python3 -c "import json,sys; print(json.load(sys.stdin)$1)"; }

echo "=== 1. onboard 4 factories -> POST /versions"
CREATE=$(curl -sS -X POST "$BASE/versions" -H 'Content-Type: application/json' \
  --data @demo/create_version.json)
echo "$CREATE" | fmt
V=$(echo "$CREATE" | field "['version']")
echo "-> current version: $V (note lambda and F3's error)"

echo "=== 2. predict for F3 with raw readings -> POST /predict"
curl -sS -X POST "$BASE/predict" -H 'Content-Type: application/json' \
  --data @demo/predict_f3.json | fmt

echo "=== 2b. pick the highest usable gear under the raw pressure interval"
echo "    and power cap -> POST /gear/select"
GEAR=$(curl -sS -X POST "$BASE/gear/select" -H 'Content-Type: application/json' \
  --data @demo/gear_select.json)
echo "$GEAR" | fmt
G=$(echo "$GEAR" | field "['gear']")
echo "-> selected gear $G (version 1)"

echo "=== 3. replace F3's zero-point correction (0 -> 500) based on version $V"
echo "    -> POST /versions/$V/correction"
curl -sS -X POST "$BASE/versions/$V/correction" -H 'Content-Type: application/json' \
  --data @demo/correction_update.json | fmt

echo "=== 3b. review gear $G under the new version -> POST /gear/review"
echo "    the old gear now violates the power cap: HTTP 409 + suggestion"
REVIEW_BODY=$(python3 - "$G" <<'PY'
import json, sys
body = json.load(open("demo/gear_select.json"))
body["gear"] = int(sys.argv[1])
print(json.dumps(body))
PY
)
REVIEW=$(curl -sS -w '\nHTTP %{http_code}\n' -X POST "$BASE/gear/review" \
  -H 'Content-Type: application/json' --data "$REVIEW_BODY")
echo "$REVIEW"

echo "=== 4. current version now: lambda + per-factory errors"
curl -sS "$BASE/versions/current" | fmt

echo "=== 5. predict again for F3 (uses the new version)"
curl -sS -X POST "$BASE/predict" -H 'Content-Type: application/json' \
  --data @demo/predict_f3.json | fmt

echo "=== 5b. select again under the new version -> POST /gear/select"
curl -sS -X POST "$BASE/gear/select" -H 'Content-Type: application/json' \
  --data @demo/gear_select.json | fmt

echo "=== 6. conflict: replay the update against stale version $V -> HTTP 409,"
echo "    current version kept"
curl -sS -o /dev/null -w "HTTP %{http_code}\n" -X POST \
  "$BASE/versions/$V/correction" -H 'Content-Type: application/json' \
  --data @demo/correction_update.json
curl -sS "$BASE/versions/current" | python3 -c \
  "import json,sys; j=json.load(sys.stdin); print('current version:', j['version'], '| F3 correction:', j['corrections']['F3'])"
