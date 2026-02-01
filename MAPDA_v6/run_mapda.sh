#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

API_URL_DEFAULT="http://10.25.0.115:8010/v1"
API_KEY_DEFAULT="any"
MODEL_DEFAULT="qwen3-32b"
PERSONALITIES_DEFAULT="mbti"
SCENARIOS_DEFAULT="INTERVIEW,DEBATE,COURT,INVESTIGATION,BUSINESS,ACADEMIC,GAME"
QUESTION_FOLDER_DEFAULT="/home/yaoy/sourcer/benchmark/MultiAgentPolyDialogueAttack/MAPDA_v6/Data/Chains"
OUTPUT_DIR_DEFAULT="${SCRIPT_DIR}/outputs"
MAX_ROWS_DEFAULT="5"
ROLE_MIN_DEFAULT="2"
ROLE_MAX_DEFAULT="5"
ROUND_MIN_DEFAULT="3"
ROUND_MAX_DEFAULT="5"
MIN_INTERVAL_DEFAULT="0"
MAX_WORKERS_DEFAULT="16"
MAX_TOKENS_DEFAULT="1024"

API_URL="${API_URL_DEFAULT}"
API_KEY="${API_KEY_DEFAULT}"
MODEL="${MODEL_DEFAULT}"
PERSONALITIES="${PERSONALITIES_DEFAULT}"
SCENARIOS="${SCENARIOS_DEFAULT}"
QUESTION_FOLDER="${QUESTION_FOLDER_DEFAULT}"
OUTPUT_DIR="${OUTPUT_DIR_DEFAULT}"
MAX_ROWS_PER_FILE="${MAX_ROWS_DEFAULT}"
ROLE_MIN="${ROLE_MIN_DEFAULT}"
ROLE_MAX="${ROLE_MAX_DEFAULT}"
ROUND_MIN="${ROUND_MIN_DEFAULT}"
ROUND_MAX="${ROUND_MAX_DEFAULT}"
MIN_INTERVAL="${MIN_INTERVAL_DEFAULT}"
MAX_WORKERS="${MAX_WORKERS_DEFAULT}"
MAX_TOKENS="${MAX_TOKENS_DEFAULT}"
DRY_RUN="0"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --api-url URL              API base URL (default: ${API_URL_DEFAULT})
  --api-key KEY              API key (default: ${API_KEY_DEFAULT})
  --model NAME               Model name (default: ${MODEL_DEFAULT})
  --personalities LIST       Comma list: none,mbti,big5 (default: ${PERSONALITIES_DEFAULT})
  --scenarios LIST           Comma list: interview,debate,court,investigation,business,academic,game
                             (default: ${SCENARIOS_DEFAULT})
  --question-folder PATH     CSV目录 (default: ${QUESTION_FOLDER_DEFAULT})
  --output-dir PATH           输出目录 (default: ${OUTPUT_DIR_DEFAULT})
  --max-rows N               每个类别前N条 (default: ${MAX_ROWS_DEFAULT})
  --role-min N               角色数最小值 (default: ${ROLE_MIN_DEFAULT})
  --role-max N               角色数最大值 (default: ${ROLE_MAX_DEFAULT})
  --round-min N              轮次最小值 (default: ${ROUND_MIN_DEFAULT})
  --round-max N              轮次最大值 (default: ${ROUND_MAX_DEFAULT})
  --dry-run                  仅打印任务列表，不执行
  --min-interval SECONDS     请求间隔(秒), 全局节流 (default: ${MIN_INTERVAL_DEFAULT})
  --max-workers N            线程数上限 (default: ${MAX_WORKERS_DEFAULT})
  --max-tokens N             输出最大token数 (default: ${MAX_TOKENS_DEFAULT})
  -h, --help                 显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-url) API_URL="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --personalities) PERSONALITIES="$2"; shift 2 ;;
    --scenarios) SCENARIOS="$2"; shift 2 ;;
    --question-folder) QUESTION_FOLDER="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --max-rows) MAX_ROWS_PER_FILE="$2"; shift 2 ;;
    --role-min) ROLE_MIN="$2"; shift 2 ;;
    --role-max) ROLE_MAX="$2"; shift 2 ;;
    --round-min) ROUND_MIN="$2"; shift 2 ;;
    --round-max) ROUND_MAX="$2"; shift 2 ;;
    --min-interval) MIN_INTERVAL="$2"; shift 2 ;;
    --max-workers) MAX_WORKERS="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --dry-run) DRY_RUN="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ ! -d "${QUESTION_FOLDER}" ]]; then
  echo "Question folder not found: ${QUESTION_FOLDER}"
  exit 1
fi

cd "${SCRIPT_DIR}"

# Default: disable proxy settings unless user exports them explicitly.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

normalize_scenarios() {
  local raw="$1"
  local out=()
  IFS=',' read -r -a items <<< "${raw}"
  for item in "${items[@]}"; do
    item="$(echo "${item}" | tr '[:lower:]' '[:upper:]' | xargs)"
    if [[ "${item}" == "BUSSINESS" ]]; then
      item="BUSINESS"
    fi
    case "${item}" in
      INTERVIEW|DEBATE|COURT|INVESTIGATION|BUSINESS|ACADEMIC|GAME)
        out+=("${item}")
        ;;
      "")
        ;;
      *)
        echo "Unknown scenario: ${item}"
        exit 1
        ;;
    esac
  done
  (IFS=','; echo "${out[*]}")
}

SCENARIOS="$(normalize_scenarios "${SCENARIOS}")"

export MAPDA_MIN_REQUEST_INTERVAL="${MIN_INTERVAL}"
export MAPDA_MAX_TOKENS="${MAX_TOKENS}"

cat > "${SCRIPT_DIR}/Api.json" <<EOF
{
  "api": [
    {
      "name": "custom",
      "url": "${API_URL}",
      "key": "${API_KEY}"
    }
  ]
}
EOF

PY_ARGS=(
  "--api-name" "custom"
  "--model" "${MODEL}"
  "--personalities" "${PERSONALITIES}"
  "--scenarios" "${SCENARIOS}"
  "--question-folder" "${QUESTION_FOLDER}"
  "--output-dir" "${OUTPUT_DIR}"
  "--max-rows-per-file" "${MAX_ROWS_PER_FILE}"
  "--role-min" "${ROLE_MIN}"
  "--role-max" "${ROLE_MAX}"
  "--round-min" "${ROUND_MIN}"
  "--round-max" "${ROUND_MAX}"
  "--max-workers" "${MAX_WORKERS}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  PY_ARGS+=("--dry-run")
fi

python "${SCRIPT_DIR}/GenerateConv.py" "${PY_ARGS[@]}"
