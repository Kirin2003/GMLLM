#!/usr/bin/env bash
set -u

TASK="${1:-}"
START_AT="${2:-manual}"
WORKDIR="/Data2/hxq/GMLLM"
LOGDIR="/Data2/hxq/GMLLM/.codex"
CONDA_SH="/home/hxq/anaconda3/etc/profile.d/conda.sh"

case "$TASK" in
  reservoir)
    SESSION="gmllm_deepseek_reservoir_replay"
    CONFIG="GMLLM/configs/ablations/reservoir_replay.yaml"
    LOGFILE="${LOGDIR}/deepseek_reservoir_replay_cron.log"
    ;;
  deepseek)
    SESSION="gmllm_deepseek"
    CONFIG="GMLLM/configs/profiles/deepseek.yaml"
    LOGFILE="${LOGDIR}/deepseek_cron.log"
    ;;
  *)
    echo "Usage: $0 {reservoir|deepseek} YYYYMMDD_HHMMSS" >&2
    exit 2
    ;;
esac

CRON_TAG="# CODEX_ONCE_${TASK}_${START_AT}"

mkdir -p "$LOGDIR"

{
  echo "[$(date "+%Y-%m-%d %H:%M:%S %Z %z")] Triggering ${TASK} scheduled_at=${START_AT} session=${SESSION}"

  if ! screen -S "$SESSION" -Q select . >/dev/null 2>&1; then
    screen -dmS "$SESSION"
    echo "Created screen session ${SESSION}"
  fi

  command_to_run="cd ${WORKDIR} && source ${CONDA_SH} && conda activate gmllm && python GMLLM/continual_learning_memory.py --configs ${CONFIG}"
  screen -S "$SESSION" -X stuff "${command_to_run}"
  screen -S "$SESSION" -X stuff "$(printf "\r")"

  echo "[$(date "+%Y-%m-%d %H:%M:%S %Z %z")] Command sent"
} >> "$LOGFILE" 2>&1

tmp_cron="$(mktemp)"
if crontab -l > "$tmp_cron" 2>/dev/null; then
  grep -vF "$CRON_TAG" "$tmp_cron" | crontab -
fi
rm -f "$tmp_cron"
