#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-http://127.0.0.1:7868}"
MODEL="${MODEL:-}"  # if empty, will pick from /api/v1/models
AUDIO="${AUDIO:-logs/111/0_gt_wavs/1_0.wav}"
LOOPS="${LOOPS:-3}"
F0_METHOD="${F0_METHOD:-rmvpe}"
SPEAKER_ID="${SPEAKER_ID:-0}"
TMPDIR="${TMPDIR:-/tmp}"

usage() {
  cat <<EOF
Usage: HOST=... MODEL=... AUDIO=... LOOPS=... bash scripts/e2e_convert.sh
Env vars:
  HOST        Target server (default: http://127.0.0.1:7868)
  MODEL       Model relative path (e.g., History.pth). If empty, auto-pick first from /api/v1/models
  AUDIO       Path to local wav/m4a/mp3 input (default: logs/111/0_gt_wavs/1_0.wav)
  LOOPS       Number of iterations (default: 3)
  F0_METHOD   pm|harvest|crepe|rmvpe (default: rmvpe)
  SPEAKER_ID  Speaker id (default: 0)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage; exit 0
fi

# health check
echo "[e2e] GET /health on ${HOST}"
HTTP_HEALTH=$(curl -sS -w "%{http_code}" -o /dev/null "${HOST}/health" || true)
if [[ "$HTTP_HEALTH" != "200" ]]; then
  echo "[e2e][ERR] health check failed: ${HTTP_HEALTH}" >&2
  exit 1
fi

echo "[e2e] GET /api/v1/models"
MODELS_JSON=$(curl -sS "${HOST}/api/v1/models")
CODE=$(echo "$MODELS_JSON" | python3 -c 'import sys, json; print(json.load(sys.stdin).get("code", 1))' || echo 1)
if [[ "$CODE" != "0" ]]; then
  echo "[e2e][ERR] list models failed: $MODELS_JSON" >&2
  exit 1
fi
if [[ -z "$MODEL" ]]; then
  MODEL=$(echo "$MODELS_JSON" | python3 -c 'import sys, json; print(json.load(sys.stdin)["models"][0])')
  echo "[e2e] Auto-picked model: $MODEL"
else
  echo "[e2e] Using model: $MODEL"
fi

if [[ ! -f "$AUDIO" ]]; then
  echo "[e2e][ERR] audio file not found: $AUDIO" >&2
  exit 1
fi

# loop convert
DURS=()
for ((i=1; i<=LOOPS; i++)); do
  OUT_WAV="${TMPDIR}/convert_${i}.wav"
  HDRS="${TMPDIR}/convert_${i}.headers"
  echo "[e2e] (${i}/${LOOPS}) POST /api/v1/convert ..."
  start_ns=$(date +%s%N)
  set +e
  curl -sS -D "$HDRS" -o "$OUT_WAV" \
    -F "model=${MODEL}" \
    -F "spk_id=${SPEAKER_ID}" \
    -F "f0_up_key=0" \
    -F "f0_method=${F0_METHOD}" \
    -F "index_rate=1.0" \
    -F "filter_radius=3" \
    -F "resample_sr=48000" \
    -F "rms_mix_rate=1.0" \
    -F "protect=0.33" \
    -F "loudnorm=-26" \
    -F "return_format=wav" \
    -F "audio_file=@${AUDIO}" \
    "${HOST}/api/v1/convert"
  rc=$?
  set -e
  end_ns=$(date +%s%N)
  dur_ms=$(( (end_ns - start_ns) / 1000000 ))

  if [[ $rc -ne 0 ]]; then
    echo "[e2e][ERR] convert request failed (rc=$rc)"
    exit $rc
  fi

  CT=$(grep -i '^content-type:' "$HDRS" | awk '{print tolower($2)}' | tr -d '\r')
  if [[ "$CT" != "audio/wav" ]]; then
    echo "[e2e][WARN] unexpected content-type: $CT"
  fi
  if head -c 4 "$OUT_WAV" | grep -q 'RIFF'; then
    echo "[e2e] (${i}/${LOOPS}) OK, ${dur_ms} ms -> $OUT_WAV"
  else
    echo "[e2e][ERR] output not RIFF WAV: $OUT_WAV" >&2
    exit 1
  fi
  DURS+=("$dur_ms")

done

echo "[e2e] Durations (ms): ${DURS[*]}"
# simple average
AVG=$(printf "%s\n" "${DURS[@]}" | awk '{s+=$1} END{ if (NR>0) printf("%d", s/NR); else print 0 }')
echo "[e2e] Average: ${AVG} ms"

echo "[e2e] Done. Note: 若第二次/第三次显著更快，通常意味着会话与 rmvpe/Hubert 等资源已缓存并复用。"
