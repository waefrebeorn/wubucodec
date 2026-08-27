#!/usr/bin/env bash
# run_ab.sh — SOTA_WARPLAN_25 C19: one-command honest A/B benchmark.
# Reruns every codec on the frozen corpus and prints bytes + PSNR.
# Usage: ./tools/run_ab.sh [1080p|8k|all]
set -u
BASE=~/codec_ab1080
run_seg() { # dir segfile rawbytes
  local dir=$1 seg=$2 raw=$3
  for f in "$dir"/out*/${seg}_*; do
    [ -e "$f" ] || continue
    local name=$(basename "$f") p
    p=$(ffmpeg -i "$f" -i "$dir/seg/${seg}.mp4" -lavfi "[0:v][1:v]psnr" -f null - 2>&1 | grep -o "average:[0-9.]*")
    printf "%-6s %-24s %10d bytes %8.1fx  PSNR %s\n" "$seg" "$name" "$(stat -c%s "$f")" \
      "$(echo "$raw $(stat -c%s $f)" | awk '{printf "%.1f", $1/$2}')" "${p#average:}"
  done
}
SCOPE=${1:-all}
echo "== WUBQ A/B corpus rerun $(date -I) =="
if [ "$SCOPE" != 8k ]; then
  for s in anime movie variety; do run_seg "$BASE" "$s" "$(stat -c%s $BASE/y4m/$s.y4m)"; done
fi
if [ "$SCOPE" != 1080p ]; then
  for f in "$BASE"/out8k/*; do
    [ -e "$f" ] || continue
    local_name=$(basename "$f"); p
    p=$(ffmpeg -i "$f" -i "$BASE/seg8k/bbb.mp4" -lavfi "[0:v][1:v]psnr" -f null - 2>&1 | grep -o "average:[0-9.]*")
    printf "8k-bbb %-22s %10d bytes %8.1fx  PSNR %s\n" "$local_name" "$(stat -c%s "$f")" \
      "$(echo "3583181314 $(stat -c%s $f)" | awk '{printf "%.1f", $1/$2}')" "${p#average:}"
  done
fi
echo "Victory bars: 1080p beat x264crf23 per content type; 8K beat x265 1.90MB @ >=40.09dB."
