#!/usr/bin/env bash
# PTB-XL 1.0.3 을 PhysioNet AWS 공개 미러에서 병렬 수집한다.
# 사용법: fetch_s3.sh <keys 파일> <병렬수>
# 이미 존재하고 크기가 0 이 아닌 파일은 건너뛰므로 중단 후 재실행해도 안전하다.
set -u

KEYS="$1"
PAR="${2:-48}"
DEST=/mnt/hdd_storage/datasets/ptb-xl/ptb-xl-1.0.3
BASE=https://physionet-open.s3.amazonaws.com

export DEST BASE

fetch_one() {
  key="$1"
  rel="${key#ptb-xl/1.0.3/}"
  out="$DEST/$rel"
  if [ -s "$out" ]; then return 0; fi
  curl -sf --retry 3 --retry-delay 1 --max-time 120 --create-dirs -o "$out" "$BASE/$key" \
    || { echo "FAIL $key" >&2; rm -f "$out"; }
}
export -f fetch_one

total=$(wc -l < "$KEYS")
echo "start: $total keys, parallel=$PAR, dest=$DEST"
date +%s > /tmp/claude-1000/ptbxl_fetch_start

xargs -a "$KEYS" -P "$PAR" -I{} bash -c 'fetch_one "$@"' _ {}

echo "done downloading. verifying counts..."
echo "records100 files: $(find $DEST/records100 -type f 2>/dev/null | wc -l)"
echo "records500 files: $(find $DEST/records500 -type f 2>/dev/null | wc -l)"
du -sh "$DEST"
