#!/bin/bash
set -ex

SCRIPT_PATH=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_PATH"/..

VIDEO_PATH=$1
if [ -z "$VIDEO_PATH" ]; then
  echo "No VIDEO_PATH provided."
  exit 1
fi

python3 ./syncnet_metric.py \
    --video_path "$VIDEO_PATH" \
    --output_path "$VIDEO_PATH.json" \
    --device cuda:0
