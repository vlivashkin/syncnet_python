#!/bin/bash
set -ex

SCRIPT_PATH=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_PATH"/..

VIDEO_PATH=$1

python3 ./syncnet_metric.py \
    --video_path $VIDEO_PATH \
    --device cuda:1