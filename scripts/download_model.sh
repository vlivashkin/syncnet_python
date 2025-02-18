#!/bin/bash
set -ex

SCRIPT_PATH=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_PATH"/..

mkdir -p weights
wget http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model -O weights/syncnet_v2.model
wget https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth -O weights/sfd_face.pth
