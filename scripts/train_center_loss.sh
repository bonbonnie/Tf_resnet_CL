#!/bin/bash
# Usage:
# ./scripts/train.sh GPU
#
# Example:
# ./scripts/train.sh 1

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1

LOG="logs/center_loss_log.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

CUDA_VISIBLE_DEVICES=${GPU_ID} ./resnet_finetune_center_loss.py
