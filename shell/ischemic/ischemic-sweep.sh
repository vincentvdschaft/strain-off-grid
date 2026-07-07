#!/bin/bash
set -euo pipefail
OUT_DIR=out/sweep
mkdir -p "$OUT_DIR"

if [ $# -ne 1 ]; then
    echo "Usage: $0 <output_name>"
    exit 1
fi

OUT_NAME=$1

PHANTOM_PATH=source/simulated_phantom_ischemic.hdf5
for first_frame in {0..19}; do
    PHANTOM_NAME=$(basename $PHANTOM_PATH .hdf5)
    SWEEP_DIR="$OUT_DIR/$OUT_NAME"
    mkdir -p "$SWEEP_DIR"

    FIRST_FRAME_LABEL=$(printf "%03d" $first_frame)

    solve "$PHANTOM_PATH" --config config/cardiac.toml --transmits 0-2 --first-frame $first_frame --solution-path $SWEEP_DIR/$PHANTOM_NAME-$FIRST_FRAME_LABEL.hdf5

done