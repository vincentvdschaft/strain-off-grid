#!/bin/bash
set -euo pipefail
OUT_DIR=out/sweep
mkdir -p "$OUT_DIR"

if [ $# -ne 1 ]; then
    echo "Usage: $0 <output_name>"
    exit 1
fi

OUT_NAME=$1

PHANTOM_PATH=source/20251222_s3_a4ch_line_dw_0000.hdf5
for first_frame in {34..34}; do
    for first_transmit in {56..64}; do
        PHANTOM_NAME=$(basename "$PHANTOM_PATH" .hdf5)
        SWEEP_DIR="$OUT_DIR/$OUT_NAME"
        FIRST_FRAME_LABEL=$(printf "%03d" "$first_frame")
        FRAME_DIR="$SWEEP_DIR/frame-$FIRST_FRAME_LABEL"
        mkdir -p "$FRAME_DIR"

        TRANSMITS=$(printf "%03d-%03d" "$first_transmit" $((first_transmit + 2)))
        SOLUTION_PATH="$FRAME_DIR/tx$TRANSMITS.hdf5"

        if [ -f "$SOLUTION_PATH" ]; then
            echo "Solution already exists: $SOLUTION_PATH"
            continue
        fi

        solve "$PHANTOM_PATH" --config config/invivo.toml --transmits "$TRANSMITS" --first-frame "$first_frame" --solution-path "$SOLUTION_PATH"

    done
done