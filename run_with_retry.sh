#!/bin/bash
# Retry inference until all samples are generated.
# Handles NVML crashes on MIG GPUs by resuming from last checkpoint.

LEVEL=$1
TARGET=$2  # expected number of generated images

if [ -z "$LEVEL" ] || [ -z "$TARGET" ]; then
    echo "Usage: $0 <level> <target_count>"
    echo "Example: $0 1 1000"
    exit 1
fi

OUTPUT_DIR="outputs/level${LEVEL}_1k/generated"
ATTEMPT=0

while true; do
    ATTEMPT=$((ATTEMPT + 1))
    COUNT=$(ls "$OUTPUT_DIR"/*.png 2>/dev/null | grep -v input | wc -l)
    echo "=== Attempt $ATTEMPT: $COUNT/$TARGET generated ==="

    if [ "$COUNT" -ge "$TARGET" ]; then
        echo "All $TARGET samples generated. Done!"
        break
    fi

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 run_experiments.py --level "$LEVEL" --phase 1 2>&1 | tail -5

    sleep 2  # brief pause between retries
done
