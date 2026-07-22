#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/jpeg_encoder_${TIMESTAMP}"
mkdir -p ${OUTPUT_DIR}

echo "============================================================"
echo "Starting Dual-View Pipeline"
echo "============================================================"
echo "Output directory: ${OUTPUT_DIR}"
echo "Started at: $(date)"
echo ""

# All parameters except --design, --nodes, --edges, and --output are in config.py

python -u main_pipeline.py \
    --design ${DESIGN} \
    --nodes ${DESIGN}_merged_nodes.csv \
    --edges ${DESIGN}_merged_edges.csv \
    --output ${OUTPUT_DIR} \
    2>&1 | tee ${OUTPUT_DIR}/training_log.txt

