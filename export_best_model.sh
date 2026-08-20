#!/bin/bash
# Export a checkpoint to plain PyTorch weights.

echo "Exporting model from best checkpoint..."

# Adjust --checkpoint if your best checkpoint lives elsewhere.
python util/export_model.py \
    --checkpoint lightning_logs/version_0/checkpoints/epoch=9-step=1880.ckpt \
    --output go_model.pt

echo ""
echo "Model exported! You can now start the server with:"
echo "  MODEL_PATH=go_model.pt python backend/server.py"
