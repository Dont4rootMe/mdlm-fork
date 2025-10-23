#!/bin/bash

# Script to launch TensorBoard for the latest training run
# Usage: bash view_latest_run.sh [experiment_name] [port]

# Default values
OUTPUTS_DIR="/mnt/virtual_ai0001071-01239_SR006-nfs1/afedorov/projects/mdlm-fork/outputs"
DEFAULT_PORT=6006
EXPERIMENT_NAME="${1:-}"
PORT="${2:-$DEFAULT_PORT}"

# Check if outputs directory exists
if [ ! -d "$OUTPUTS_DIR" ]; then
    echo "Error: Outputs directory $OUTPUTS_DIR does not exist!"
    exit 1
fi

# If experiment name is provided, search in that subdirectory
if [ -n "$EXPERIMENT_NAME" ]; then
    SEARCH_DIR="$OUTPUTS_DIR"
    # Find all tensorboard_logs directories in experiment subdirectories
    LATEST_RUN=$(find "$SEARCH_DIR" -type d -path "*/$EXPERIMENT_NAME/*/tensorboard_logs" | head -1)
else
    # Find the most recently modified tensorboard_logs directory
    LATEST_RUN=$(find "$OUTPUTS_DIR" -type d -name "tensorboard_logs" -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
fi

if [ -z "$LATEST_RUN" ]; then
    echo "Error: No TensorBoard logs found!"
    echo ""
    if [ -n "$EXPERIMENT_NAME" ]; then
        echo "Searched for experiment: $EXPERIMENT_NAME"
    fi
    echo "Searched in: $OUTPUTS_DIR"
    echo ""
    echo "Available experiments:"
    find "$OUTPUTS_DIR" -type d -name "tensorboard_logs" | head -5
    exit 1
fi

# Get the parent directory (the run directory)
RUN_DIR=$(dirname "$LATEST_RUN")

echo "================================"
echo "Launching TensorBoard for Latest Run"
echo "================================"
echo "Run directory: $RUN_DIR"
echo "TensorBoard logs: $LATEST_RUN"
echo "Port: $PORT"
echo ""
echo "TensorBoard will be available at:"
echo "  http://localhost:$PORT"
echo ""
echo "If running on a remote server, use SSH tunneling:"
echo "  ssh -L $PORT:localhost:$PORT user@remote_server"
echo ""
echo "Press Ctrl+C to stop TensorBoard"
echo "================================"
echo ""

# Launch TensorBoard
tensorboard --logdir="$LATEST_RUN" --port="$PORT" --bind_all

