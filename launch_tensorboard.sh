#!/bin/bash

# Script to launch TensorBoard for viewing training logs
# Usage: bash launch_tensorboard.sh [logdir] [port]

# Default values
DEFAULT_LOGDIR="/mnt/virtual_ai0001071-01239_SR006-nfs1/afedorov/projects/mdlm-fork/outputs"
DEFAULT_PORT=6006

# Get arguments or use defaults
LOGDIR="${1:-$DEFAULT_LOGDIR}"
PORT="${2:-$DEFAULT_PORT}"

# Check if logdir exists
if [ ! -d "$LOGDIR" ]; then
    echo "Error: Directory $LOGDIR does not exist!"
    echo "Please provide a valid log directory."
    echo ""
    echo "Usage: bash launch_tensorboard.sh [logdir] [port]"
    echo "Example: bash launch_tensorboard.sh /path/to/outputs 6006"
    exit 1
fi

echo "================================"
echo "Launching TensorBoard"
echo "================================"
echo "Log directory: $LOGDIR"
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
tensorboard --logdir="$LOGDIR" --port="$PORT" --bind_all

