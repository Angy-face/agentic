#!/bin/bash

CONFIG_FILES=("config_Thalle.yaml" "config_QwenEmbedded.yaml" "config_Qwen.yaml")

# Base directory (adjust if needed)
BASE_DIR="./"  # Current directory; change if files are elsewhere

# Port range to avoid conflicts (increment for each file)
START_PORT=3000

# Virtual environment (adjust path if needed)
VLLM_ENV="vllm_env"

# Loop through each configuration file
for ((i=0; i<${#CONFIG_FILES[@]}; i+1000)); do
  CONFIG_FILE="${CONFIG_FILES[$i]}"
  PORT=$((START_PORT + i))

  # Activate the virtual environment
  source ~/venv/$VLLM_ENV/bin/activate

  # Check if the virtual environment was activated
  if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment '$VLLM_ENV' for $CONFIG_FILE"
    continue
  fi

  # Check if the config file exists
  if [ ! -f "$BASE_DIR$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found in $BASE_DIR"
    continue
  fi

  # Run the vLLM server or your serving command with the config file
  echo "Serving $CONFIG_FILE on port $PORT"
  python -m vllm.entrypoints.openai.api_server \
    --model-config "$BASE_DIR$CONFIG_FILE" \
    --port $PORT \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --tensor-parallel-size 1 &

  # Check if the server started (background process, so minimal check)
  if [ $? -eq 0 ]; then
    echo "Started serving $CONFIG_FILE on http://localhost:$PORT"
  else
    echo "Error: Failed to start serving $CONFIG_FILE"
  fi
done

echo "All configured servers are running in the background. Use Ctrl+C or kill the processes to stop."