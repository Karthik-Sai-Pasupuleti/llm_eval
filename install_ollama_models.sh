#!/bin/bash

# Script to run each Ollama model once and automatically exit with /bye
# Useful if you want to both install and test the model startup process.

set -e

models=(
  "llama3.1:8b"
  "phi3:3.8b"
  "qwen2.5:7b"
  "qwen2.5:14b"
  "deepseek-r1:8b"
  "deepseek-r1:14b"
  "phi4-mini:3.8b"
  "phi4:14b"
  "gemma2:9b"
  "mistral:7b"
  "mistral-small:24b"
  "internlm/internlm3-8b-instruct"
)

echo "Starting Ollama model installation with auto-exit..."
echo "==================================================="

for model in "${models[@]}"; do
  echo "Running model: $model (will exit automatically)"
  ollama run "$model" <<EOF
/bye
EOF

  if [ $? -eq 0 ]; then
    echo "Successfully ran and exited: $model"
  else
    echo "Error during run: $model"
  fi
  echo "---------------------------------------------"
done

echo "All models installed and exited successfully."
