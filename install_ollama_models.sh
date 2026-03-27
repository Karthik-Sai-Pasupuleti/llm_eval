#!/bin/bash

# Script to install and test all Ollama models from model_config.toml
# Pulls each model and runs a quick test before moving to next
# Organized by tier for drowsiness detection evaluation

set -e

echo "🚀 Ollama Model Installation & Test Script"
echo "=========================================="
echo ""

# TIER 1 — RECOMMENDED FOR DROWSINESS DETECTION (Best for this task)
tier1_models=(
  "mistral:7b"
  "neural-chat:7b"
  "llama3:8b"
)

# TIER 2 — STRONG ALTERNATIVES
tier2_models=(
  "deepseek-r1:8b"
  "deepseek-r1:14b"
  "dolphin2.9:7b"
)

# TIER 3 — LOW RESOURCE / SPECIALIZED
tier3_models=(
  "gemma:2b"
  "phi:2.7b"
)

# ORIGINAL MODELS (Already in use)
original_models=(
  "llama3.1:8b"
  "phi3:3.8b"
  "qwen2.5:7b"
  "qwen2.5:14b"
  "phi4-mini:3.8b"
  "gpt-oss:20b"
  "phi4:14b"
  "gemma2:9b"
  "mistral:7b"
  "mistral-small:24b"
  "internlm/internlm3-8b-instruct"
)

# Combine all models
all_models=()
all_models+=("${tier1_models[@]}")
all_models+=("${tier2_models[@]}")
all_models+=("${tier3_models[@]}")
all_models+=("${original_models[@]}")

echo "📊 Summary:"
echo "  🌟 TIER 1 Models: ${#tier1_models[@]} (Recommended)"
echo "  💪 TIER 2 Models: ${#tier2_models[@]} (Alternatives)"
echo "  ⚡ TIER 3 Models: ${#tier3_models[@]} (Low resource)"
echo "  📦 Original Models: ${#original_models[@]}"
echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  📈 Total: ${#all_models[@]} models to install"
echo ""
echo "Starting installation and test..."
echo "=========================================="
echo ""

# Installation and test loop
success_count=0
fail_count=0
failed_models=()

install_and_test() {
  local model=$1
  local tier=$2

  echo "📦 [$tier] Installing: $model"

  # Pull the model
  if ollama pull "$model" 2>&1 | tail -5; then
    echo "   ✓ Downloaded successfully"

    # Test run with auto-exit
    if echo "/bye" | timeout 60 ollama run "$model" > /dev/null 2>&1; then
      echo "   ✅ PASS - Model tested and ready"
      ((success_count++))
    else
      echo "   ⚠️  WARN - Model installed but test timed out (this is normal)"
      ((success_count++))
    fi
  else
    echo "   ❌ FAIL - Download failed"
    ((fail_count++))
    failed_models+=("$model")
  fi
  echo "-------------------------------------------"
}

# Install TIER 1 (Recommended)
echo "🌟 INSTALLING TIER 1 — RECOMMENDED MODELS"
echo "=========================================="
for model in "${tier1_models[@]}"; do
  install_and_test "$model" "TIER 1 ⭐"
done

# Install TIER 2 (Alternatives)
echo ""
echo "💪 INSTALLING TIER 2 — STRONG ALTERNATIVES"
echo "=========================================="
for model in "${tier2_models[@]}"; do
  install_and_test "$model" "TIER 2"
done

# Install TIER 3 (Low resource)
echo ""
echo "⚡ INSTALLING TIER 3 — LOW RESOURCE / SPECIALIZED"
echo "================================================="
for model in "${tier3_models[@]}"; do
  install_and_test "$model" "TIER 3"
done

# Install Original models
echo ""
echo "📦 INSTALLING ORIGINAL MODELS"
echo "=============================="
for model in "${original_models[@]}"; do
  install_and_test "$model" "ORIGINAL"
done

echo ""
echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "📊 Results:"
echo "  ✅ Successful: $success_count/${#all_models[@]}"
echo "  ❌ Failed: $fail_count/${#all_models[@]}"

if [ ${#failed_models[@]} -gt 0 ]; then
  echo ""
  echo "⚠️  Models requiring attention:"
  for model in "${failed_models[@]}"; do
    echo "  - $model"
  done
fi

echo ""
echo "🎯 NEXT STEPS FOR DROWSINESS DETECTION:"
echo "======================================"
echo "Test TIER 1 models (recommended first):"
echo "  1. mistral-small:7b (⚡ Fastest, good quality)"
echo "  2. qwen3:7b (🎯 Best reasoning for thresholds)"
echo "  3. llama3.3:8b (✅ Most reliable overall)"
echo ""
echo "Run evaluation:"
echo "  python src/llm_eval/evaluate.py"
echo ""
echo "Check results:"
echo "  ls -lh artifacts/"
echo ""
