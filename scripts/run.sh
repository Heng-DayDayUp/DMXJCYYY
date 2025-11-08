#!/usr/bin/env bash
set -euo pipefail

# scripts/run.sh
# 自动化脚本：创建 venv、安装依赖、下载数据、运行 baseline 与消融实验、保存 artifacts、生成 PDF 报告
# 适用于 Linux/macOS 或 Windows WSL。Windows 原生可参考下面的 .bat / PowerShell 示例。

ROOT_DIR=$(dirname "$(dirname "$0")")   # assume scripts/ is under repo root
REPO_ROOT=$(realpath "$ROOT_DIR")
DATA_DIR="$REPO_ROOT/data"
RESULTS_DIR="$REPO_ROOT/results"
SCRIPTS_DIR="$REPO_ROOT/scripts"
REPORT_MD="$REPO_ROOT/report.md"
REPORT_PDF="$REPO_ROOT/report.pdf"
VENV_DIR="$REPO_ROOT/venv_run"

# Configuration (changeable)
PYTHON_BIN=python3
PYTORCH_WHEEL=""  # leave empty to let pip choose; if you want a specific wheel, set it here
EPOCHS=6
BATCH_SIZE=32
SEQ_LEN=128
SEED=42

echo "Repository root: $REPO_ROOT"
echo "Data dir: $DATA_DIR"
echo "Results dir: $RESULTS_DIR"
echo "Venv dir: $VENV_DIR"

# 1) Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment at $VENV_DIR ..."
  $PYTHON_BIN -m venv "$VENV_DIR"
fi

# Activate venv for this script
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 2) Install dependencies
echo "Installing Python packages..."
pip install --upgrade pip
# minimal requirements
pip install torch matplotlib requests tqdm

# Optional: install pandoc via system package manager; we attempt to use pandoc if available.
PANDOC_OK=true
if ! command -v pandoc >/dev/null 2>&1; then
  echo "Warning: pandoc not found. report.md -> PDF conversion will be skipped. Install pandoc (and LaTeX) to enable conversion."
  PANDOC_OK=false
fi

# 3) Prepare directories
mkdir -p "$DATA_DIR"
mkdir -p "$RESULTS_DIR"

# 4) Download dataset (tiny shakespeare) via data.py helper
echo "Downloading dataset (tiny Shakespeare) ..."
python - <<PY
from data import download_tiny_shakespeare
download_tiny_shakespeare("$DATA_DIR/tiny_shakespeare.txt")
print("Downloaded to $DATA_DIR/tiny_shakespeare.txt")
PY

# 5) Run baseline seq2seq experiment (auto-encoding mode) with positional + relative bias
BASE_DIR="$RESULTS_DIR/seq2seq_base"
mkdir -p "$BASE_DIR"
echo "Running baseline seq2seq experiment -> $BASE_DIR"
python train.py \
  --task seq2seq \
  --data "$DATA_DIR/tiny_shakespeare.txt" \
  --seq_len $SEQ_LEN \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --d_model 128 \
  --d_ff 512 \
  --heads 4 \
  --layers 2 \
  --use_pos_encoding \
  --relative_pos \
  --save "$BASE_DIR" \
  --seed $SEED

# 6) Ablation experiments
# A) No positional encoding
NO_POS_DIR="$RESULTS_DIR/no_pos"
mkdir -p "$NO_POS_DIR"
echo "Running ablation: no positional encoding -> $NO_POS_DIR"
python train.py \
  --task seq2seq \
  --data "$DATA_DIR/tiny_shakespeare.txt" \
  --seq_len $SEQ_LEN \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --d_model 128 \
  --d_ff 512 \
  --heads 4 \
  --layers 2 \
  --save "$NO_POS_DIR" \
  --seed $SEED

# B) Single-head
ONE_HEAD_DIR="$RESULTS_DIR/one_head"
mkdir -p "$ONE_HEAD_DIR"
echo "Running ablation: single-head -> $ONE_HEAD_DIR"
python train.py \
  --task seq2seq \
  --data "$DATA_DIR/tiny_shakespeare.txt" \
  --seq_len $SEQ_LEN \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --d_model 128 \
  --d_ff 512 \
  --heads 1 \
  --layers 2 \
  --use_pos_encoding \
  --save "$ONE_HEAD_DIR" \
  --seed $SEED

# C) No residual connections
NO_RES_DIR="$RESULTS_DIR/no_residual"
mkdir -p "$NO_RES_DIR"
echo "Running ablation: no residual -> $NO_RES_DIR"
python train.py \
  --task seq2seq \
  --data "$DATA_DIR/tiny_shakespeare.txt" \
  --seq_len $SEQ_LEN \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --d_model 128 \
  --d_ff 512 \
  --heads 4 \
  --layers 2 \
  --use_pos_encoding \
  --no_residual \
  --save "$NO_RES_DIR" \
  --seed $SEED

echo "All experiments finished. Artifacts saved under $RESULTS_DIR"

# 7) Convert report.md -> report.pdf (optional, requires pandoc + LaTeX)
if [ "$PANDOC_OK" = true ]; then
  echo "Converting report.md -> report.pdf using pandoc..."
  # use PDF engine pdflatex; assumes LaTeX installed (texlive / miktex)
  pandoc "$REPORT_MD" -o "$REPORT_PDF" --pdf-engine=pdflatex --toc --listings
  echo "Report generated at $REPORT_PDF"
else
  echo "Skipping report PDF conversion (pandoc not found). report.md remains in repo root."
fi

echo "Run script completed."
