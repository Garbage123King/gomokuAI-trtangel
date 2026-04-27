#!/bin/bash

# 检查目录是否存在
MODEL_DIR="model"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Directory '$MODEL_DIR' not found."
    exit 1
fi

echo "------------------------------------------------"
echo "  Starting Batch Evaluation (Sorted by Time)   "
echo "------------------------------------------------"

# 1. 按时间顺序（从旧到新）排列 .onnx 文件
# ls -tr: t 按时间排序, r 逆序(从旧到新)
models=$(ls -tr $MODEL_DIR/*.onnx 2>/dev/null)

if [ -z "$models" ]; then
    echo "No .onnx files found in $MODEL_DIR."
    exit 1
fi
for model_path in $models; do
    filename=$(basename "$model_path")

    # 加上 -v 参数，并直接输出结果而不只是 grep Final Score
    # 这样可以看到每个文件的详细统计
    echo "Evaluating Model: $filename"
    python texam.py "$model_path" -v 2>/dev/null
    echo "------------------------------------------------"
done

echo "------------------------------------------------"
echo "  Evaluation Complete."
