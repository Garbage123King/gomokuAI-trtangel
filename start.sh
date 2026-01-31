#!/bin/bash

# 1. 初始化清理
echo "Initializing: Cleaning up old directories..."
rm -rf selfplay model log
mkdir -p log

epoch=0

# 开始无限循环
while true; do
    echo "------------------------------------------------"
    echo "Starting Iteration: Epoch $epoch"
    iter_start_time=$(date +%s)

    # --- Step 1: 运行 Selfplay ---
    echo "Step 1: Running Selfplay..."
    selfplay_start=$(date +%s)
    
    if [ "$epoch" -eq 0 ]; then
        # 第一次循环
        ./trtangel_release selfplay 128 10000 2 "epoch$epoch" > "log/selfplay$epoch.log" 2>&1
    else
        # 以后每一次
        ./trtangel_release selfplay 128 3000 0 "epoch$epoch" > "log/selfplay$epoch.log" 2>&1
    fi
    
    selfplay_end=$(date +%s)
    echo "Selfplay $epoch finished. Time taken: $((selfplay_end - selfplay_start)) seconds."

    # --- Step 2: 运行 Train ---
    echo "Step 2: Running Train..."
    train_start=$(date +%s)
    
    # 按照你的要求，序号与 selfplay 一致
    python train.py > "log/train$epoch.log" 2>&1
    
    train_end=$(date +%s)
    echo "Train $epoch finished. Time taken: $((train_end - train_start)) seconds."

    # --- Step 3: 更新软链接 ---
    # 逻辑：model.pth -> candidate.pth 指向的目标
    echo "Step 3: Updating symbolic links in 'model' directory..."
    cd model || { echo "Error: 'model' directory not found!"; exit 1; }

    # 定义需要链接的文件后缀
    for ext in pth pt onnx; do
        target_link="candidate.$ext"
        main_link="model.$ext"
        
        if [ -L "$target_link" ] || [ -f "$target_link" ]; then
            # 获取 candidate.xxx 指向的真实目标路径
            real_target=$(readlink -f "$target_link")
            # 删除旧的 model.xxx，创建新链接指向该真实目标
            rm -f "$main_link"
            ln -s "$real_target" "$main_link"
            echo "Linked $main_link -> $real_target"
        else
            echo "Warning: $target_link not found, skipping."
        fi
    done
    
    # 返回根目录
    cd ..

    # --- 结束循环处理 ---
    iter_end_time=$(date +%s)
    echo "Epoch $epoch completed in $((iter_end_time - iter_start_time)) seconds."
    
    # 增加计数器
    epoch=$((epoch + 1))
done