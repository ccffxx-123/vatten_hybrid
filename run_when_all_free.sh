#!/bin/bash

# 判断 GPU 空闲的显存阈值 (MB)
FREE_MEM_THRESHOLD=1000
# 检查间隔时间 (秒)
SLEEP_TIME=10
# 目标空闲 GPU 数量
TARGET_FREE_GPUS=4
# 你的目标工作目录
WORK_DIR="/workspace/vatten_hybrid"

echo "开始监控，等待 4 张 GPU 全部空闲..."

while true; do
    # 获取所有卡的已用显存
    # 注意：前提是启动 Docker 时映射了 GPU（例如用了 --gpus all），否则容器内找不到 nvidia-smi
    MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    
    FREE_COUNT=0
    
    for MEM in $MEM_USED; do
        if [ "$MEM" -lt "$FREE_MEM_THRESHOLD" ]; then
            FREE_COUNT=$((FREE_COUNT+1))
        fi
    done
    
    if [ "$FREE_COUNT" -eq "$TARGET_FREE_GPUS" ]; then
        echo -e "\n[$(date)] 4 张 GPU 均已空闲! 准备运行 Benchmark..."
        
        # 强制进入你的工作目录，如果进入失败则报错并退出，防止在错的目录下跑代码
        cd "$WORK_DIR" || { echo "无法进入目录 $WORK_DIR，请检查路径！"; exit 1; }
        
        # 显式指定使用全部 4 张卡
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        
        # 运行你的评测程序
        python scripts/benchmark_e2e_dynamic_trace.py
        
        echo "[$(date)] Benchmark 运行完毕，退出监控。"
        exit 0
    else
        echo -ne "\r[$(date)] 当前有 $FREE_COUNT/4 张卡空闲，继续等待..."
    fi
    
    sleep $SLEEP_TIME
done