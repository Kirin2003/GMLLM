#!/bin/bash

# 重复运行 continual_learning_memory.py 多次
# 每次保存结果到不同文件，避免覆盖
#
# 用法: ./run_multiple_times.sh <model> [iterations]
# 示例:
#   ./run_multiple_times.sh deepseek        # 使用 deepseek 模型，运行 5 次
#   ./run_multiple_times.sh llama2 3        # 使用 llama2 模型，运行 3 次

set -e

# 解析命令行参数
if [ $# -lt 1 ]; then
    echo "用法: $0 <model> [iterations]"
    echo "示例:"
    echo "  $0 deepseek        # 使用 deepseek 模型，运行 5 次"
    echo "  $0 llama2 3        # 使用 llama2 模型，运行 3 次"
    exit 1
fi

MODEL=$1
ITERATIONS=${2:-5}  # 默认运行 5 次

# 根据模型组装配置文件路径；也支持直接传入配置文件路径
if [ -f "$MODEL" ]; then
    CONFIG_FILE="$MODEL"
    MODEL=$(basename "$MODEL" .yaml)
else
    CONFIG_FILE="./configs/profiles/${MODEL}.yaml"
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 $CONFIG_FILE 不存在"
    echo "可用的配置文件:"
    ls -1 ./configs/profiles/*.yaml 2>/dev/null || echo "  (无)"
    exit 1
fi

# 结果文件的基础名称（不含扩展名）
RESULT_BASE="CL_unk_test_than_train"

cd /Data2/hxq/GMLLM/GMLLM

echo "=========================================="
echo "配置信息"
echo "=========================================="
echo "模型: $MODEL"
echo "配置文件: $CONFIG_FILE"
echo "运行次数: $ITERATIONS"
echo "=========================================="
echo ""

for i in $(seq 1 $ITERATIONS)
do
    echo "=========================================="
    echo "Running iteration $i/$ITERATIONS"
    echo "=========================================="

    # 运行训练 (-u 禁用Python输出缓冲，确保实时看到log输出)
    # 每次使用不同的seed确保随机性
    CUDA_VISIBLE_DEVICES=2 python -u continual_learning_memory.py --config "$CONFIG_FILE" --seed $i 2>&1 | tee run_${MODEL}_$i.log

    # 重命名结果文件（如果存在）
    for result_type in "future_month" "seen_month"; do
        src_file="../results/${RESULT_BASE}_${result_type}_${MODEL}.json"
        dst_file="../results/${RESULT_BASE}_${result_type}_${MODEL}_run$i.json"
        if [ -f "$src_file" ]; then
            mv "$src_file" "$dst_file"
            echo "Saved: ${RESULT_BASE}_${result_type}_${MODEL}_run$i.json"
        fi
    done

    # 从日志中提取 Future month average 结果
    echo ""
    echo "Future month results for run $i:"
    grep "Future month average:" run_${MODEL}_$i.log || echo "Result not found in log"
    echo ""

    echo "=========================================="
    echo "Iteration $i completed"
    echo "=========================================="
    echo ""
done

echo "All $ITERATIONS iterations completed!"
echo ""
echo "Summary of Future month results:"
echo "--------------------------------"
for i in $(seq 1 $ITERATIONS)
do
    echo "Run $i:"
    grep "Future month average:" run_${MODEL}_$i.log 2>/dev/null || echo "  No result found"
done
