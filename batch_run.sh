#!/bin/bash

# ================= 配置区域 =================

# 你的 Python 脚本
ANNOTATOR_SCRIPT="auto_workspace_annotator.py"

# 需要搜索的目录数组
SEARCH_DIRS=(
    "/home/ubuntu/codespace/isaacsim_auto_label/OpenRobotLab___GRScenes/scenes/GRScenes-100/commercial_scenes/scenes"
    "/home/ubuntu/codespace/isaacsim_auto_label/OpenRobotLab___GRScenes/scenes/GRScenes-100/home_scenes/scenes"
)

# 目标文件名
TARGET_FILENAME="start_result_raw.usd"

# ===========================================

# 检查脚本是否存在
if [ ! -f "$ANNOTATOR_SCRIPT" ]; then
    echo "错误: 找不到脚本 $ANNOTATOR_SCRIPT"
    exit 1
fi

echo "正在扫描文件..."
FILES=()

# 遍历目录查找文件
for dir in "${SEARCH_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        # 使用 find 命令查找并添加到数组
        while IFS= read -r file; do
            FILES+=("$file")
        done < <(find "$dir" -name "$TARGET_FILENAME")
    else
        echo "警告: 目录不存在 $dir"
    fi
done

TOTAL=${#FILES[@]}

if [ "$TOTAL" -eq 0 ]; then
    echo "未找到任何文件。"
    exit 0
fi

echo "共找到 $TOTAL 个文件。"
echo "----------------------------------------"

CURRENT=0

# 开始循环处理
for file in "${FILES[@]}"; do
    ((CURRENT++))
    echo "进度 [$CURRENT/$TOTAL] -> 正在处理: $file"
    
    # 执行 Python 脚本
    python "$ANNOTATOR_SCRIPT" "$file"
    
    # 检查执行结果
    if [ $? -ne 0 ]; then
        echo ">>> 错误: 处理该文件时发生异常!"
    else
        echo ">>> 完成."
    fi
    echo "----------------------------------------"
done

echo "所有任务执行完毕。"