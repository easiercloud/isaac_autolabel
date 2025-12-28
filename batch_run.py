import os
import subprocess
import sys
from pathlib import Path

# ================= 配置区域 =================

# 你的处理脚本路径 (假设在同级目录)
ANNOTATOR_SCRIPT = "auto_workspace_annotator.py"

# 需要遍历的根目录列表
SEARCH_DIRS = [
    "/home/ubuntu/codespace/isaacsim_auto_label/OpenRobotLab___GRScenes/scenes/GRScenes-100/commercial_scenes/scenes",
    "/home/ubuntu/codespace/isaacsim_auto_label/OpenRobotLab___GRScenes/scenes/GRScenes-100/home_scenes/scenes"
]

# 目标文件名
TARGET_FILENAME = "start_result_raw.usd"

# ===========================================

def main():
    # 1. 确保处理脚本存在
    if not os.path.exists(ANNOTATOR_SCRIPT):
        print(f"[错误] 找不到处理脚本: {ANNOTATOR_SCRIPT}")
        sys.exit(1)

    print("正在扫描目标文件，请稍候...")

    # 2. 收集所有符合条件的文件路径
    files_to_process = []
    for search_dir in SEARCH_DIRS:
        root_path = Path(search_dir)
        if not root_path.exists():
            print(f"[警告] 目录不存在，跳过: {search_dir}")
            continue
            
        # rglob 用于递归查找
        for found_file in root_path.rglob(TARGET_FILENAME):
            files_to_process.append(str(found_file))

    total_files = len(files_to_process)
    
    if total_files == 0:
        print("[提示] 未找到任何符合条件的文件。")
        sys.exit(0)

    print(f"共找到 {total_files} 个文件待处理。\n")

    # 3. 循环执行
    success_count = 0
    fail_count = 0
    failed_files = []

    for index, file_path in enumerate(files_to_process, 1):
        print("-" * 60)
        print(f"进度 [{index}/{total_files}] | 正在处理: {file_path}")
        print("-" * 60)

        try:
            # 调用 python 命令执行你的脚本
            # check=True 会在脚本返回非0状态码时抛出异常
            subprocess.run(
                ["python", ANNOTATOR_SCRIPT, file_path], 
                check=True
            )
            success_count += 1
            
        except subprocess.CalledProcessError as e:
            print(f"\n[错误] 处理失败: {file_path}")
            print(f"错误信息: {e}\n")
            fail_count += 1
            failed_files.append(file_path)
        except KeyboardInterrupt:
            print("\n[中断] 用户手动停止任务。")
            break

    # 4. 总结
    print("\n" + "=" * 60)
    print(f"任务完成!")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    if failed_files:
        print("失败的文件列表:")
        for f in failed_files:
            print(f" - {f}")
    print("=" * 60)

if __name__ == "__main__":
    main()