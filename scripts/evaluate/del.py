import os
import shutil

def cleanup_experiment_dirs(target_path, min_file_count=10):
    # 转换为绝对路径防止出错
    target_path = os.path.abspath(target_path)
    
    if not os.path.exists(target_path):
        print(f"路径不存在: {target_path}")
        return

    # 遍历目标路径下的第一层子文件夹
    # 我们只删除 e2e_static_eval_page 下的直接子目录
    for folder_name in os.listdir(target_path):
        folder_path = os.path.join(target_path, folder_name)
        
        # 只处理文件夹
        if os.path.isdir(folder_path):
            file_counter = 0
            
            # 递归统计该文件夹内所有层级的文件数量
            for root, dirs, files in os.walk(folder_path):
                file_counter += len(files)
            
            print(f"文件夹: {folder_name} 总文件数: {file_counter}")
            # 判断逻辑
            if file_counter < min_file_count:
                print(f"正在删除: {folder_name} (文件数: {file_counter})")
                try:
                    shutil.rmtree(folder_path)
                except Exception as e:
                    print(f"删除 {folder_name} 失败: {e}")
            else:
                print(f"保留: {folder_name} (文件数: {file_counter})")

if __name__ == "__main__":
    # 请确保路径正确
    path = "../../experiments"
    for p in os.listdir(path):
        p = os.path.join(path, p)
        # print(p)
        cleanup_experiment_dirs(p)
    