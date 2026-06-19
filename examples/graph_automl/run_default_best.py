import json
import os
import subprocess
import sys
from dance.settings import EXAMPLESDIR

method_name = os.environ['db_method_name']

config_path = os.path.join(EXAMPLESDIR, "evolo/benchmarks_config.json")
work_dir = os.path.join(EXAMPLESDIR, f"graph_automl/{method_name}")

benchmarks_key = f"{method_name.split('_')[1].lower()}_benchmarks"

if not os.path.exists(config_path):
    print(f"错误: 找不到配置文件 {config_path}")
    sys.exit(1)

with open(config_path, "r") as f:
    benchmarks = json.load(f).get(benchmarks_key, {})

if not benchmarks:
    print(f"警告: {benchmarks_key} 配置为空或未找到！")

for config_suffix in ["default", "best"]:
    print("=" * 50)
    print(f"开始运行 config_suffix={config_suffix}")
    print("=" * 50)

    for dataset_name, args in benchmarks.items():
        args_with_mode = args + ["--tune_mode", "params", "--config_suffix", config_suffix, "--count", "1"]
        if method_name == "cta_scrgcl":
            command = ["env", "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5", sys.executable, "main.py"] + args_with_mode
        else:
            command = [sys.executable, "main.py"] + args_with_mode

        print("-" * 50)
        print(f"config_suffix={config_suffix}  数据集: {dataset_name}")
        print(f"工作目录: {work_dir}")
        print(f"完整命令: {' '.join(command)}")
        print("-" * 50)

        try:
            subprocess.run(command, check=True, cwd=work_dir)
            print(f"\n>>> [{config_suffix}] 数据集 {dataset_name} 运行完成")

        except subprocess.CalledProcessError as e:
            print(f"\nXXX 运行出错 (config_suffix={config_suffix}, 数据集: {dataset_name})")
            print(f"退出代码: {e.returncode}")
        except FileNotFoundError as e:
            print(e)
            print(f"\nXXX 错误: 在 {work_dir} 下找不到 main.py，请检查路径。")
