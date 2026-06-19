import argparse
import gc
import json
import os
import pprint
import sys
import time
from pathlib import Path
from typing import get_args

import numpy as np
import torch
import wandb
import tempfile
import anndata
from dance import logger
from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.modules.single_modality.cell_type_annotation.scrgcl import scRGCLWrapper
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.typing import LogLevel
from dance.utils import set_seed
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--dropout_ratio", type=float, default=0.1, help='Dropout ratio')
    parser.add_argument("--gpu", type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument("--init_lr", type=float, default=0.001, help='Initial learning rate')
    parser.add_argument("--log_level", type=str, default="INFO", choices=get_args(LogLevel))
    parser.add_argument("--num_runs", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--quantile", type=float, default=0.99, help='Quantile threshold for network filtering')
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--species", default="human", type=str)
    parser.add_argument("--test_dataset", nargs="+", type=int, default=[138], help="Testing dataset IDs")
    parser.add_argument("--tissue", default="Brain", type=str)
    parser.add_argument("--train_dataset", nargs="+", default=[328], help="List of training dataset ids.")
    parser.add_argument("--valid_dataset", nargs="+", default=None, help="List of valid dataset ids.")
    parser.add_argument("--val_size", type=float, default=0.2, help="val size")

    parser.add_argument("--tune_mode", default="pipeline_params", choices=["pipeline", "params", "pipeline_params"])
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--config_suffix", type=str, default="", help="Suffix appended to tune_mode for config file and result file naming (e.g., 'default', 'best').")
    parser.add_argument("--summary_file_path", default="results/pipeline/best_test_acc.csv", type=str)
    parser.add_argument("--root_path", default=str(Path(__file__).resolve().parent), type=str)
    parser.add_argument('--additional_sweep_ids', action='append', type=str, help='get prior runs')
    args = parser.parse_args()
    logger.setLevel(args.log_level)
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "2000"
    logger.info(f"Running scRGCL with the following parameters:\n{pprint.pformat(vars(args))}")
    file_root_path = Path(
        args.root_path, "_".join([
            "-".join([str(num) for num in dataset])
            for dataset in [args.train_dataset, args.valid_dataset, args.test_dataset]
            if (dataset is not None and dataset != [])
        ])).resolve()
    logger.info(f"\n files is saved in {file_root_path}")
    pipeline_planer = PipelinePlaner.from_config_file(f"{Path(args.root_path).resolve()}/{args.tune_mode + '_' + args.config_suffix if args.config_suffix else args.tune_mode}_tuning_config.yaml")
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "2000"

    # ================= MODIFIED FUNCTION STARTS HERE =================
    def evaluate_pipeline(tune_mode=args.tune_mode, pipeline_planer=pipeline_planer):
        wandb.init(settings=wandb.Settings(start_method='thread'))

        train_scores = []
        valid_scores = []
        test_scores = []

        # 用于记录每次 run 的详细信息
        run_details = []

        # Start Timer
        start_time = time.time()

        for run_idx in range(args.num_runs):
            current_seed = args.seed + run_idx
            set_seed(current_seed)
            logger.info(f"Starting Run {run_idx + 1}/{args.num_runs}")
            run_start_time = time.time()
            with tempfile.TemporaryDirectory() as temp_dir:
                device = torch.device("cuda:" + str(args.gpu))
                model = scRGCLWrapper(
                    out_dir=temp_dir,
                    dropout_ratio=args.dropout_ratio,
                    init_lr=args.init_lr,
                    seed=current_seed,
                    device=device,
                )

                # Load data and perform necessary preprocessing
                dataloader = CellTypeAnnotationDataset(train_dataset=args.train_dataset, test_dataset=args.test_dataset,
                                                species=args.species, tissue=args.tissue, val_size=args.val_size,data_dir="../temp_data")
                data = dataloader.load_data(transform=None, cache=args.cache)
                # Prepare preprocessing pipeline and apply it to data
                kwargs = {tune_mode: dict(wandb.config)}
                preprocessing_pipeline = pipeline_planer.generate(**kwargs)
                if run_idx == 0:
                    print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")
                preprocessing_pipeline(data)

                # Extract Train/Test Data
                x_train, y_train = data.get_train_data(return_type="torch")
                x_val, y_val = data.get_val_data(return_type="torch")
                x_test, y_test = data.get_test_data(return_type="torch")

                # Convert Labels (One-hot -> Index)
                y_train_indices = y_train.argmax(1).cpu().numpy()
                
                train_adata = anndata.AnnData(X=x_train.cpu().numpy())
                train_adata.uns=data.data.uns
                train_adata.obs['cell_type'] = y_train_indices
                if hasattr(data.data, "var_names"):
                    train_adata.var_names = data.data.var_names

                # Initialize model
                

                # Construct AnnData for Training
                
                # Train the model
                logger.info("Training scRGCL model...")
                model.fit(adata=train_adata, batch_size=args.batch_size)

                # Evaluate the model
                logger.info("Evaluating...")

                # Calculate scores
                run_train_score = model.score(x_train, y_train, score_func="acc")  # using train as proxy
                run_valid_score = model.score(x_val, y_val, score_func="acc")
                run_test_score = model.score(x_test, y_test, score_func="acc")

                train_scores.append(run_train_score)
                valid_scores.append(run_valid_score)
                test_scores.append(run_test_score)

                run_time_seconds = time.time() - run_start_time
                # 记录单次 run 的数据
                run_details.append({
                    "run_idx": run_idx + 1,
                    "seed": current_seed,
                    "train_acc": float(run_train_score),
                    "valid_acc": float(run_valid_score),
                    "test_acc": float(run_test_score),
                    "time_seconds": float(run_time_seconds)
                })

                logger.info(f"Run {run_idx + 1} finished. Valid Acc: {run_valid_score:.4f}, Test Acc: {run_test_score:.4f}")

            

        # Stop Timer
        total_time_seconds = time.time() - start_time

        avg_train_score = np.mean(train_scores)
        avg_valid_score = np.mean(valid_scores)
        avg_test_score = np.mean(test_scores)

        # Calculate Speed Score and Combined Score
        speed_score = 1.0 / (1.0 + total_time_seconds / 300.0)
        combined_score = 0.8 * avg_valid_score + 0.2 * speed_score

        logger.info(f"Averaged over {args.num_runs} runs - Valid Acc: {avg_valid_score:.4f}, Time: {total_time_seconds:.2f}s, Combined Score: {combined_score:.4f}")

        # ================= JSON 记录逻辑开始 =================
        train_ds_str = "_".join(map(str, args.train_dataset))
        test_ds_str = "_".join(map(str, args.test_dataset))
        dataset_key = f"{args.species}_{args.tissue}_train_{train_ds_str}_test_{test_ds_str}"

        experiment_summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hyperparameters": dict(wandb.config) if wandb.config else {},
            "runs": run_details,
            "avg_train_acc": float(avg_train_score),
            "avg_valid_acc": float(avg_valid_score),
            "avg_test_acc": float(avg_test_score),
            "total_time_seconds": float(total_time_seconds),
            "speed_score": float(speed_score),
            "combined_score": float(combined_score)
        }

        json_log_path = Path(args.root_path) / f"experiment_results_{tune_mode}{'_' + args.config_suffix if args.config_suffix else ''}.json"

        if json_log_path.exists():
            try:
                with open(json_log_path, "r", encoding="utf-8") as f:
                    all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = {}
        else:
            all_results = {}

        if dataset_key not in all_results:
            all_results[dataset_key] = []
        all_results[dataset_key].append(experiment_summary)

        with open(json_log_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        # ================= JSON 记录逻辑结束 =================

        wandb.log({
            "train_acc": avg_train_score,
            "acc": avg_valid_score,
            "test_acc": avg_test_score,
            "time": total_time_seconds,
            "speed_score": speed_score,
            "combined_score": combined_score
        })
        wandb.finish()
    # ================= MODIFIED FUNCTION ENDS HERE =================

    entity, project, sweep_id = pipeline_planer.wandb_sweep_agent(
        evaluate_pipeline, sweep_id=args.sweep_id, count=args.count)  # Score can be recorded for each epoch
    save_summary_data(entity, project, sweep_id, summary_file_path=args.summary_file_path, root_path=file_root_path,
                      additional_sweep_ids=args.additional_sweep_ids)
    if args.tune_mode == "pipeline" or args.tune_mode == "pipeline_params":
        get_step3_yaml(
            result_load_path=f"{args.summary_file_path}",
            step2_pipeline_planer=pipeline_planer,
            conf_load_path=f"{Path(args.root_path).resolve().parent}/step3_default_params.yaml",
            root_path=file_root_path,
            required_funs=["CellFeatureGraph", "SetConfig"],
            required_indexes=[sys.maxsize - 1, sys.maxsize],
        )
        if args.tune_mode == "pipeline_params":
            run_step3(file_root_path, evaluate_pipeline, tune_mode="params", step2_pipeline_planer=pipeline_planer)
"""To reproduce the benchmarking results, please run the following command:

Human Brain
$ python main.py --species human --tissue Brain --train_dataset 328 --test_dataset 138

Human Lung
$ python main.py --species human --tissue Lung --train_dataset 645 --test_dataset 646

"""