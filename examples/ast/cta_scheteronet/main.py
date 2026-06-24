import argparse
import ast
import random
import traceback
import time
import os  # 新增：用于检查文件是否存在
from typing import Optional

import dgl
import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pandas as pd
from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.modules.single_modality.cell_type_annotation.scheteronet import (
    convert_dgl_to_original_format,
    eval_acc,
    print_statistics,
    scHeteroNet,
    set_graph_split,
    set_split,
)
from dance.registry import register_preprocessor
from dance.transforms.base import BaseTransform
from dance.transforms.filter import FilterCellsScanpy, FilterCellsType, HighlyVariableGenesLogarithmizedByTopGenes
from dance.transforms.interface import AnnDataTransform
from dance.transforms.misc import Compose, SaveRaw, SetConfig
from dance.transforms.normalize import Log1P, NormalizeTotal, UpdateSizeFactors
from dance.typing import LogLevel
from dance.utils import set_seed, sub_data

# ==========================================
# AST 随机突变器定义
# ==========================================
class RandomGraphMutator(ast.NodeTransformer):
    def __init__(self, mutation_rate=0.5):
        self.mutation_rate = mutation_rate
        self.distance_metrics = ['l2', 'euclidean', 'cosine', 'manhattan', 'chebyshev']
        self.knn_candidates = [3, 5, 10, 15, 20, 30, 50]

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value in self.distance_metrics:
            if random.random() < self.mutation_rate:
                new_metric = random.choice(self.distance_metrics)
                return ast.Constant(value=new_metric)
        
        # 限制范围 >= 3，避免突变索引
        if isinstance(node.value, int) and 3 <= node.value <= 100:
            if random.random() < self.mutation_rate:
                new_k = random.choice(self.knn_candidates)
                return ast.Constant(value=new_k)
                
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'kneighbors' and random.random() < self.mutation_rate:
                node.func.attr = random.choice(['kneighbors', 'radius_neighbors'])
        return node

# 原始的基准代码块
EVOLVE_BLOCK_CODE = """
@register_preprocessor("graph", "cell",overwrite=True)
class HeteronetGraph(BaseTransform):
    def __init__(self, knn_num: int = 5, distance_metrics: str = 'l2', random_state: int = 0,
                 channel: Optional[str] = None, channel_type: Optional[str] = "X", ignore_first: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.knn_num = knn_num
        self.distance_metrics = distance_metrics
        self.random_state = random_state
        self.channel = channel
        self.ignore_first = ignore_first
        self.channel_type = channel_type
        
    def build_graph(self, features_np, radius=None, knears=None, distance_metrics='l2'):
        coor = pd.DataFrame(features_np)
        if (radius):
            nbrs = NearestNeighbors(radius=radius, metric=distance_metrics).fit(coor)
            _, indices = nbrs.radius_neighbors(coor, return_distance=True)
        else:
            nbrs = NearestNeighbors(n_neighbors=knears + 1, metric=distance_metrics).fit(coor)
            _, indices = nbrs.kneighbors(coor)

        edge_list = np.array([[i, j] for i, sublist in enumerate(indices) for j in sublist])
        return edge_list

    def __call__(self, data):
        adata = data.data
        features_np = data.get_feature(return_type="numpy", channel=self.channel, channel_type=self.channel_type)
        features = torch.as_tensor(features_np, dtype=torch.float32)
        num_nodes = features.shape[0]

        labels_np = np.argmax(adata.obsm['cell_type'].copy(), axis=1)
        labels = torch.as_tensor(labels_np, dtype=torch.long)

        batchs = adata.obs.get('batch_id', None)

        if self.ignore_first:
            labels[labels == 0] = -1

        edge_list_np = self.build_graph(features_np, knears=self.knn_num, distance_metrics=self.distance_metrics)
        if edge_list_np.shape[0] == 0:
            src = torch.tensor([], dtype=torch.long)
            dst = torch.tensor([], dtype=torch.long)
        else:
            edge_list_tensor = torch.tensor(edge_list_np.T, dtype=torch.long)
            src, dst = edge_list_tensor[0], edge_list_tensor[1]

        g = dgl.graph((src, dst), num_nodes=num_nodes)

        g.ndata['feat'] = features
        g.ndata['label'] = labels
        if batchs is not None:
            g.ndata['batch_id'] = torch.from_numpy(batchs.values.astype(int)).long()
        adata.uns[self.out] = g
"""

def mutate_and_inject(mutation_rate=0.4):
    """动态突变代码并注入到全局环境"""
    tree = ast.parse(EVOLVE_BLOCK_CODE)
    mutator = RandomGraphMutator(mutation_rate=mutation_rate)
    mutated_tree = mutator.visit(tree)
    ast.fix_missing_locations(mutated_tree)
    mutated_code = ast.unparse(mutated_tree)
    # 注入到全局命名空间，覆盖旧的 HeteronetGraph 定义
    exec(mutated_code, globals())
    return mutated_code


def get_preprocessing_pipeline(log_level: LogLevel = "INFO"):
    transforms = []
    transforms.append(FilterCellsType())
    transforms.append(AnnDataTransform(sc.pp.filter_genes, min_counts=3))
    transforms.append(FilterCellsScanpy(min_counts=1))
    transforms.append(HighlyVariableGenesLogarithmizedByTopGenes(n_top_genes=4000, flavor="cell_ranger"))
    transforms.append(SaveRaw())
    transforms.append(NormalizeTotal())
    transforms.append(UpdateSizeFactors())
    transforms.append(Log1P())
    # 这里的 HeteronetGraph 会使用当前全局环境中最新突变后的版本
    transforms.append(HeteronetGraph())
    transforms.append(SetConfig({"label_channel": "cell_type"}))
    return Compose(*transforms, log_level=log_level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_mutations", type=int, default=20, help="Number of mutation trials to search for best pipeline")
    parser.add_argument("--mutation_rate", type=float, default=0.4, help="Probability of AST node mutation")
    
    parser.add_argument("--test_dataset", nargs="+", type=int, default=[1759], help="Testing dataset IDs")
    parser.add_argument("--tissue", default="Spleen", type=str)
    parser.add_argument("--train_dataset", nargs="+", type=int, default=[1970], help="List of training dataset ids.")
    parser.add_argument("--val_size", type=float, default=0.2, help="val size")
    parser.add_argument("--species", default="mouse", type=str)
    parser.add_argument('--data_dir', type=str, default='../temp_data')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_step', type=int, default=1, help='how often to print')
    parser.add_argument("--num_runs", type=int, default=3, help="Number of repetitions per mutation")
    parser.add_argument('--train_prop', type=float, default=.6, help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.2, help='validation label proportion')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc', 'f1'], help='evaluation metric')
    parser.add_argument('--knn_num', type=int, default=5, help='number of k for KNN graph')
    parser.add_argument('--T', type=float, default=1.0, help='temperature for Softmax')
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers for deep methods')
    parser.add_argument('--num_mlp_layers', type=int, default=1, help='number of mlp layers')
    parser.add_argument('--use_bn', action='store_true', help='use layernorm')
    parser.add_argument('--m_in', type=float, default=-5, help='upper bound for in-distribution energy')
    parser.add_argument('--m_out', type=float, default=-1, help='lower bound for in-distribution energy')
    parser.add_argument('--use_prop', action='store_true', help='whether to use energy belief propagation')
    parser.add_argument('--oodprop', type=int, default=2, help='number of layers for energy belief propagation')
    parser.add_argument('--oodalpha', type=float, default=0.3, help='weight for residual connection in propagation')
    parser.add_argument('--use_zinb', action='store_true', help='whether to use ZINB loss')
    parser.add_argument('--use_2hop', action='store_false', help='whether to use 2-hop propagation')
    parser.add_argument('--zinb_weight', type=float, default=1e-4)
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument('--display_step', type=int, default=10, help='how often to print')
    parser.add_argument('--print_prop', action='store_true', help='print proportions of predicted class')
    parser.add_argument('--print_args', action='store_true', help='print args for hyper-parameter searching')
    parser.add_argument('--cl_weight', type=float, default=0.0)
    parser.add_argument('--mask_ratio', type=float, default=0.8)
    parser.add_argument('--spatial', action='store_false', help='read spatial')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--obs_nums",type=int,default=None)
    
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}") if args.gpu != -1 and torch.cuda.is_available() else torch.device("cpu")
    
    best_mutation_score = -1.0
    best_mutation_code = ""
    
    # 新增：用于记录所有突变尝试的细粒度历史记录
    search_history = []

    # ==========================================
    # 外层循环：进行多次突变搜索
    # ==========================================
    for mutation_idx in range(args.num_mutations):
        print(f"\n{'='*50}")
        print(f"Starting Mutation Trial {mutation_idx + 1}/{args.num_mutations}")
        print(f"{'='*50}")
        
        # 1. 突变并注入代码
        current_mutated_code = mutate_and_inject(mutation_rate=args.mutation_rate)
        
        results = []
        inner_scores = []
        
        # 2. 尝试运行当前突变版本，加入 try-except 捕获异常
        for run in range(args.num_runs):
            current_seed = args.seed + run
            ref_data_name = f"{args.species}_{args.tissue}_{args.train_dataset}"
            
            run_start_time = time.time()
            try:
                set_seed(current_seed)
                dataloader = CellTypeAnnotationDataset(species=args.species, tissue=args.tissue, test_dataset=args.test_dataset,
                                                       train_dataset=args.train_dataset, data_dir=args.data_dir,
                                                       val_size=args.val_size)
                
                preprocessing_pipeline = get_preprocessing_pipeline()
                data = dataloader.load_data(transform=None, cache=args.cache)
                
                if args.obs_nums is not None:
                    sub_data(data.data, args.obs_nums)
                    train_idx, test_idx = train_test_split(range(args.obs_nums), test_size=0.2, random_state=args.seed)
                    train_idx, val_idx = train_test_split(train_idx, test_size=args.val_size, random_state=args.seed)
                    data.set_split_idx("train", train_idx)
                    data.set_split_idx("test", test_idx)
                    data.set_split_idx("val", val_idx)
                    
                preprocessing_pipeline(data)
                set_split(data, data.train_idx, data.val_idx, data.test_idx)
                
                g = data.data.uns['HeteronetGraph']
                dataset_ind, dataset_ood_tr, dataset_ood_te, adata = convert_dgl_to_original_format(g, data.data, ref_data_name)
                
                if len(dataset_ind.y.shape) == 1:
                    dataset_ind.y = dataset_ind.y.unsqueeze(1)
                if len(dataset_ood_tr.y.shape) == 1:
                    dataset_ood_tr.y = dataset_ood_tr.y.unsqueeze(1)
                if isinstance(dataset_ood_te, list):
                    for single_dataset_ood_te in dataset_ood_te:
                        if len(single_dataset_ood_te.y.shape) == 1:
                            single_dataset_ood_te.y = single_dataset_ood_te.y.unsqueeze(1)
                else:
                    if len(dataset_ood_te.y.shape) == 1:
                        dataset_ood_te.y = dataset_ood_te.y.unsqueeze(1)

                c = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])
                d = dataset_ind.graph['node_feat'].shape[1]
                model = scHeteroNet(d, c, dataset_ind.edge_index.to(device), dataset_ind.num_nodes,
                                    hidden_channels=args.hidden_channels, num_layers=args.num_layers, dropout=args.dropout,
                                    use_bn=args.use_bn, device=device, min_loss=100000)
                criterion = nn.NLLLoss()
                model.train()
                model.reset_parameters()
                model.to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                for epoch in range(args.epochs):
                    loss = model.fit(dataset_ind, dataset_ood_tr, args.use_zinb, adata, args.zinb_weight, args.cl_weight,
                                     args.mask_ratio, criterion, optimizer)
                    
                test_score = model.score(dataset_ind, dataset_ind.y, data.test_idx)
                inner_score = model.score(dataset_ind, dataset_ind.y, data.train_idx)
                
                run_end_time = time.time()
                run_time = run_end_time - run_start_time
                
                results.append(test_score)
                inner_scores.append(inner_score)
                
                # 记录单次成功的 run
                search_history.append({
                    "mutation_idx": mutation_idx + 1,
                    "dataset": ref_data_name,
                    "run_idx": run,
                    "seed": current_seed,
                    "status": "Success",
                    "test_score": test_score,
                    "inner_score": inner_score,
                    "time_seconds": run_time,
                    "error_message": None,
                    "code": current_mutated_code
                })
                
            except Exception as e:
                run_end_time = time.time()
                run_time = run_end_time - run_start_time
                
                # 捕获异常，防止程序崩溃，并记录失败的 run
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"Mutation {mutation_idx + 1} Run {run} Failed! Error: {error_msg}")
                
                search_history.append({
                    "mutation_idx": mutation_idx + 1,
                    "dataset": ref_data_name,
                    "run_idx": run,
                    "seed": current_seed,
                    "status": "Failed",
                    "test_score": None,
                    "inner_score": None,
                    "time_seconds": run_time,
                    "error_message": error_msg,
                    "code": current_mutated_code
                })
                # 如果当前突变在某一次 run 失败了，通常意味着代码逻辑有致命错误（如 axis=50），
                # 可以选择 break 跳过当前突变的剩余 run，节省时间。
                break 

        # 3. 记录最优突变 (仅当有成功的 run 时)
        if len(results) > 0:
            mean_test_score = np.mean(results)
            mean_inner_score = np.mean(inner_scores)
            print(f"Mutation {mutation_idx + 1} Results (Successful runs: {len(results)}/{args.num_runs}):")
            print(f"mean_score: {mean_test_score:.5f} +/- {np.std(results):.5f}")
            print(f"mean_inner_score: {mean_inner_score:.5f} +/- {np.std(inner_scores):.5f}")
            
            if mean_test_score > best_mutation_score:
                best_mutation_score = mean_test_score
                best_mutation_code = current_mutated_code
                print(f"--> New Best Mutation Found! Mean Score: {best_mutation_score:.5f}")

    # ==========================================
    # 搜索结束，输出汇总信息
    # ==========================================
    print(f"\n{'='*50}")
    print(f"Search Completed. Best Score: {best_mutation_score:.5f}")
    
    # 将历史记录转换为 DataFrame 并打印概览
    history_df = pd.DataFrame(search_history)
    print("\nMutation Search History Summary (Per Run):")
    # 打印前几列核心信息展示
    print(history_df[['mutation_idx', 'dataset', 'run_idx', 'status', 'test_score', 'inner_score', 'time_seconds', 'error_message']].to_string(index=False))
    
    # ==========================================
    # 追加保存到 CSV 文件
    # ==========================================
    csv_filename = "mutation_search_history_raw.csv"
    # 检查文件是否存在，如果不存在则写入表头，如果存在则追加且不写表头
    file_exists = os.path.exists(csv_filename)
    history_df.to_csv(csv_filename, mode='a', index=False, header=not file_exists)
    
    print(f"\nDetailed history appended to '{csv_filename}'")

    print("\nBest Mutated Code:")
    print(best_mutation_code)