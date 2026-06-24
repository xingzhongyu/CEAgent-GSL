import argparse
import ast
import random
import traceback
import time
import os
import pprint
from typing import Optional, Union, get_args

import dgl
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split
import torch
import pandas as pd

from dance import logger
from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.modules.single_modality.cell_type_annotation.scdeepsort import ScDeepSort
from dance.registry import register_preprocessor
from dance.transforms.base import BaseTransform
from dance.transforms.misc import Compose, SetConfig
from dance.typing import LogLevel
from dance.utils import set_seed, sub_data
from dance.utils.matrix import normalize
from dance.utils.wrappers import add_mod_and_transform

# ==========================================
# 升级版 AST 结构突变器定义
# ==========================================
class StructuralMutator(ast.NodeTransformer):
    def __init__(self, mutation_rate=0.5):
        self.mutation_rate = mutation_rate
        self.n_components_candidates = [200, 400, 600]
        self.norm_modes = ['normalize', 'l2']
        self.norm_axes = [0, 1]

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value in self.norm_modes:
            if random.random() < self.mutation_rate:
                return ast.Constant(value=random.choice(self.norm_modes))

        if isinstance(node.value, int) and not isinstance(node.value, bool):
            if 100 <= node.value <= 600:
                if random.random() < self.mutation_rate:
                    return ast.Constant(value=random.choice(self.n_components_candidates))
            
            # 删除原先在这里的 self.norm_axes 突变判断
            # 避免误伤如 src.shape[0] 中的 0

        return node

    def visit_keyword(self, node):
        self.generic_visit(node)
        
        # 突变 bool 参数
        if node.arg == 'normalize_edges' and isinstance(node.value, ast.Constant):
            if random.random() < self.mutation_rate:
                node.value = ast.Constant(value=not node.value.value)
                
        # 安全地仅针对关键字参数为 'axis' 或 'feat_norm_axis' 的值进行突变
        if node.arg in ['axis', 'feat_norm_axis'] and isinstance(node.value, ast.Constant):
            if node.value.value in self.norm_axes:
                if random.random() < self.mutation_rate:
                    node.value = ast.Constant(value=random.choice(self.norm_axes))
                    
        return node

    # 2. 结构突变：替换函数调用 (例如将 PCA 换成 TruncatedSVD)
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            if node.func.id == 'PCA' and random.random() < self.mutation_rate:
                # 结构性改变：更换降维算法
                node.func.id = random.choice(['PCA', 'TruncatedSVD'])
        return node

    # 3. 结构突变：修改二元操作 (例如修改矩阵乘法的左操作数，去掉归一化)
    def visit_BinOp(self, node):
        self.generic_visit(node)
        # 针对 cell_feat = normalize(...) @ gene_feat
        if isinstance(node.op, ast.MatMult) and random.random() < self.mutation_rate:
            if isinstance(node.left, ast.Call) and getattr(node.left.func, 'id', '') == 'normalize':
                # 突变：去掉 normalize，直接使用原始变量 x (即 normalize 的第一个参数)
                if len(node.left.args) > 0:
                    node.left = node.left.args[0]
        return node

    # 4. 结构突变：插入/删除计算步骤
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.name == '__call__' and random.random() < self.mutation_rate:
            new_body = []
            for stmt in node.body:
                new_body.append(stmt)
                # 寻找 gene_feat = ... 的赋值语句
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    target = stmt.targets[0]
                    if isinstance(target, ast.Name) and target.id == 'gene_feat':
                        # 突变：在 gene_feat 计算后，额外插入一次归一化操作
                        if random.random() < 0.5:
                            # 构造 AST 节点: gene_feat = normalize(gene_feat, mode="normalize", axis=0)
                            extra_norm_code = 'gene_feat = normalize(gene_feat, mode="normalize", axis=0)'
                            extra_norm_stmt = ast.parse(extra_norm_code).body[0]
                            new_body.append(extra_norm_stmt)
            node.body = new_body
        return node


# 原始的基准代码块 (增加了 TruncatedSVD 的导入)
EVOLVE_BLOCK_CODE = """
from sklearn.decomposition import PCA, TruncatedSVD

@register_preprocessor("feature", "cell", overwrite=True)
@add_mod_and_transform
class WeightedFeaturePCA(BaseTransform):
    _DISPLAY_ATTRS = ("n_components", "split_name", "feat_norm_mode", "feat_norm_axis")

    def __init__(self, n_components: int = 400, split_name=None,
                 feat_norm_mode=None, feat_norm_axis: int = 0, save_info=False, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.split_name = split_name
        self.feat_norm_mode = feat_norm_mode
        self.feat_norm_axis = feat_norm_axis
        self.save_info = save_info

    def __call__(self, data):
        feat = data.get_x(self.split_name)
        if self.feat_norm_mode is not None:
            feat = normalize(feat, mode=self.feat_norm_mode, axis=self.feat_norm_axis)
        if self.n_components > min(feat.shape):
            self.n_components = min(feat.shape)
        gene_pca = PCA(n_components=self.n_components)
        gene_feat = gene_pca.fit_transform(feat.T)
        x = data.get_x()
        cell_feat = normalize(x, mode="normalize", axis=1) @ gene_feat
        data.data.obsm[self.out] = cell_feat.astype(np.float32)
        data.data.varm[self.out] = gene_feat.astype(np.float32)
        return data


@register_preprocessor("graph", "cell", overwrite=True)
class CellFeatureGraph(BaseTransform):

    def __init__(self, cell_feature_channel: str, gene_feature_channel=None, *,
                 mod=None, normalize_edges: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.cell_feature_channel = cell_feature_channel
        self.gene_feature_channel = gene_feature_channel or cell_feature_channel
        self.mod = mod
        self.normalize_edges = normalize_edges

    def __call__(self, data):
        feat = data.get_feature(return_type="default", mod=self.mod)
        num_cells, num_feats = feat.shape

        row, col = np.nonzero(feat)
        edata = np.array(feat[row, col]).ravel()[:, None]

        row = row + num_feats
        col, row = np.hstack((col, row)), np.hstack((row, col))
        edata = np.vstack((edata, edata))

        col = torch.LongTensor(col)
        row = torch.LongTensor(row)
        edata = torch.FloatTensor(edata)

        g = dgl.graph((row, col))
        g.edata["weight"] = edata
        g.ndata["cell_id"] = torch.concat((torch.arange(num_feats, dtype=torch.int32),
                                           -torch.ones(num_cells, dtype=torch.int32)))
        g.ndata["feat_id"] = torch.concat((-torch.ones(num_feats, dtype=torch.int32),
                                           torch.arange(num_cells, dtype=torch.int32)))

        if self.normalize_edges:
            in_deg = g.in_degrees()
            for i in range(g.number_of_nodes()):
                src, dst, eidx = g.in_edges(i, form="all")
                if src.shape[0] > 0:
                    edge_w = g.edata["weight"][eidx]
                    g.edata["weight"][eidx] = in_deg[i] * edge_w / edge_w.sum()
        g.add_edges(g.nodes(), g.nodes(), {"weight": torch.ones(g.number_of_nodes())[:, None]})

        gene_feature = data.get_feature(return_type="torch", channel=self.gene_feature_channel, mod=self.mod,
                                        channel_type="varm")
        cell_feature = data.get_feature(return_type="torch", channel=self.cell_feature_channel, mod=self.mod,
                                        channel_type="obsm")
        g.ndata["features"] = torch.vstack((gene_feature, cell_feature))

        data.data.uns[self.out] = g
        return data


@register_preprocessor("graph", "cell", overwrite=True)
class PCACellFeatureGraph(BaseTransform):

    _DISPLAY_ATTRS = ("n_components", "split_name")

    def __init__(self, n_components: int = 400, split_name=None, *,
                 normalize_edges: bool = True, feat_norm_mode=None,
                 feat_norm_axis: int = 0, mod=None, log_level="WARNING"):
        super().__init__(log_level=log_level)
        self.n_components = n_components
        self.split_name = split_name
        self.normalize_edges = normalize_edges
        self.feat_norm_mode = feat_norm_mode
        self.feat_norm_axis = feat_norm_axis
        self.mod = mod

    def __call__(self, data):
        WeightedFeaturePCA(self.n_components, self.split_name, feat_norm_mode=self.feat_norm_mode,
                           feat_norm_axis=self.feat_norm_axis, log_level=self.log_level)(data)
        CellFeatureGraph(cell_feature_channel="WeightedFeaturePCA", mod=self.mod,
                         normalize_edges=self.normalize_edges, log_level=self.log_level)(data)
        return data
"""


def mutate_and_inject(mutation_rate=0.4):
    """动态突变代码并注入到全局环境"""
    tree = ast.parse(EVOLVE_BLOCK_CODE)
    mutator = StructuralMutator(mutation_rate=mutation_rate)
    mutated_tree = mutator.visit(tree)
    ast.fix_missing_locations(mutated_tree)
    mutated_code = ast.unparse(mutated_tree)
    exec(mutated_code, globals())
    return mutated_code


def get_preprocessing_pipeline(n_components: int = 400, log_level: LogLevel = "INFO"):
    return Compose(
        PCACellFeatureGraph(n_components=n_components, split_name="train"),
        SetConfig({"label_channel": "cell_type"}),
        log_level=log_level,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_mutations", type=int, default=5, help="Number of mutation trials to search for best pipeline")
    parser.add_argument("--mutation_rate", type=float, default=0.2, help="Probability of AST node mutation")

    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--dense_dim", type=int, default=400, help="number of hidden gcn units")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout probability")
    parser.add_argument("--hidden_dim", type=int, default=200, help="number of hidden gcn units")
    parser.add_argument("--log_level", type=str, default="INFO", choices=get_args(LogLevel))
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--n_epochs", type=int, default=300, help="number of training epochs")
    parser.add_argument("--n_layers", type=int, default=1, help="number of hidden gcn layers")
    parser.add_argument("--species", default="mouse", type=str)
    parser.add_argument("--test_dataset", nargs="+", type=int, default=[1759], help="Testing dataset IDs")
    parser.add_argument("--test_rate", type=float, default=0.2)
    parser.add_argument("--tissue", default="Spleen", type=str)
    parser.add_argument("--train_dataset", nargs="+", type=int, default=[1970], help="List of training dataset ids.")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--num_runs", type=int, default=2, help="Number of repetitions per mutation")
    parser.add_argument("--val_size", type=float, default=0.0, help="val size")
    parser.add_argument("--obs_nums", type=int, default=None)

    args = parser.parse_args()
    logger.setLevel(args.log_level)

    best_mutation_score = -1.0
    best_mutation_code = ""

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

                model = ScDeepSort(args.dense_dim, args.hidden_dim, args.n_layers, args.species, args.tissue,
                                   dropout=args.dropout, batch_size=args.batch_size, device=args.device)
                preprocessing_pipeline = get_preprocessing_pipeline(n_components=args.dense_dim)

                dataloader = CellTypeAnnotationDataset(species=args.species, tissue=args.tissue,
                                                       test_dataset=args.test_dataset,
                                                       train_dataset=args.train_dataset,
                                                       data_dir="../temp_data", val_size=args.val_size)
                data = dataloader.load_data(transform=None, cache=args.cache)

                if args.obs_nums is not None:
                    sub_data(data.data, args.obs_nums)
                    train_idx, test_idx = train_test_split(range(args.obs_nums), test_size=0.2,
                                                           random_state=current_seed)
                    data.set_split_idx("train", train_idx)
                    data.set_split_idx("test", test_idx)

                preprocessing_pipeline(data)

                y_train = data.get_y(split_name="train", return_type="torch")
                y_test = data.get_y(split_name="test", return_type="torch")

                g = data.data.uns["CellFeatureGraph"]
                num_genes = data.shape[1]
                gene_ids = torch.arange(num_genes)
                train_cell_ids = torch.LongTensor(data.train_idx) + num_genes
                test_cell_ids = torch.LongTensor(data.test_idx) + num_genes
                g_train = g.subgraph(torch.concat((gene_ids, train_cell_ids)))
                g_test = g.subgraph(torch.concat((gene_ids, test_cell_ids)))

                model.fit(g_train, y_train.argmax(1), epochs=args.n_epochs, lr=args.lr,
                          weight_decay=args.weight_decay, val_ratio=args.test_rate)
                score = model.score(g_test, y_test)
                inner_score = model.score(g_train, y_train)

                run_end_time = time.time()
                run_time = run_end_time - run_start_time

                results.append(score.item())
                inner_scores.append(inner_score.item())

                search_history.append({
                    "mutation_idx": mutation_idx + 1,
                    "dataset": ref_data_name,
                    "run_idx": run,
                    "seed": current_seed,
                    "status": "Success",
                    "test_score": score.item(),
                    "inner_score": inner_score.item(),
                    "time_seconds": run_time,
                    "error_message": None,
                    "code": current_mutated_code
                })

            except Exception as e:
                run_end_time = time.time()
                run_time = run_end_time - run_start_time

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

    history_df = pd.DataFrame(search_history)
    print("\nMutation Search History Summary (Per Run):")
    print(history_df[['mutation_idx', 'dataset', 'run_idx', 'status', 'test_score', 'inner_score', 'time_seconds', 'error_message']].to_string(index=False))

    # ==========================================
    # 追加保存到 CSV 文件
    # ==========================================
    csv_filename = "mutation_search_history_raw.csv"
    file_exists = os.path.exists(csv_filename)
    history_df.to_csv(csv_filename, mode='a', index=False, header=not file_exists)

    print(f"\nDetailed history appended to '{csv_filename}'")

    print("\nBest Mutated Code:")
    print(best_mutation_code)