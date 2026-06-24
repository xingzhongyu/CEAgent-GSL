import argparse
import ast
import math
import os
import random
import time
import traceback
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.spatial import distance
from skimage import img_as_ubyte
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.neighbors import BallTree, KDTree, NearestNeighbors
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import BatchNorm, Sequential
from torch_sparse import SparseTensor
from torchvision import transforms
from tqdm import tqdm

from dance import logger
from dance.data.base import Data
from dance.datasets.spatial import SpatialLIBDDataset
from dance.modules.spatial.spatial_domain.EfNST import (
    EfNsSTRunner,
    EfNSTAugmentTransform,
    EfNSTConcatgTransform,
    EfNSTGraphTransform,
    EfNSTImageTransform,
)
from dance.registry import register_preprocessor
from dance.transforms.base import BaseTransform
from dance.transforms.cell_feature import CellPCA
from dance.transforms.filter import (
    FilterGenesPercentile,
    HighlyVariableGenesLogarithmizedByTopGenes,
)
from dance.transforms.misc import Compose, SetConfig
from dance.typing import Optional
from dance.utils import set_seed, sub_data
from dance.utils.metrics import calculate_unified_scores, resolve_score_func

try:
    from typing import Literal
except ImportError:
    try:
        from typing_extensions import Literal
    except ImportError:

        class LiteralMeta(type):

            def __getitem__(cls, values):
                if not isinstance(values, tuple):
                    values = (values, )
                return type("Literal_", (Literal, ), dict(__args__=values))

        class Literal(metaclass=LiteralMeta):
            pass


# ==========================================
# 升级版 AST 结构突变器定义
# ==========================================
class StructuralMutator(ast.NodeTransformer):
    def __init__(self, mutation_rate=0.5):
        self.mutation_rate = mutation_rate
        self.n_components_candidates = [20, 30, 50, 100]
        self.spatial_k_candidates = [10, 20, 30, 50, 60]
        self.rate_candidates = [2, 3, 4, 5]
        self.spatial_type_candidates = ["KDTree", "BallTree", "NearestNeighbors"]
        self.dist_type_candidates = ["cosine", "euclidean", "correlation"]

    # 1. 超参数突变：字符串常量和数值常量
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            if node.value in self.spatial_type_candidates:
                if random.random() < self.mutation_rate:
                    return ast.Constant(value=random.choice(self.spatial_type_candidates))
            if node.value in self.dist_type_candidates:
                if random.random() < self.mutation_rate:
                    return ast.Constant(value=random.choice(self.dist_type_candidates))

        # 排除 bool 类型，防止 True/False 被当成 1/0 突变
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            if node.value in self.n_components_candidates:
                if random.random() < self.mutation_rate:
                    return ast.Constant(value=random.choice(self.n_components_candidates))
            if node.value in self.spatial_k_candidates:
                if random.random() < self.mutation_rate:
                    return ast.Constant(value=random.choice(self.spatial_k_candidates))
            if node.value in self.rate_candidates:
                if random.random() < self.mutation_rate:
                    return ast.Constant(value=random.choice(self.rate_candidates))

        return node

    # 2. 结构突变：在 cal_weight_matrix 中插入基因权重裁剪后处理步骤
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.name == 'cal_weight_matrix' and random.random() < self.mutation_rate:
            new_body = []
            for stmt in node.body:
                new_body.append(stmt)
                # 在 gene_correlation 赋值后插入裁剪操作
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    target = stmt.targets[0]
                    if isinstance(target, ast.Name) and target.id == 'gene_correlation':
                        if random.random() < 0.5:
                            extra_code = 'gene_correlation = gene_correlation.clip(0, 1)'
                            extra_stmt = ast.parse(extra_code).body[0]
                            new_body.append(extra_stmt)
            node.body = new_body
        return node


# 原始的基准代码块
EVOLVE_BLOCK_CODE = '''
def cal_spatial_weight(
    data,
    spatial_k=50,
    spatial_type="KDTree",
):
    from sklearn.neighbors import BallTree, KDTree, NearestNeighbors
    if spatial_type == "NearestNeighbors":
        nbrs = NearestNeighbors(n_neighbors=spatial_k + 1, algorithm='ball_tree').fit(data)
        _, indices = nbrs.kneighbors(data)
    elif spatial_type == "KDTree":
        tree = KDTree(data, leaf_size=2)
        _, indices = tree.query(data, k=spatial_k + 1)
    elif spatial_type == "BallTree":
        tree = BallTree(data, leaf_size=2)
        _, indices = tree.query(data, k=spatial_k + 1)
    indices = indices[:, 1:]
    spatial_weight = np.zeros((data.shape[0], data.shape[0]))
    for i in range(indices.shape[0]):
        ind = indices[i]
        for j in ind:
            spatial_weight[i][j] = 1
    return spatial_weight


def cal_gene_weight(data, n_components=50, gene_dist_type="cosine"):
    pca = PCA(n_components=n_components)
    if isinstance(data, np.ndarray):
        data_pca = pca.fit_transform(data)
    elif isinstance(data, csr_matrix):
        data = data.toarray()
        data_pca = pca.fit_transform(data)
    gene_correlation = 1 - pairwise_distances(data_pca, metric=gene_dist_type)
    return gene_correlation


def cal_weight_matrix(adata, platform="Visium", pd_dist_type="euclidean", md_dist_type="cosine",
                      gb_dist_type="correlation", n_components=50, no_morphological=True, spatial_k=30,
                      spatial_type="KDTree", verbose=False):
    if platform == "Visium":
        img_row = adata.obsm['spatial_pixel']['x_pixel']
        img_col = adata.obsm['spatial_pixel']['y_pixel']
        array_row = adata.obsm["spatial"]['x']
        array_col = adata.obsm["spatial"]['y']
        rate = 3
        reg_row = LinearRegression().fit(array_row.values.reshape(-1, 1), img_row)
        reg_col = LinearRegression().fit(array_col.values.reshape(-1, 1), img_col)
        unit = math.sqrt(reg_row.coef_**2 + reg_col.coef_**2)
        coords = adata.obsm['spatial_pixel'][["y_pixel", "x_pixel"]].values
        n_spots = coords.shape[0]
        radius = rate * unit
        nbrs = NearestNeighbors(radius=radius, metric=pd_dist_type, n_jobs=-1).fit(coords)
        distances, indices = nbrs.radius_neighbors(coords, return_distance=True)
        row_ind = []
        col_ind = []
        for i in range(n_spots):
            row_ind.extend([i] * len(indices[i]))
            col_ind.extend(indices[i])
        data = np.ones(len(row_ind), dtype=np.int8)
        physical_distance = csr_matrix((data, (row_ind, col_ind)), shape=(n_spots, n_spots))
    else:
        physical_distance = cal_spatial_weight(adata.obsm['spatial'], spatial_k=spatial_k, spatial_type=spatial_type)

    gene_counts = adata.X.copy()
    gene_correlation = cal_gene_weight(data=gene_counts, gene_dist_type=gb_dist_type, n_components=n_components)
    del gene_counts
    if verbose:
        adata.obsm["gene_correlation"] = gene_correlation
        adata.obsm["physical_distance"] = physical_distance

    if platform == 'Visium':
        morphological_similarity = 1 - pairwise_distances(np.array(adata.obsm["image_feat_pca"]), metric=md_dist_type)
        morphological_similarity[morphological_similarity < 0] = 0
        if verbose:
            adata.obsm["morphological_similarity"] = morphological_similarity
        adata.obsm["weights_matrix_all"] = (physical_distance * gene_correlation * morphological_similarity)
        if no_morphological:
            adata.obsm["weights_matrix_nomd"] = (gene_correlation * physical_distance)
    else:
        adata.obsm["weights_matrix_nomd"] = (gene_correlation * physical_distance)
    return adata
'''


def mutate_and_inject(mutation_rate=0.4):
    """动态突变代码并注入到全局环境"""
    tree = ast.parse(EVOLVE_BLOCK_CODE)
    mutator = StructuralMutator(mutation_rate=mutation_rate)
    mutated_tree = mutator.visit(tree)
    ast.fix_missing_locations(mutated_tree)
    mutated_code = ast.unparse(mutated_tree)
    exec(mutated_code, globals())
    return mutated_code


class SpatialImageDataset(Dataset):

    def __init__(self, paths, transform=None):
        self.paths = paths
        self.spot_names = paths.index.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        spot_name = self.spot_names[idx]
        img_path = self.paths.iloc[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, spot_name


def extract_features_batch(adata, model, device, batch_size=64, num_workers=4):
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = SpatialImageDataset(paths=adata.obs['slices_path'], transform=preprocess)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    all_features = []
    all_spot_names = []
    with torch.no_grad():
        for image_batch, spot_name_batch in tqdm(data_loader, desc="Extracting features"):
            image_batch = image_batch.to(device)
            result_batch = model(image_batch)
            all_features.append(result_batch.cpu().numpy())
            all_spot_names.extend(list(spot_name_batch))
    final_features = np.concatenate(all_features, axis=0)
    feat_df = pd.DataFrame(final_features, index=all_spot_names)
    return adata, feat_df


class Image_Feature:

    def __init__(self, adata, pca_components=50, cnnType='efficientnet-b0', verbose=False, seeds=88, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.adata = adata
        self.pca_components = pca_components
        self.verbose = verbose
        self.seeds = seeds
        self.cnnType = cnnType

    def efficientNet_model(self):
        efficientnet_versions = {
            'efficientnet-b0': 'efficientnet-b0',
            'efficientnet-b1': 'efficientnet-b1',
            'efficientnet-b2': 'efficientnet-b2',
            'efficientnet-b3': 'efficientnet-b3',
            'efficientnet-b4': 'efficientnet-b4',
            'efficientnet-b5': 'efficientnet-b5',
            'efficientnet-b6': 'efficientnet-b6',
            'efficientnet-b7': 'efficientnet-b7',
        }
        if self.cnnType in efficientnet_versions:
            model_version = efficientnet_versions[self.cnnType]
            cnn_pretrained_model = EfficientNet.from_pretrained(model_version)
            cnn_pretrained_model.to(self.device)
        else:
            raise ValueError(f"{self.cnnType} is not a valid EfficientNet type.")
        return cnn_pretrained_model

    def Extract_Image_Feature(self):
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomAutocontrast(),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.)),
            transforms.RandomInvert(),
            transforms.RandomAdjustSharpness(random.uniform(0, 1)),
            transforms.RandomSolarize(random.uniform(0, 1)),
            transforms.RandomAffine(45, translate=(0.3, 0.3), scale=(0.8, 1.2), shear=(-0.3, 0.3, -0.3, 0.3)),
            transforms.RandomErasing()
        ]
        img_to_tensor = transforms.Compose(transform_list)
        feat_df = pd.DataFrame()
        model = self.efficientNet_model()
        model.eval()
        if "slices_path" not in self.adata.obs.keys():
            raise ValueError("Please run the function image_crop first")
        _, feat_df = extract_features_batch(self.adata, model, self.device)
        feat_df = feat_df.transpose()
        self.adata.obsm["image_feat"] = feat_df.transpose().to_numpy()
        if self.verbose:
            print("The image feature is added to adata.obsm['image_feat'] !")
        pca = PCA(n_components=self.pca_components, random_state=self.seeds)
        pca.fit(feat_df.transpose().to_numpy())
        self.adata.obsm["image_feat_pca"] = pca.transform(feat_df.transpose().to_numpy())
        if self.verbose:
            print("The pca result of image feature is added to adata.obsm['image_feat_pca'] !")
        return self.adata


def image_crop(adata, save_path, crop_size=50, target_size=224, verbose=False, quality='hires'):
    image = adata.uns["image"]
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    img_pillow = Image.fromarray(img_as_ubyte(image))
    tile_names = []
    with tqdm(total=len(adata), desc="Tiling image", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        for imagerow, imagecol in zip(adata.obsm["spatial_pixel"]['x_pixel'], adata.obsm["spatial_pixel"]['y_pixel']):
            imagerow_down = imagerow - crop_size / 2
            imagerow_up = imagerow + crop_size / 2
            imagecol_left = imagecol - crop_size / 2
            imagecol_right = imagecol + crop_size / 2
            tile = img_pillow.crop((imagecol_left, imagerow_down, imagecol_right, imagerow_up))
            tile.thumbnail((target_size, target_size), Image.LANCZOS)
            tile.resize((target_size, target_size))
            tile_name = str(imagecol) + "-" + str(imagerow) + "-" + str(crop_size)
            out_tile = Path(save_path) / (tile_name + ".png")
            tile_names.append(str(out_tile))
            if verbose:
                print("generate tile at location ({}, {})".format(str(imagecol), str(imagerow)))
            tile.save(out_tile, "PNG")
            pbar.update(1)
    adata.obs["slices_path"] = tile_names
    if verbose:
        print("The slice path of image feature is added to adata.obs['slices_path'] !")
    return adata


class graph:

    def __init__(self, data, rad_cutoff, k, distType='euclidean'):
        super().__init__()
        self.data = data
        self.distType = distType
        self.k = k
        self.rad_cutoff = rad_cutoff
        self.num_cell = data.shape[0]

    def graph_computing(self):
        graphList = []
        if self.distType == "KDTree":
            from sklearn.neighbors import KDTree
            tree = KDTree(self.data)
            dist, ind = tree.query(self.data, k=self.k + 1)
            indices = ind[:, 1:]
            graphList = [(node_idx, indices[node_idx][j]) for node_idx in range(self.data.shape[0])
                         for j in range(indices.shape[1])]
        elif self.distType == "kneighbors_graph":
            from sklearn.neighbors import kneighbors_graph
            A = kneighbors_graph(self.data, n_neighbors=self.k, mode='connectivity', include_self=False)
            A = A.toarray()
            graphList = [(node_idx, indices[j]) for node_idx in range(self.data.shape[0])
                         for j in np.where(A[node_idx] == 1)[0]]
        elif self.distType == "Radius":
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(radius=self.rad_cutoff).fit(self.data)
            distances, indices = nbrs.radius_neighbors(self.data, return_distance=True)
            graphList = [(node_idx, indices[node_idx][j]) for node_idx in range(indices.shape[0])
                         for j in range(indices[node_idx].shape[0]) if distances[node_idx][j] > 0]
        return graphList

    def List2Dict(self, graphList):
        graphdict = {}
        tdict = {}
        for end1, end2 in graphList:
            tdict[end1] = ""
            tdict[end2] = ""
            graphdict.setdefault(end1, []).append(end2)
        for i in range(self.num_cell):
            if i not in tdict:
                graphdict[i] = []
        return graphdict

    def mx2SparseTensor(self, mx):
        mx = mx.tocoo().astype(np.float32)
        row = torch.from_numpy(mx.row).to(torch.long)
        col = torch.from_numpy(mx.col).to(torch.long)
        values = torch.from_numpy(mx.data)
        adj = SparseTensor(row=row, col=col, value=values, sparse_sizes=mx.shape)
        adj_ = adj.t()
        return adj_

    def preprocess_graph(self, adj):
        adj = sp.coo_matrix(adj)
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return self.mx2SparseTensor(adj_normalized)

    def main(self):
        adj_mtx = self.graph_computing()
        graph_dict = self.List2Dict(adj_mtx)
        adj_org = nx.adjacency_matrix(nx.from_dict_of_lists(graph_dict))
        adj_pre = adj_org - sp.dia_matrix((adj_org.diagonal()[np.newaxis, :], [0]), shape=adj_org.shape)
        adj_pre.eliminate_zeros()
        adj_norm = self.preprocess_graph(adj_pre)
        adj_label = adj_pre + sp.eye(adj_pre.shape[0])
        adj_label = torch.FloatTensor(adj_label.toarray())
        norm = adj_pre.shape[0] * adj_pre.shape[0] / float((adj_pre.shape[0] * adj_pre.shape[0] - adj_pre.sum()) * 2)
        graph_dict = {"adj_norm": adj_norm, "adj_label": adj_label, "norm_value": norm}
        return graph_dict

    def combine_graph_dicts(self, dict_1, dict_2):
        tmp_adj_norm = torch.block_diag(dict_1['adj_norm'].to_dense(), dict_2['adj_norm'].to_dense())
        graph_dict = {
            "adj_norm": SparseTensor.from_dense(tmp_adj_norm),
            "adj_label": torch.block_diag(dict_1['adj_label'], dict_2['adj_label']),
            "norm_value": np.mean([dict_1['norm_value'], dict_2['norm_value']])
        }
        return graph_dict


def find_adjacent_spot(adata, use_data="raw", neighbour_k=4, weights='weights_matrix_all', verbose=False):
    if use_data == "raw":
        if isinstance(adata.X, (csr_matrix, np.ndarray)):
            gene_matrix = adata.X.toarray()
        elif isinstance(adata.X, np.ndarray):
            gene_matrix = adata.X
        elif isinstance(adata.X, pd.Dataframe):
            gene_matrix = adata.X.values
        else:
            raise ValueError(f"""{type(adata.X)} is not a valid type.""")
    else:
        gene_matrix = adata.obsm[use_data]
    weights_matrix = adata.obsm[weights]
    weights_list = []
    final_coordinates = []
    for i in range(adata.shape[0]):
        if weights == "physical_distance":
            current_spot = adata.obsm[weights][i].argsort()[-(neighbour_k + 3):][:(neighbour_k + 2)]
        else:
            current_spot = adata.obsm[weights][i].argsort()[-neighbour_k:][:neighbour_k - 1]
        spot_weight = adata.obsm[weights][i][current_spot]
        spot_matrix = gene_matrix[current_spot]
        if spot_weight.sum() > 0:
            spot_weight_scaled = spot_weight / spot_weight.sum()
            weights_list.append(spot_weight_scaled)
            spot_matrix_scaled = np.multiply(spot_weight_scaled.reshape(-1, 1), spot_matrix)
            spot_matrix_final = np.sum(spot_matrix_scaled, axis=0)
        else:
            spot_matrix_final = np.zeros(gene_matrix.shape[1])
            weights_list.append(np.zeros(len(current_spot)))
        final_coordinates.append(spot_matrix_final)
    adata.obsm['adjacent_data'] = np.array(final_coordinates)
    if verbose:
        adata.obsm['adjacent_weight'] = np.array(weights_list)
    return adata


def augment_gene_data(adata, Adj_WT=0.2):
    adjacent_gene_matrix = adata.obsm["adjacent_data"].astype(float)
    if isinstance(adata.X, np.ndarray):
        augment_gene_matrix = adata.X + Adj_WT * adjacent_gene_matrix
    elif isinstance(adata.X, csr_matrix):
        augment_gene_matrix = adata.X.toarray() + Adj_WT * adjacent_gene_matrix
    adata.obsm["augment_gene_data"] = augment_gene_matrix
    del adjacent_gene_matrix
    return adata


def augment_adata(adata, platform="Visium", pd_dist_type="euclidean", md_dist_type="cosine",
                  gb_dist_type="correlation", n_components=50, no_morphological=False, use_data="raw",
                  neighbour_k=4, weights="weights_matrix_all", Adj_WT=0.2, spatial_k=30, spatial_type="KDTree"):
    adata = cal_weight_matrix(
        adata,
        platform=platform,
        pd_dist_type=pd_dist_type,
        md_dist_type=md_dist_type,
        gb_dist_type=gb_dist_type,
        n_components=n_components,
        no_morphological=no_morphological,
        spatial_k=spatial_k,
        spatial_type=spatial_type,
    )
    adata = find_adjacent_spot(adata, use_data=use_data, neighbour_k=neighbour_k, weights=weights)
    adata = augment_gene_data(adata, Adj_WT=Adj_WT)
    return adata


def get_preprocessing_pipeline(verbose=False, cnnType='efficientnet-b0', pca_n_comps=200, distType="KDTree", k=12,
                                dim_reduction=True, min_cells=3, platform="Visium", device=None):
    return Compose(
        EfNSTImageTransform(verbose=verbose, cnnType=cnnType),
        EfNSTAugmentTransform(),
        EfNSTGraphTransform(distType=distType, k=k),
        EfNSTConcatgTransform(dim_reduction=dim_reduction, min_cells=min_cells, platform=platform,
                              pca_n_comps=pca_n_comps),
        SetConfig({
            "feature_channel": ["feature.cell", "EfNSTGraph"],
            "feature_channel_type": ["obsm", "uns"],
            "label_channel": "label",
            "label_channel_type": "obs"
        }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_mutations", type=int, default=5,
                        help="Number of mutation trials to search for best pipeline")
    parser.add_argument("--mutation_rate", type=float, default=0.4, help="Probability of AST node mutation")

    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--sample_number", type=str, default="151507",
                        help="12 human dorsolateral prefrontal cortex datasets for the spatial domain task.")
    parser.add_argument("--n_components", type=int, default=50, help="Number of PC components.")
    parser.add_argument("--neighbors", type=int, default=17, help="Number of neighbors.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--cnnType", type=str, default="efficientnet-b0")
    parser.add_argument("--pretrain", action="store_true", help="Pretrain the model.")
    parser.add_argument("--pre_epochs", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--Conv_type", type=str, default="ResGatedGraphConv")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information.")
    parser.add_argument("--pca_n_comps", type=int, default=200, help="Number of PCA components.")
    parser.add_argument("--distType", type=str, default="KDTree", help="Distance type.")
    parser.add_argument("--k", type=int, default=12, help="Number of neighbors.")
    parser.add_argument("--no_dim_reduction", action="store_true", help="Disable dimensionality reduction.")
    parser.add_argument("--min_cells", type=int, default=3, help="Minimum number of cells.")
    parser.add_argument("--platform", type=str, default="Visium", help="Platform type.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda', 'cpu', 'cuda:0').")
    parser.add_argument("--obs_nums", type=int, default=10000)
    args = parser.parse_args()

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
            ref_data_name = f"LIBD_{args.sample_number}"

            run_start_time = time.time()
            adata = None
            # try:
            set_seed(current_seed, extreme_mode=True)

            EfNST = EfNsSTRunner(
                platform=args.platform,
                pre_epochs=args.pre_epochs,
                epochs=args.epochs,
                cnnType=args.cnnType,
                Conv_type=args.Conv_type,
                random_state=current_seed)
            dataloader = SpatialLIBDDataset(data_id=args.sample_number)
            data = dataloader.load_data(transform=None, cache=args.cache)
            sub_data(data.data, n_cells=args.obs_nums)
            data.data.uns['data_name'] = args.sample_number
            preprocessing_pipeline = get_preprocessing_pipeline(
                verbose=args.verbose, cnnType=args.cnnType,
                pca_n_comps=args.pca_n_comps, distType=args.distType,
                k=args.k, dim_reduction=not args.no_dim_reduction,
                min_cells=args.min_cells, platform=args.platform,
                device=args.device)
            preprocessing_pipeline(data)
            (x, adj), y = data.get_data()
            adata = data.data
            adata = EfNST.fit(adata, x, graph_dict=adj, pretrain=args.pretrain)
            n_domains = len(np.unique(y))
            adata = EfNST._get_cluster_data(adata, n_domains=n_domains, priori=True)
            y_pred = EfNST.predict(adata)

            silhouette_score = resolve_score_func("silhouette")
            calinski_harabasz_score = resolve_score_func("calinski_harabasz")
            davies_bouldin_score = resolve_score_func("davies_bouldin")
            inner_score = calculate_unified_scores({
                "silhouette": silhouette_score(x, y_pred),
                "calinski_harabasz": calinski_harabasz_score(x, y_pred),
                "davies_bouldin": davies_bouldin_score(x, y_pred)
            })

            score = adjusted_rand_score(y, y_pred)

            run_end_time = time.time()
            run_time = run_end_time - run_start_time

            results.append(score)
            inner_scores.append(inner_score)

            search_history.append({
                "mutation_idx": mutation_idx + 1,
                "dataset": ref_data_name,
                "run_idx": run,
                "seed": current_seed,
                "status": "Success",
                "test_score": score,
                "inner_score": inner_score,
                "time_seconds": run_time,
                "error_message": None,
                "code": current_mutated_code
            })

            # except Exception as e:
            #     run_end_time = time.time()
            #     run_time = run_end_time - run_start_time

            #     error_msg = f"{type(e).__name__}: {str(e)}"
            #     print(f"Mutation {mutation_idx + 1} Run {run} Failed! Error: {error_msg}")

            #     search_history.append({
            #         "mutation_idx": mutation_idx + 1,
            #         "dataset": ref_data_name,
            #         "run_idx": run,
            #         "seed": current_seed,
            #         "status": "Failed",
            #         "test_score": None,
            #         "inner_score": None,
            #         "time_seconds": run_time,
            #         "error_message": error_msg,
            #         "code": current_mutated_code
            #     })
            #     break

            # finally:
            #     if adata is not None:
            #         try:
            #             EfNST.delete_imgs(adata)
            #         except Exception:
            #             pass

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
    print(history_df[['mutation_idx', 'dataset', 'run_idx', 'status', 'test_score', 'inner_score', 'time_seconds',
                       'error_message']].to_string(index=False))

    # ==========================================
    # 追加保存到 CSV 文件
    # ==========================================
    csv_filename = "mutation_search_history_raw.csv"
    file_exists = os.path.exists(csv_filename)
    history_df.to_csv(csv_filename, mode='a', index=False, header=not file_exists)

    print(f"\nDetailed history appended to '{csv_filename}'")

    print("\nBest Mutated Code:")
    print(best_mutation_code)
