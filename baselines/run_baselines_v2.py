#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_baselines_v2.py — 统一协议 (Unified Protocol) 下的通用推荐基线

与 v1 (run_baselines.py) 的区别 —— 目的：消除与 MV-GCN 对比中的混杂因素。

v1 的两个协议失配（导致基线退化为 popularity 解）：
  1. 目标失配：v1 用 BPR 逐对训练；MV-GCN 用频率加权多标签 BCE。
  2. 粒度失配：v1 训练单元是单个症状；测试单元是症状集合（集合嵌入取平均）。

v2 协议（架构不变，只统一训练与选择流程）：
  - 训练单元: 处方级 —— 症状集合 -> 全草药表上的多热标签 (与 MV-GCN 相同)
  - 损失函数: 频率加权 BCE，w_i = 1 + log(freq_max / freq_i)  (镜像 load_data.py:101)
  - 集合表示: 各基线编码器的症状嵌入在集合内平均池化 (不使用 VLR —— 那是 MV-GCN 的贡献)
  - 预测: logits = set_repr @ E_herb^T  (与 MV-GCN 相同的点积打分形式)
  - 模型选择: 训练集划出 val_ratio 作验证集，按验证集 R@20 选最优 epoch，
              报告该 epoch 的测试集指标 (无测试集泄漏)
  - 超参与 MV-GCN 对齐: Adam lr=2e-4, explicit L2 decay=7e-3, batch=1024, epoch=200

模型与架构 (与 v1 完全一致，未做任何改动)：
  - MF:       纯嵌入
  - NGCF:     3 层 [64] x3, 拼接各层, 对称归一化邻接 (含自环)
  - LightGCN: 3 层, 各层均值融合, 对称归一化邻接 (含自环)
  - GAT:      2 层 (8 heads x 8 dim -> 1 head x 64), S-H 二部图

用法:
  python run_baselines_v2.py --dataset smgcn --model all --seeds 20
  python run_baselines_v2.py --dataset TCM_PD --model LightGCN --seeds 20 --tag dim256 --dim 256
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KS = [5, 10, 20]
EXPERIMENT_SCHEMA = "unified-v2-p1-20260929"
FLOW_EXPERIMENT_SCHEMA = "unified-v2-flow-v1-20261006"
MODEL_NAMES = ["Popularity", "MF", "NGCF", "LightGCN", "GAT", "MVGCNLite",
               "MVGCN", "MVGCNFlow", "MLP", "SMGCN", "SMGCN_MVANA"]
MODEL_GROUPS = {
    # 论文基础对照：非图非线性、协同图传播、中医领域图模型。
    "basic": ["MLP", "NGCF", "SMGCN"],
    # 论文正式横向比较：从非图方法、经典协同图模型到领域图模型。
    # MVGCNFlow 是失败方向的留档模型，不进入该正式比较组。
    "comparison": ["Popularity", "MF", "MLP", "NGCF", "LightGCN", "GAT",
                   "SMGCN", "MVGCNLite", "MVGCN", "SMGCN_MVANA"],
    # 保留旧 all 的精确含义，避免现有批处理突然多跑模型。
    "all": ["MF", "NGCF", "LightGCN", "GAT", "MVGCNLite", "MVGCN"],
}
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("MVANA_DATA_ROOT", os.path.join(BASE, "..", "Data"))
RESULT_DIR = os.environ.get("MVANA_RESULT_DIR", os.path.join(BASE, "results"))

# 全局计算设备；main() 依据 --device 设定（默认 auto：有 CUDA 就用，否则 CPU）
DEVICE = torch.device("cpu")


def resolve_device(spec):
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] 请求 {spec} 但 CUDA 不可用，回退 CPU", flush=True)
        return torch.device("cpu")
    return torch.device(spec)


def experiment_fingerprint(args, model_name):
    """生成影响实验结果的配置指纹。

    seeds / seed_start 只控制要跑哪些重复试验，trace 只控制日志，
    tag 已经是文件名的一部分；这些字段不应阻止合法续跑。
    其余参数任何变化都必须使用新结果文件，避免混合配置。
    """
    ignored = {"seeds", "seed_start", "trace", "tag", "model"}
    # Flow-only knobs must not invalidate or fragment existing baseline runs.
    if model_name != "MVGCNFlow":
        ignored.update({"lambda_syn", "lambda_trt", "lambda_con",
                        "contrast_temp", "flow_teacher", "flow_gamma_init"})
    config = {k: v for k, v in vars(args).items() if k not in ignored}
    config["model_name"] = model_name
    config["experiment_schema"] = (FLOW_EXPERIMENT_SCHEMA
                                   if model_name == "MVGCNFlow"
                                   else EXPERIMENT_SCHEMA)
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], config


def resolve_models(spec):
    """将单模型或预设比较组解析成有序模型列表。"""
    if spec in MODEL_GROUPS:
        return list(MODEL_GROUPS[spec])
    if spec not in MODEL_NAMES:
        raise ValueError("unknown model/group %r; choices=%s" %
                         (spec, MODEL_NAMES + sorted(MODEL_GROUPS)))
    return [spec]


# ----------------------------------------------------------------------
# 数据
# ----------------------------------------------------------------------
def load_dataset(name):
    path = os.path.join(DATA_ROOT, name)
    train, test = [], []
    n_users = n_items = 0
    for fname, bucket in (("train.txt", train), ("test.txt", test)):
        with open(os.path.join(path, fname), encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                try:
                    syms = [int(x) for x in parts[0].split()]
                    herbs = [int(x) for x in parts[1].split()]
                except ValueError:
                    continue
                if not syms or not herbs:
                    continue
                bucket.append((syms, herbs))
                n_users = max(n_users, max(syms) + 1)
                n_items = max(n_items, max(herbs) + 1)
    return train, test, n_users, n_items


def build_R(records, n_users, n_items, weighting="binary"):
    """症状-草药二部图（仅由训练数据构建）。

    binary 保留旧协议；count 保留共现强度；ppmi 去除症状/药物边际
    流行度并截断负 PMI，更适合稠密的 TCM_Lung 图。
    """
    R = np.zeros((n_users, n_items), dtype=np.float32)
    for syms, herbs in records:
        for s in set(syms):
            R[s, list(set(herbs))] += 1.0
    if weighting == "binary":
        R = (R > 0).astype(np.float32)
    elif weighting == "ppmi":
        total = float(R.sum())
        row = R.sum(1, keepdims=True)
        col = R.sum(0, keepdims=True)
        denom = row @ col
        mask = R > 0
        out = np.zeros_like(R)
        out[mask] = np.maximum(
            np.log((R[mask] * total) / np.maximum(denom[mask], 1e-12)), 0.0)
        R = out
    elif weighting != "count":
        raise ValueError(f"unknown interaction weighting: {weighting}")
    return R


def item_weights(records, n_items):
    """逆频率权重, 镜像 MV-GCN 的 utility/load_data.py:97-103"""
    freq = np.zeros(n_items, dtype=np.float64)
    for _, herbs in records:
        for h in herbs:
            freq[h] += 1
    fmax = freq.max()
    w = np.where(freq > 0, 1.0 + np.log(np.maximum(fmax, 1) / np.maximum(freq, 1)),
                 1.0 + np.log(np.maximum(fmax, 1)))
    return w.astype(np.float32), freq


def build_direct_prior(records, n_users, n_items, shrink=5.0):
    """经验贝叶斯 P(药物|症状)，只由实际训练子集估计。

    低频症状向全局药物频率收缩，避免一次共现就得到极端概率。
    返回 [n_users, n_items]，供症状直连残差初始化。
    """
    pair = np.zeros((n_users, n_items), dtype=np.float32)
    sym_freq = np.zeros(n_users, dtype=np.float32)
    herb_freq = np.zeros(n_items, dtype=np.float32)
    for syms, herbs in records:
        syms, herbs = list(set(syms)), list(set(herbs))
        sym_freq[syms] += 1.0
        herb_freq[herbs] += 1.0
        pair[np.ix_(syms, herbs)] += 1.0
    global_p = herb_freq / max(float(len(records)), 1.0)
    alpha = max(float(shrink), 0.0)
    return ((pair + alpha * global_p[None, :]) /
            np.maximum(sym_freq[:, None] + alpha, 1e-12)).astype(np.float32)


def make_labels(records, n_items):
    Y = np.zeros((len(records), n_items), dtype=np.float32)
    for r, (_, herbs) in enumerate(records):
        Y[r, herbs] = 1.0
    return Y


def build_pool(records, n_users):
    """症状集合的平均池化矩阵 [n_records, n_users] (行归一化)"""
    P = np.zeros((len(records), n_users), dtype=np.float32)
    for r, (syms, _) in enumerate(records):
        for s in syms:
            P[r, s] = 1.0
    P /= np.maximum(P.sum(1, keepdims=True), 1e-12)
    return torch.from_numpy(P)


# ----------------------------------------------------------------------
# 模型 (与 v1 完全一致)
# ----------------------------------------------------------------------
class MF(nn.Module):
    name = "MF"

    def __init__(self, n_users, n_items, dim, **kw):
        super().__init__()
        self.E0 = nn.Parameter(torch.randn(n_users + n_items, dim) * 0.01)

    def embeddings(self):
        return self.E0


class Popularity(nn.Module):
    """非个性化高频药基线，用来识别模型是否只学到了流行度。"""

    name = "Popularity"
    use_popularity = True

    def __init__(self, n_users, n_items, item_scores=None, **kw):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.register_buffer("item_scores", item_scores.float())
        self.register_buffer("E_static", torch.zeros(n_users + n_items, 1))

    def embeddings(self):
        return self.E_static


def _scores(E, set_repr, n_users, model=None, P=None):
    """统一的打分：默认双线性；若模型带 MLP 头则改用非线性头"""
    if model is not None and getattr(model, "use_popularity", False):
        return model.item_scores.unsqueeze(0).expand(set_repr.shape[0], -1)
    if model is not None and getattr(model, "use_flow_head", False):
        return model.flow_forward(set_repr, E)["logits"]
    if model is not None and getattr(model, "use_mlp_head", False):
        logits = model.head(set_repr) @ E[n_users:].t()
    else:
        logits = set_repr @ E[n_users:].t()
    if model is not None and getattr(model, "use_direct_residual", False):
        if P is None:
            raise ValueError("direct residual requires the raw symptom pool matrix P")
        logits = model.fuse_direct(P, logits)
    return logits


class MLP(nn.Module):
    """MLP —— 非图、非线性基线（FordNet 式解码器）

    症状嵌入平均池化 -> MLP -> 与草药嵌入点积。
    用途：分离「图传播」与「非线性」两个因素——若 MLP 追平图模型，
    说明收益来自非线性拟合而非邻域传播。
    """

    name = "MLP"
    use_mlp_head = True

    def __init__(self, n_users, n_items, dim, **kw):
        super().__init__()
        self.E0 = nn.Parameter(torch.randn(n_users + n_items, dim) * 0.01)
        self.fc1 = nn.Linear(dim, 2 * dim)
        self.fc2 = nn.Linear(2 * dim, dim)

    def embeddings(self):
        return self.E0

    def head(self, set_repr):
        return self.fc2(F.relu(self.fc1(set_repr)))


class SMGCN(nn.Module):
    """SMGCN —— 中医领域图基线

    症状-草药二部图传播 + 症状相似度(S-S)图传播，每层 W + tanh，
    层间固定均值组合。相对我们的 MV-GCN 少了两样东西：
      1. 草药-草药(H-H)视图
      2. 逐节点自适应的深度组合 (SANA)
    因此 MV-GCN 与 SMGCN 的差距，恰好度量「多视图 + 自适应深度」的贡献。
    """

    name = "SMGCN"

    def __init__(self, n_users, n_items, dim, A_norm=None, A_ss=None,
                 n_layers=3, **kw):
        super().__init__()
        N = n_users + n_items
        self.n_users, self.n_items = n_users, n_items
        self.n_layers = n_layers
        self.E0 = nn.Parameter(torch.randn(N, dim) * 0.01)
        self.register_buffer("A", A_norm)
        self.has_ss = A_ss is not None
        if self.has_ss:
            self.register_buffer("A_ss", A_ss)
        self.Ws = nn.ParameterList([nn.Parameter(torch.empty(dim, dim))
                                    for _ in range(n_layers)])
        for W in self.Ws:
            nn.init.xavier_uniform_(W)

    def _prop(self, A, E0):
        Es = [E0]
        E = E0
        for W in self.Ws:
            E = torch.tanh((A @ E) @ W)
            Es.append(E)
        return torch.stack(Es, dim=0).mean(0)

    def embeddings(self):
        E = self._prop(self.A, self.E0)
        if self.has_ss:
            nu = self.n_users
            E = torch.cat([E[:nu] + self._prop(self.A_ss, self.E0[:nu]),
                           E[nu:]], dim=0)
        return E


class NGCF(nn.Module):
    name = "NGCF"

    def __init__(self, n_users, n_items, dim, A_norm=None, n_layers=3, **kw):
        super().__init__()
        self.n_layers = n_layers
        self.E0 = nn.Parameter(torch.randn(n_users + n_items, dim) * 0.01)
        self.register_buffer("A", A_norm)
        self.Ws = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])

    def embeddings(self):
        Es = [self.E0]
        E = self.E0
        for W in self.Ws:
            E = F.leaky_relu(W(self.A @ E))
            Es.append(E)
        return torch.cat(Es, dim=-1)


class LightGCN(nn.Module):
    name = "LightGCN"

    def __init__(self, n_users, n_items, dim, A_norm=None, n_layers=3, **kw):
        super().__init__()
        self.n_layers = n_layers
        self.E0 = nn.Parameter(torch.randn(n_users + n_items, dim) * 0.01)
        self.register_buffer("A", A_norm)

    def embeddings(self):
        Es = [self.E0]
        E = self.E0
        for _ in range(self.n_layers):
            E = self.A @ E
            Es.append(E)
        return torch.stack(Es, dim=0).mean(0)


class GAT(nn.Module):
    """GAT (ICLR'18) — 邻域稀疏注意力实现 (数学上等价于 mask 版, 但复杂度 O(E) 而非 O(N^2))
    layer1: 8 heads x 8 dim (concat) -> layer2: 1 head x 64"""

    name = "GAT"

    def __init__(self, n_users, n_items, dim, edge_src=None, edge_dst=None,
                 n_heads=8, **kw):
        super().__init__()
        N = n_users + n_items
        self.N = N
        self.E0 = nn.Parameter(torch.randn(N, dim) * 0.01)
        self.register_buffer("edge_src", edge_src)
        self.register_buffer("edge_dst", edge_dst)
        h = dim // n_heads
        self.n_heads, self.head_dim = n_heads, h
        self.W1 = nn.Parameter(torch.empty(n_heads, dim, h))
        self.a1_src = nn.Parameter(torch.empty(n_heads, h))
        self.a1_dst = nn.Parameter(torch.empty(n_heads, h))
        self.W2 = nn.Parameter(torch.empty(1, dim, dim))
        self.a2_src = nn.Parameter(torch.empty(1, dim))
        self.a2_dst = nn.Parameter(torch.empty(1, dim))
        for p in (self.W1, self.a1_src, self.a1_dst, self.W2, self.a2_src, self.a2_dst):
            nn.init.xavier_uniform_(p)

    def _attn(self, h, W, a_src, a_dst):
        """h: [N, d_in] -> [N, d_out]; 注意力仅在同边邻居间归一化"""
        hw = h @ W                                            # [N, F]
        s = hw[self.edge_src] @ a_src                         # [E]
        d = hw[self.edge_dst] @ a_dst                         # [E]
        e = F.leaky_relu(s + d)
        emax = torch.full((self.N,), -1e9, device=h.device)
        emax = emax.index_reduce(0, self.edge_dst, e, "amax", include_self=True)
        ex = torch.exp(e - emax[self.edge_dst])
        denom = torch.zeros(self.N, device=h.device).index_add(0, self.edge_dst, ex).clamp(min=1e-12)
        alpha = ex / denom[self.edge_dst]                     # [E]
        out = torch.zeros_like(hw).index_add(
            0, self.edge_dst, alpha.unsqueeze(1) * hw[self.edge_src])
        return out

    def embeddings(self):
        x = self.E0
        outs = [self._attn(x, self.W1[k], self.a1_src[k], self.a1_dst[k])
                for k in range(self.n_heads)]
        x = F.elu(torch.cat(outs, dim=-1))
        return self._attn(x, self.W2[0], self.a2_src[0], self.a2_dst[0])


# ----------------------------------------------------------------------
# 图构建
# ----------------------------------------------------------------------
def build_norm_adj(R, symmetric=True, self_loop=True):
    n_users, n_items = R.shape
    N = n_users + n_items
    A = np.zeros((N, N), dtype=np.float32)
    A[:n_users, n_users:] = R
    A[n_users:, :n_users] = R.T
    if self_loop:
        A += np.eye(N, dtype=np.float32)
    if symmetric:
        d = A.sum(1)
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
        A = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    else:
        d = A.sum(1)
        A = A / np.maximum(d, 1e-12)[:, None]
    return torch.from_numpy(A)


def build_adj_mask(R):
    n_users, n_items = R.shape
    N = n_users + n_items


# ----------------------------------------------------------------------
# 证型 / 治法 中介视图（级联式辨证论治先验）
# ----------------------------------------------------------------------
def load_label_lists(ds_name, fname):
    """读取与 train.txt / test.txt 逐行同序的标签文件（每行 = 空格分隔的 id）"""
    path = os.path.join(DATA_ROOT, ds_name, fname)
    if not os.path.exists(path):
        return None
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append([int(x) for x in line.split() if x.strip() != ""])
    return out


def remap_label_space(train_labels, test_labels=None):
    """将可能从 1 开始或不连续的证候/治法 ID 映射为 0..C-1。"""
    if train_labels is None:
        return None, test_labels, {}, 0
    universe = sorted({x for row in train_labels for x in row} |
                      ({x for row in (test_labels or []) for x in row}))
    mapping = {old: new for new, old in enumerate(universe)}
    convert = lambda rows: ([[mapping[x] for x in row] for row in rows]
                            if rows is not None else None)
    return convert(train_labels), convert(test_labels), mapping, len(universe)


def make_multilabel(label_lists, n_classes):
    """将逐处方的类别 ID 列表转为多热标签。"""
    Y = np.zeros((len(label_lists), n_classes), dtype=np.float32)
    for i, labels in enumerate(label_lists):
        if labels:
            Y[i, labels] = 1.0
    return Y


def _norm_rect(C):
    """矩形二部关系的对称度归一化 D_l^-1/2 C D_r^-1/2。"""
    dl = np.maximum(C.sum(1), 1e-12)
    dr = np.maximum(C.sum(0), 1e-12)
    return ((1.0 / np.sqrt(dl))[:, None] * C *
            (1.0 / np.sqrt(dr))[None, :]).astype(np.float32)


def build_four_partite_relations(records, syn_labels, trt_labels,
                                  n_users, n_items, n_syn, n_trt):
    """仅由实际训练子集构建 S-Z、Z-T、T-H 三个有向语义关系。"""
    if not (len(records) == len(syn_labels) == len(trt_labels)):
        raise ValueError("four-partite labels are not aligned with records")
    C_sz = np.zeros((n_users, n_syn), dtype=np.float32)
    C_zt = np.zeros((n_syn, n_trt), dtype=np.float32)
    C_th = np.zeros((n_trt, n_items), dtype=np.float32)
    for (syms, herbs), syns, trts in zip(records, syn_labels, trt_labels):
        for s in set(syms):
            for z in set(syns):
                C_sz[s, z] += 1.0
        for z in set(syns):
            for t in set(trts):
                C_zt[z, t] += 1.0
        for t in set(trts):
            for h in set(herbs):
                C_th[t, h] += 1.0
    return (torch.from_numpy(_norm_rect(C_sz)),
            torch.from_numpy(_norm_rect(C_zt)),
            torch.from_numpy(_norm_rect(C_th)))


def build_mediator_adj(tr, syn_lab, trt_lab, n_users, n_items, mu=1.0,
                       score="lift"):
    """构建「症状 -> 证型/治法 -> 草药」中介视图的邻接矩阵。

    中医处方的基本推理链是「辨证 -> 立法 -> 选药」。本函数把这条链编码成
    图上的两跳路径 S -> M -> H（M = 证型节点 + 治法节点），让其与主二部图、
    S-S / H-H 共现视图一样，以残差形式注入 MV-GCN。

    边权有两种取法：
      cond : 条件概率   w(s,m)=P(m|s), w(m,h)=P(h|m)
      lift : 提升度     w(s,m)=P(m|s)/P(m), w(m,h)=P(h|m)/P(h)
                        减掉边际分布，只保留「差异化偏好」部分（默认）
    lift 更接近 PMI，能避免高频证型退化成全局流行度（本数据集中证型 1 占
    46%，其 P(h|y) 几乎等于 P(h)，cond 形式下这条边基本不带信息）。

    只用训练集统计，避免测试集标签泄漏。

    返回 (A_med, n_med)，A_med 形状 [(n_users+n_items+n_med)^2]，
    节点排布: 症状 | 草药 | 证型 | 治法。
    """
    n_syn = (max((max(y) for y in syn_lab if y), default=0) + 1) if syn_lab else 0
    n_trt = (max((max(y) for y in trt_lab if y), default=0) + 1) if trt_lab else 0
    n_med = n_syn + n_trt
    if n_med == 0:
        return None, 0
    N = n_users + n_items
    A = np.zeros((N + n_med, N + n_med), dtype=np.float32)

    blocks = []          # (n_left, n_right, cnt_lr, cnt_l, cnt_r)
    if syn_lab:
        blocks.append((n_users, n_syn, "syn"))
    if trt_lab:
        blocks.append((n_users, n_trt, "trt"))

    for nl, nr, kind in blocks:
        lab = syn_lab if kind == "syn" else trt_lab
        off = 0 if kind == "syn" else n_syn
        cnt_lr = np.zeros((nl, nr), dtype=np.float64)   # count(s, m)
        cnt_rl = np.zeros((nr, n_items), dtype=np.float64)  # count(m, h)
        cnt_l = np.zeros(nl, dtype=np.float64)          # count(s)
        cnt_r = np.zeros(nr, dtype=np.float64)          # count(m) 记录级
        marg_l = np.zeros(nr, dtype=np.float64)         # 中介的边际(记录级)
        marg_h = np.zeros(n_items, dtype=np.float64)    # 草药边际
        n_rec = len(tr)
        for i, (syms, herbs) in enumerate(tr):
            ms = lab[i] if i < len(lab) else []
            if not ms:
                continue
            for m in ms:
                marg_l[m] += 1.0
                for h in herbs:
                    cnt_rl[m, h] += 1.0
                    marg_h[h] += 1.0 / max(len(ms), 1)
            for s in syms:
                cnt_l[s] += 1.0
                for m in ms:
                    cnt_lr[s, m] += 1.0
        # 左半边 S -> M  (lift 用 PMI 形式: log(P(m|s)/P(m)), 保留流行度乘子;
        # 直接减 1 会把高频证型的权重压到 0, 排序被低频噪声药占据, 实测 P@5 崩到 0.015)
        pr = marg_l / max(marg_l.sum(), 1e-12)
        for s in range(nl):
            if cnt_l[s] <= 0:
                continue
            p = cnt_lr[s] / cnt_l[s]                     # P(m|s)
            w = np.log(np.maximum(p, 1e-12) / np.maximum(pr, 1e-12)) if score == "lift" else p
            w = np.maximum(w, 0.0)
            nz = np.nonzero(w)[0]
            for m in nz:
                A[s, N + off + m] += float(mu * w[m])
                A[N + off + m, s] += float(mu * w[m])
        # 右半边 M -> H
        ph = marg_h / max(marg_h.sum(), 1e-12)
        for m in range(nr):
            if marg_l[m] <= 0:
                continue
            p = cnt_rl[m] / marg_l[m]                    # P(h|m)
            w = np.log(np.maximum(p, 1e-12) / np.maximum(ph, 1e-12)) if score == "lift" else p
            w = np.maximum(w, 0.0)
            nz = np.nonzero(w)[0]
            for h in nz:
                A[N + off + m, n_users + h] += float(mu * w[h])
                A[n_users + h, N + off + m] += float(mu * w[h])

    # 自环 + 对称归一化（与 build_norm_adj 一致）
    A += np.eye(N + n_med, dtype=np.float32)
    d = A.sum(1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    A = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    return torch.from_numpy(A.astype(np.float32)), n_med
    M = np.zeros((N, N), dtype=bool)
    M[:n_users, n_users:] = R > 0
    M[n_users:, :n_users] = R.T > 0
    np.fill_diagonal(M, True)
    return torch.from_numpy(M)


def build_edges(R):
    """GAT 用的边表 (双向 + 自环), 与 build_adj_mask 的邻域集合完全一致"""
    n_users, n_items = R.shape
    N = n_users + n_items
    u, v = np.nonzero(R)
    h = v + n_users
    src = np.concatenate([u, h, np.arange(N)])
    dst = np.concatenate([h, u, np.arange(N)])
    return (torch.from_numpy(src.astype(np.int64)),
            torch.from_numpy(dst.astype(np.int64)))


def build_pairs_from_data(records, n_users, n_items, sym_thresh, herb_thresh):
    """从训练集重新生成 S-S / H-H 共现边（阈值可调，用于敏感性实验）"""
    S = np.zeros((n_users, n_users), dtype=np.int32)
    H = np.zeros((n_items, n_items), dtype=np.int32)
    for syms, herbs in records:
        s = sorted(set(syms))
        h = sorted(set(herbs))
        for a in range(len(s)):
            for b in range(a + 1, len(s)):
                S[s[a], s[b]] += 1
                S[s[b], s[a]] += 1
        for a in range(len(h)):
            for b in range(a + 1, len(h)):
                H[h[a], h[b]] += 1
                H[h[b], h[a]] += 1
    si, sj = np.nonzero(S >= sym_thresh)
    sym_pairs = [(int(a), int(b)) for a, b in zip(si, sj) if a < b]
    hi, hj = np.nonzero(H >= herb_thresh)
    herb_pairs = [(int(a), int(b)) for a, b in zip(hi, hj) if a < b]
    return sym_pairs, herb_pairs


def load_pairs(ds_name, fname):
    """加载 S-S / H-H 视图的边表（与 SMGCN 使用同一批文件，保证公平）"""
    names = fname if isinstance(fname, (list, tuple)) else [fname]
    out = []
    for fn in names:
        path = os.path.join(DATA_ROOT, ds_name, fn)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                p = line.split()
                if len(p) >= 2:
                    out.append((int(p[0]), int(p[1])))
    return out


def build_view_adj(pairs, n):
    """单侧相似图（S-S 或 H-H）的对称归一化邻接，含自环。

    注意：这是「单侧」图，只在该侧节点内传播，不作为整个 N×N 块对角图使用。
    若做成块对角（另一侧补单位阵），融合时会把未经传播的随机嵌入混进来稀释信号，
    实测会让模型直接塌缩到 popularity 解。
    """
    if not pairs:
        return None
    A = np.zeros((n, n), dtype=np.float32)
    for a, b in pairs:
        if a < n and b < n:
            A[a, b] = 1.0
            A[b, a] = 1.0
    A += np.eye(n, dtype=np.float32)
    np.minimum(A, 1.0, out=A)
    d = A.sum(1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    A = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    return torch.from_numpy(A)


class MVGCNLite(nn.Module):
    """MV-GCN-Lite —— 减法路线原型

    相对 LightGCN 只保留三处改动，参数增量可忽略（α: L+1 个 + γ: 2 个标量）：
      1. 层组合权重 α 可学习（LightGCN 用固定均值）—— 替代退化为恒等映射的 SANA
      2. 症状侧叠加 S-S 图传播、草药侧叠加 H-H 图传播，以残差形式加入（γ 初始 0，
         训练起点等价于纯 LightGCN）
      3. DropNode 改为 element-wise（原实现是整行置零 + 通道掩码率分配有 bug）

    删掉的：W_gc / Q 矩阵、tanh 非线性、多通道独立参数（原占 411k）、SANA、多样性损失
    参数量：约 71k（仅嵌入），对比 MV-GCN 的 515k
    """

    name = "MVGCNLite"

    def __init__(self, n_users, n_items, dim, A_norm=None, A_ss=None, A_hh=None,
                 n_layers=3, node_drop=0.0, gamma_init=0.0, no_hh=False,
                 use_views=True, mlp_head=False, A_med=None, n_med=0, **kw):
        super().__init__()
        N = n_users + n_items
        self.n_users, self.n_items = n_users, n_items
        self.n_layers = n_layers
        self.node_drop = node_drop
        self.mlp_head = mlp_head
        self.use_mlp_head = mlp_head
        if mlp_head:                      # 与 MVGCN 一致的轻量非线性评分头
            self.fc1 = nn.Linear(dim, 2 * dim)
            self.fc2 = nn.Linear(2 * dim, dim)
        self.E0 = nn.Parameter(torch.randn(N, dim) * 0.01)
        self.register_buffer("A", A_norm)
        self.has_ss = (A_ss is not None) and use_views
        self.has_hh = (A_hh is not None) and use_views and not no_hh
        if self.has_ss:
            self.register_buffer("A_ss", A_ss)
        if self.has_hh:
            self.register_buffer("A_hh", A_hh)
        self.alpha = nn.Parameter(torch.ones(n_layers + 1))
        self.gamma_s = nn.Parameter(torch.full((1,), float(gamma_init)))
        self.gamma_h = nn.Parameter(torch.full((1,), float(gamma_init)))
        # 证型/治法中介视图（可选）：与主图、S-S/H-H 视图同样以残差形式注入
        self.has_med = A_med is not None
        if self.has_med:
            self.register_buffer("A_med", A_med)
            self.E0_med = nn.Parameter(torch.randn(n_med, dim) * 0.01)
            self.gamma_m = nn.Parameter(torch.full((1,), float(gamma_init)))

    def _med_residual(self, E0):
        """在中介图上传播，取回症状/草药节点的残差 [N, d]"""
        Em = self._prop(self.A_med, torch.cat([E0, self.E0_med], dim=0))
        return Em[:E0.shape[0]]

    def head(self, set_repr):
        return self.fc2(F.relu(self.fc1(set_repr)))

    def _prop(self, A, E0):
        """线性传播 + 可学习层组合（无 W / Q / tanh）"""
        Es = [E0]
        E = E0
        for _ in range(self.n_layers):
            E = A @ E
            Es.append(E)
        w = F.softmax(self.alpha, dim=0).view(-1, 1, 1)
        return (torch.stack(Es, dim=0) * w).sum(0)

    def embeddings(self):
        E0 = F.dropout(self.E0, p=self.node_drop, training=self.training)
        E = self._prop(self.A, E0)
        nu = self.n_users
        if self.has_ss:
            E = torch.cat([E[:nu] + self.gamma_s * self._prop(self.A_ss, E0[:nu]),
                           E[nu:]], dim=0)
        if self.has_hh:
            E = torch.cat([E[:nu],
                           E[nu:] + self.gamma_h * self._prop(self.A_hh, E0[nu:])], dim=0)
        if getattr(self, "has_med", False):
            E = E + self.gamma_m * self._med_residual(E0)
        return E


class MVGCN(nn.Module):
    """MV-GCN —— 完整版，含「逐节点自适应」的 SANA

    与 MVGCNLite 的关键区别：Lite 的层组合权重 α 是全局共享的 L+1 个标量，
    所有节点用同一套深度偏好；SANA 的原意是 *逐节点* 自适应邻域感知：
    每个节点依据自身各层表示，自行决定采信多少浅层/深层信息。
    这在本数据集上有明确动机 —— 症状出现频次差异极大，罕见症状需要更多
    邻域平滑，高频症状则应保留自身特性。

    参数量：E0 (71k) + W_a (64x64=4k)，远小于原实现的 515k。
    """

    name = "MVGCN"

    def __init__(self, n_users, n_items, dim, A_norm=None, A_ss=None, A_hh=None,
                 n_layers=3, node_drop=0.0, use_sana=True, use_views=True,
             sana_calib="none", sana_temp=1.0, mlp_head=False,
             gamma_init=0.0, no_hh=False, A_med=None, n_med=0, **kw):
        super().__init__()
        N = n_users + n_items
        self.n_users, self.n_items = n_users, n_items
        self.n_layers = n_layers
        self.node_drop = node_drop
        self.use_sana = use_sana          # False -> 退化为全局可学习层组合（等同 Lite）
        self.use_views = use_views        # False -> 不用 S-S / H-H 视图残差
        # 尺度校准：对称归一化让 ||A^l E0|| 只有 ||E0|| 的 5%~18%，
        # 而 SANA 的层权重是逐节点的，于是"选浅层的节点"与"选深层的节点"
        # 嵌入范数相差一个量级。点积评分下，top-K 会被范数大的 item 占据，
        # 表现为 P@5 持平但 R@20 下滑。下面两种校准都不改架构，只修尺度：
        #   layer: 把每层整体 rescale 到第 0 层的量级（移除系统性的逐层衰减）
        #   node : 逐节点 rescale 到"均匀组合"的范数（保留 SANA 选的方向，抹掉尺度差）
        self.sana_calib = sana_calib
        self.sana_temp = sana_temp        # softmax 温度，>1 使注意力更平、更接近 Lite
        self.mlp_head = mlp_head          # 在图编码之上加轻量非线性评分头
        self.use_mlp_head = mlp_head
        self.E0 = nn.Parameter(torch.randn(N, dim) * 0.01)
        self.register_buffer("A", A_norm)
        self.has_ss = (A_ss is not None) and use_views
        self.has_hh = (A_hh is not None) and use_views and not no_hh
        if self.has_ss:
            self.register_buffer("A_ss", A_ss)
        if self.has_hh:
            self.register_buffer("A_hh", A_hh)
        # SANA: 逐节点层注意力
        self.W_a = nn.Parameter(torch.empty(dim, dim))
        self.q = nn.Parameter(torch.empty(dim))
        nn.init.xavier_uniform_(self.W_a)
        nn.init.normal_(self.q, std=0.01)
        self.gamma_s = nn.Parameter(torch.full((1,), float(gamma_init)))
        self.gamma_h = nn.Parameter(torch.full((1,), float(gamma_init)))
        # use_sana=False 时改用全局可学习层组合（与 Lite 一致），用于消融
        if not use_sana:
            self.alpha = nn.Parameter(torch.ones(n_layers + 1))
        if mlp_head:                      # 轻量非线性评分头（FordNet 式解码器）
            self.fc1 = nn.Linear(dim, 2 * dim)
            self.fc2 = nn.Linear(2 * dim, dim)
        # 证型/治法中介视图（可选）
        self.has_med = A_med is not None
        if self.has_med:
            self.register_buffer("A_med", A_med)
            self.E0_med = nn.Parameter(torch.randn(n_med, dim) * 0.01)
            self.gamma_m = nn.Parameter(torch.full((1,), float(gamma_init)))

    def _med_residual(self, E0):
        Em = self._prop(self.A_med, torch.cat([E0, self.E0_med], dim=0))
        return Em[:E0.shape[0]]

    def head(self, set_repr):
        return self.fc2(F.relu(self.fc1(set_repr)))

    def _prop(self, A, E0):
        Es = [E0]
        E = E0
        for _ in range(self.n_layers):
            E = A @ E
            Es.append(E)
        Es = torch.stack(Es, dim=0)                    # [L+1, N, d]
        if self.sana_calib == "layer":                 # 逐层整体 rescale 到第 0 层量级
            ln = Es.detach().norm(dim=(1, 2)).clamp(min=1e-8)   # [L+1]
            Es = Es * (ln[0] / ln).view(-1, 1, 1)
        if not self.use_sana:
            w = F.softmax(self.alpha, dim=0).view(-1, 1, 1)
            return (Es * w).sum(0)
        s = torch.tanh(Es @ self.W_a) @ self.q         # [L+1, N]
        w = F.softmax(s / self.sana_temp, dim=0).unsqueeze(-1)  # [L+1, N, 1]
        out = (Es * w).sum(0)
        if self.sana_calib == "node":
            # 注意力只决定方向/层偏好，不允许它通过范数改变点积排名。
            ref = Es.mean(0).detach().norm(dim=1, keepdim=True).clamp(min=1e-8)
            cur = out.norm(dim=1, keepdim=True).clamp(min=1e-8)
            out = out * (ref / cur)
        return out

    def embeddings(self):
        E0 = F.dropout(self.E0, p=self.node_drop, training=self.training)
        E = self._prop(self.A, E0)
        nu = self.n_users
        if self.has_ss:
            E = torch.cat([E[:nu] + self.gamma_s * self._prop(self.A_ss, E0[:nu]),
                           E[nu:]], dim=0)
        if self.has_hh:
            E = torch.cat([E[:nu],
                           E[nu:] + self.gamma_h * self._prop(self.A_hh, E0[nu:])], dim=0)
        if getattr(self, "has_med", False):
            E = E + self.gamma_m * self._med_residual(E0)
        return E


class MVGCNFlow(MVGCN):
    """MV-GCN + 「症状→证候→治法→药物」过程引导分支。

    原 MV-GCN 直接药物分支完整保留；流程分支使用仅由训练集构建的
    S-Z / Z-T / T-H 关系增强语义原型，再逐处方预测证候与治法。最终由
    可学习门控在直接分支与流程分支之间选择，避免中间任务误差强制级联。
    """

    name = "MVGCNFlow"
    use_flow_head = True
    requires_state_snapshot = True

    def __init__(self, n_users, n_items, dim, n_syn, n_trt,
                 A_sz, A_zt, A_th, flow_gamma_init=0.1, **kw):
        super().__init__(n_users=n_users, n_items=n_items, dim=dim, **kw)
        self.n_syn, self.n_trt = n_syn, n_trt
        self.E_syn = nn.Parameter(torch.randn(n_syn, dim) * 0.01)
        self.E_trt = nn.Parameter(torch.randn(n_trt, dim) * 0.01)
        self.register_buffer("A_sz", A_sz)
        self.register_buffer("A_zt", A_zt)
        self.register_buffer("A_th", A_th)

        self.gamma_sz = nn.Parameter(torch.full((1,), float(flow_gamma_init)))
        self.gamma_zt = nn.Parameter(torch.full((1,), float(flow_gamma_init)))
        self.gamma_th = nn.Parameter(torch.full((1,), float(flow_gamma_init)))

        self.syn_head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(),
                                      nn.Linear(dim, n_syn))
        self.trt_head = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(),
                                      nn.Linear(dim, n_trt))
        self.flow_head = nn.Sequential(nn.Linear(3 * dim, 2 * dim), nn.ReLU(),
                                       nn.Linear(2 * dim, dim))
        self.flow_gate = nn.Linear(3 * dim, 1)
        # 训练起点更信任已验证的直接分支，再由数据决定是否提高流程权重。
        nn.init.constant_(self.flow_gate.bias, -2.0)

    @staticmethod
    def _normalise_membership(prob, target=None, teacher_forcing=0.0):
        pred = prob / prob.sum(1, keepdim=True).clamp(min=1e-12)
        if target is None or teacher_forcing <= 0:
            return pred
        gold = target / target.sum(1, keepdim=True).clamp(min=1.0)
        has_gold = (target.sum(1, keepdim=True) > 0).float()
        mixed = teacher_forcing * gold + (1.0 - teacher_forcing) * pred
        return has_gold * mixed + (1.0 - has_gold) * pred

    def semantic_prototypes(self, E):
        """S→Z→T→H 顺序传播，返回关系增强的证候/治法/药物原型。"""
        sym = E[:self.n_users]
        herb = E[self.n_users:]
        syn = self.E_syn + self.gamma_sz * (self.A_sz.t() @ sym)
        trt = self.E_trt + self.gamma_zt * (self.A_zt.t() @ syn)
        herb_flow = herb + self.gamma_th * (self.A_th.t() @ trt)
        return syn, trt, herb_flow

    def flow_forward(self, set_repr, E, syn_targets=None, trt_targets=None,
                     teacher_forcing=0.0):
        syn_proto, trt_proto, herb_flow = self.semantic_prototypes(E)
        direct_repr = self.head(set_repr) if self.use_mlp_head else set_repr
        direct_logits = direct_repr @ E[self.n_users:].t()

        syn_logits = self.syn_head(set_repr)
        syn_w = self._normalise_membership(torch.sigmoid(syn_logits), syn_targets,
                                           teacher_forcing)
        z_syn = syn_w @ syn_proto

        trt_logits = self.trt_head(torch.cat([set_repr, z_syn], dim=1))
        trt_w = self._normalise_membership(torch.sigmoid(trt_logits), trt_targets,
                                           teacher_forcing)
        z_trt = trt_w @ trt_proto

        context = torch.cat([set_repr, z_syn, z_trt], dim=1)
        flow_repr = self.flow_head(context)
        flow_logits = flow_repr @ herb_flow.t()
        gate = torch.sigmoid(self.flow_gate(context))
        logits = (1.0 - gate) * direct_logits + gate * flow_logits
        return {"logits": logits, "direct_logits": direct_logits,
                "flow_logits": flow_logits, "syn_logits": syn_logits,
                "trt_logits": trt_logits, "z_trt": z_trt,
                "herb_flow": herb_flow, "gate": gate}

    @staticmethod
    def contrastive_loss(z_trt, herb_flow, herb_targets, temperature=0.2):
        """治法语义与当前处方真实药物集合的 batch InfoNCE。"""
        denom = herb_targets.sum(1, keepdim=True).clamp(min=1.0)
        z_herb = (herb_targets @ herb_flow) / denom
        q = F.normalize(z_trt, dim=1)
        k = F.normalize(z_herb, dim=1)
        scores = (q @ k.t()) / max(float(temperature), 1e-6)
        labels = torch.arange(scores.shape[0], device=scores.device)
        return F.cross_entropy(scores, labels)


class _GateSANA(nn.Module):
    """门控 SANA（逐节点）: fused = sigmoid(MLP([h_prev; h_last])) * h_last
                          + (1 - sig) * h_prev_aligned, 再 LayerNorm。
    忠实移植 TF model/sana_layer.py：sana 只融合「倒数第二层」与「最后一层」。"""

    def __init__(self, d_prev, d_out, h_dim=256, att_layers=2):
        super().__init__()
        self.d_prev, self.d_out = d_prev, d_out
        layers = []
        for i in range(att_layers):
            ind = (d_prev + d_out) if i == 0 else h_dim
            layers += [nn.Linear(ind, h_dim), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)               # [N, h_dim]
        self.W_att = nn.Linear(h_dim + d_out, 1)         # 打分 concat([z; h_last])
        self.W_h = nn.Linear(d_prev, d_out) if d_prev != d_out else None

    def forward(self, h_prev, h_last):
        z = self.mlp(torch.cat([h_prev, h_last], dim=-1))     # [N, h_dim]
        score = self.W_att(torch.cat([z, h_last], dim=-1))    # [N, 1]
        a = torch.sigmoid(score)
        h_aligned = self.W_h(h_prev) if self.W_h is not None else h_prev
        fused = a * h_last + (1.0 - a) * h_aligned
        return F.layer_norm(fused, (self.d_out,))


class SMGCN_MVANA(nn.Module):
    """SMGCN-MVANA —— 忠实移植 TF 重写版（用户 2026-09-29 重写的 model/*.py）

    架构（对应 TF: multi_channel_encoder.py + sana_layer.py + vlr_aggregation.py + SMGCN_MVANA.py）:
      1. 多通道并行 GCN 编码器: sz_c 个独立权重的 GraphSAGE 式通道
         （tanh 邻域变换 + 自连接拼接 + l2 归一化），每个通道对初始嵌入施加不同 DropNode 掩码
         （MFM: 掩码率按通道线性铺开），以强迫通道学习不同视角。
      2. 门控 SANA: 逐节点 sigmoid 门控融合「倒数第二层」与「最后一层」。
      3. 配对图残差: S-S 图 / H-H 图分别 tanh 投影后加性融合进 user/item 通道嵌入。
      4. 多通道 VLR KV-Attention 集合聚合: 每个通道一组可学习 Key/Value + 可学习 query 向量，
         仅在症状集合内做注意力，各通道结果拼接后 LayerNorm 得到 [B, sz_c*d_kv]。
      5. 预测 MLP 头: [sz_c*d_kv] -> [d_out] (ReLU)，再与 item 嵌入点积。

    与统一协议其余模型唯一刻意差异：训练损失沿用统一协议（频率加权多标签 BCE + e0 L2），
    不加 MVANA 的「通道去相关多样性损失」（该正则器基线模型都没有，为公平对比不加）。
    """

    name = "SMGCN_MVANA"
    use_vlr = True           # 训练/评测回路改用 set_representation()
    use_mlp_head = True      # _scores 走 MLP 头

    def __init__(self, n_users, n_items, dim, A_norm=None, A_ss=None, A_hh=None,
                 n_layers=3, node_drop=0.0, **kw):
        super().__init__()
        N = n_users + n_items
        self.n_users, self.n_items = n_users, n_items
        self.dim = dim                              # E0 维度 (64)
        self.n_layers = 2                           # GCN 层数 (TF len([128,256])=2)
        self.weight_size = [128, 256]               # 每层 GCN 输出维度 (TF layer_size)
        self.d_out = self.weight_size[-1]           # 256
        self.sz_c = int(kw.get("sz_c", 2))          # 通道数
        self.d_kv = int(kw.get("d_kv", 64))         # VLR KV 维度
        self.mask_rate = float(kw.get("mask_rate", 0.2))
        self.sana_alpha = float(kw.get("sana_alpha", 0.2))   # 末两层融合权重
        self.mfm = kw.get("mfm", "MFM")
        self.node_drop = node_drop
        self.use_vlr_attention = not bool(kw.get("no_vlr", False))
        self.use_direct_residual = bool(kw.get("direct_residual", False))
        self.use_sana = not bool(kw.get("no_sana", False))
        self.use_views = not bool(kw.get("no_views", False))
        self.use_hh = not bool(kw.get("no_hh", False))

        # MFM 各通道掩码率（与 TF 一致: mask_rate*(c+1)/sz_c）
        if self.mfm == "MFM":
            self.all_mask_rates = [self.mask_rate * (c + 1) / self.sz_c
                                   for c in range(self.sz_c)]
        else:
            self.all_mask_rates = [self.mask_rate] * self.sz_c

        self.E0 = nn.Parameter(torch.randn(N, dim) * 0.01)
        self.register_buffer("A", A_norm)
        self.has_ss = self.use_views and A_ss is not None
        self.has_hh = self.use_views and self.use_hh and A_hh is not None
        if self.has_ss:
            self.register_buffer("A_ss", A_ss)
        if self.has_hh:
            self.register_buffer("A_hh", A_hh)

        # 各通道独立 GCN 权重建表（user/item 侧），扁平存为 ParameterList
        self.W_gc_u, self.b_gc_u, self.Q_u = nn.ParameterList(), nn.ParameterList(), nn.ParameterList()
        self.W_gc_i, self.b_gc_i, self.Q_i = nn.ParameterList(), nn.ParameterList(), nn.ParameterList()
        for c in range(self.sz_c):
            for k in range(self.n_layers):
                in_d = self.dim if k == 0 else self.weight_size[k - 1]
                out_d = self.weight_size[k]
                for Wg, bg, Q in ((self.W_gc_u, self.b_gc_u, self.Q_u),
                                  (self.W_gc_i, self.b_gc_i, self.Q_i)):
                    Wg.append(nn.Parameter(torch.empty(2 * in_d, out_d)))
                    bg.append(nn.Parameter(torch.zeros(1, out_d)))
                    Q.append(nn.Parameter(torch.empty(in_d, in_d)))
                    nn.init.xavier_uniform_(Wg[-1]); nn.init.xavier_uniform_(Q[-1])

        # 门控 SANA（每通道, user/item 侧）
        d_prev = self.weight_size[-2]
        self.sana_u, self.sana_i = nn.ModuleList(), nn.ModuleList()
        for c in range(self.sz_c):
            self.sana_u.append(_GateSANA(d_prev, self.d_out))
            self.sana_i.append(_GateSANA(d_prev, self.d_out))

        # 配对图投影矩阵 M_user / M_item: [dim, d_out]
        if self.has_ss:
            self.M_user = nn.Linear(dim, self.d_out)
            nn.init.xavier_uniform_(self.M_user.weight)
        if self.has_hh:
            self.M_item = nn.Linear(dim, self.d_out)
            nn.init.xavier_uniform_(self.M_item.weight)

        # VLR: 各通道共享 KV 权重 + 每通道可学习 query
        self.W_k = nn.Parameter(torch.empty(self.d_out, self.d_kv))
        self.W_v = nn.Parameter(torch.empty(self.d_out, self.d_kv))
        self.queries = nn.Parameter(torch.empty(self.sz_c, self.d_kv))
        nn.init.xavier_uniform_(self.W_k); nn.init.xavier_uniform_(self.W_v)
        nn.init.normal_(self.queries, std=0.01)

        # VLR 消融：各通道症状嵌入先均值，再投影到与 VLR 相同输出维度，
        # 保证预测头参数量和维度不因聚合器切换而改变。
        self.mean_pool_proj = nn.Linear(self.d_out, self.sz_c * self.d_kv)
        nn.init.xavier_uniform_(self.mean_pool_proj.weight)

        if self.use_direct_residual:
            prior = kw.get("direct_prior")
            if prior is None or tuple(prior.shape) != (n_users, n_items):
                raise ValueError("direct_residual requires [n_users, n_items] prior")
            self.direct_weight = nn.Parameter(prior.float().clone())
            # 冻结先验：把「症状—药物条件概率」作为不可学习的可靠下界，
            # 让训练只能通过 gate/scale 与图分支提升排序，保证不会低于先验。
            if bool(kw.get("freeze_direct", False)):
                self.direct_weight.requires_grad_(False)
            self.direct_log_scale = nn.Parameter(torch.zeros(1))
            # 初始时主要信任可解释的 P(h|s)，图分支从小残差开始学习。
            gate0 = min(max(float(kw.get("direct_gate_init", 0.01)), 1e-5), 1-1e-5)
            self.residual_gate_logit = nn.Parameter(torch.tensor([
                np.log(gate0 / (1.0 - gate0))], dtype=torch.float32))

        # 预测 MLP 头: [sz_c*d_kv] -> [d_out]
        self.fc_pred = nn.Linear(self.sz_c * self.d_kv, self.d_out)
        nn.init.xavier_uniform_(self.fc_pred.weight)

    # ---- 工具 ----
    def _init_masked(self, c):
        rate = self.all_mask_rates[c]
        if self.training and rate > 0:
            mask = (torch.rand(self.E0.shape[0], 1, device=self.E0.device) > rate).float()
            return self.E0 * mask
        return self.E0

    def _gcn_prop(self, E0, c, side):
        Wg = self.W_gc_u if side == "u" else self.W_gc_i
        bg = self.b_gc_u if side == "u" else self.b_gc_i
        Q = self.Q_u if side == "u" else self.Q_i
        E = E0
        Es = [E]
        for k in range(self.n_layers):
            agg = self.A @ E
            agg = torch.tanh(agg @ Q[c * self.n_layers + k])
            concat = torch.cat([E, agg], dim=1)
            E = torch.tanh(concat @ Wg[c * self.n_layers + k] + bg[c * self.n_layers + k])
            E = F.dropout(E, p=self.node_drop, training=self.training)
            E = F.normalize(E, p=2, dim=1)
            Es.append(E)
        if self.use_sana and self.sana_alpha > 0 and self.n_layers >= 2:
            sana = self.sana_u[c] if side == "u" else self.sana_i[c]
            h_fused = sana(Es[-2], Es[-1])
            E = self.sana_alpha * h_fused + (1.0 - self.sana_alpha) * Es[-1]
        return E

    def embeddings(self):
        ua_list, ia_list = [], []
        for c in range(self.sz_c):
            E0m = self._init_masked(c)
            ua_full = self._gcn_prop(E0m, c, "u")   # [N, d_out]
            ia_full = self._gcn_prop(E0m, c, "i")   # [N, d_out]
            ua_list.append(ua_full[:self.n_users])
            ia_list.append(ia_full[self.n_users:])
        # 配对图残差（S-S / H-H tanh 投影）
        user_pair = (torch.tanh(self.M_user(self.A_ss @ self.E0[:self.n_users]))
                     if self.has_ss else 0.0)
        item_pair = (torch.tanh(self.M_item(self.A_hh @ self.E0[self.n_users:]))
                     if self.has_hh else 0.0)
        fused_ua = [ua_list[c] + user_pair for c in range(self.sz_c)]
        fused_ia = [ia_list[c] + item_pair for c in range(self.sz_c)]
        self._fused_ua = fused_ua
        self._fused_ia = fused_ia
        ua_mean = torch.stack(fused_ua, 0).mean(0)
        ia_mean = torch.stack(fused_ia, 0).mean(0)
        return torch.cat([ua_mean, ia_mean], dim=0)   # [N, d_out]

    def set_representation(self, P):
        """P: [B, n_users] 症状集合矩阵（统一协议为均值池化矩阵；(P>0) 即存在遮罩）"""
        if not self.use_vlr_attention:
            ua = torch.stack(self._fused_ua, dim=0).mean(0)  # [n_users, d_out]
            pooled = P @ ua                                  # P 已按症状数归一化
            out = self.mean_pool_proj(pooled)
            return F.layer_norm(out, (self.sz_c * self.d_kv,))
        present = (P > 0).float()                       # [B, n_users]
        chans = []
        for c in range(self.sz_c):
            ua = self._fused_ua[c]
            K = ua @ self.W_k                          # [n_users, d_kv]
            V = ua @ self.W_v                          # [n_users, d_kv]
            node_logits = (K @ self.queries[c]) / (self.d_kv ** 0.5)  # [n_users]
            L = present * node_logits.unsqueeze(0)      # [B, n_users]
            L = L.masked_fill(present < 0.5, -1e9)
            att = torch.softmax(L, dim=1)
            att = att * present
            att = att / att.sum(1, keepdim=True).clamp(min=1e-12)
            chans.append(att @ V)                       # [B, d_kv]
        set_repr = torch.cat(chans, dim=1)              # [B, sz_c*d_kv]
        return F.layer_norm(set_repr, (self.sz_c * self.d_kv,))

    def head(self, set_repr):
        return F.relu(self.fc_pred(set_repr))           # [B, d_out]

    def fuse_direct(self, P, graph_logits):
        direct = P @ self.direct_weight
        # 标准化不改变每个病例内的药物排序，但让 BCE 的数值尺度稳定。
        mean = direct.mean(1, keepdim=True)
        std = direct.std(1, keepdim=True, unbiased=False).clamp(min=1e-6)
        direct = (direct - mean) / std
        scale = torch.exp(self.direct_log_scale.clamp(-3.0, 3.0))
        gate = torch.sigmoid(self.residual_gate_logit)
        return scale * direct + gate * graph_logits


# ----------------------------------------------------------------------
# 评估 (与 batch_test.py 一致的指标定义)
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(E, P, Y, n_users, ks=KS, model=None, core=None):
    """E: [n_users+n_items, d]; P: 池化矩阵 [N, n_users]; Y: 多热标签 [N, n_items]

    core: 可选的 bool 数组 [n_items]，标记「核心药」。
    临床处方中少数高频药（本数据集 top-20 占 36% 开方量）几乎每张方都出现，
    它们在 P@K / R@K 里贡献了大部分命中，却完全不体现辨证差异。
    屏蔽后只评估「差异化用药」部分——这才是模型真正需要推理的部分。
    """
    set_repr = model.set_representation(P) if getattr(model, "use_vlr", False) \
        else P @ E[:n_users]                                     # [N, d] / [N, sz_c*d_kv]
    scores = _scores(E, set_repr, n_users, model, P=P)           # [N, n_items]
    Yt = torch.from_numpy(Y).to(E.device)
    if core is not None:
        cm = torch.from_numpy(np.asarray(core)).bool().to(E.device)
        scores = scores.masked_fill(cm[None, :], -1e9)
        Yt = Yt.masked_fill(cm[None, :], 0.0)
    top = torch.topk(scores, max(ks), dim=1).indices             # [N, max_k]
    n_gt = Yt.sum(1).clamp(min=1e-9)
    if core is not None:            # 剔除后 gold 为空的记录没有可评估对象, 不计入
        keep = Yt.sum(1) > 0
        Yt, top, n_gt = Yt[keep], top[keep], n_gt[keep]
    out = {}
    for k in ks:
        hits = Yt.gather(1, top[:, :k]).sum(1)                   # [N]
        p = hits / k
        r = hits / n_gt
        f = torch.where(p + r > 0, 2 * p * r / (p + r).clamp(min=1e-9), torch.zeros_like(p))
        out[f"P@{k}"] = float(p.mean())
        out[f"R@{k}"] = float(r.mean())
        out[f"F1@{k}"] = float(f.mean())
    return out


def evaluate_masked(E, P, Y, n_users, mask, ks=KS, model=None, core=None):
    """在 mask 指定的测试子集上评估（用于 seen / unseen 症状组合分层）"""
    with torch.no_grad():
        set_repr = model.set_representation(P) if getattr(model, "use_vlr", False) \
            else P @ E[:n_users]
        scores = _scores(E, set_repr, n_users, model, P=P)
        if core is not None:
            cm = torch.from_numpy(np.asarray(core)).bool().to(E.device)
            scores = scores.masked_fill(cm[None, :], -1e9)
        top = torch.topk(scores, max(ks), dim=1).indices
    msk = torch.from_numpy(mask).long().to(E.device)
    Yt = torch.from_numpy(Y).to(E.device).index_select(0, msk)
    top = top.index_select(0, msk)
    if core is not None:
        cm = torch.from_numpy(np.asarray(core)).bool().to(E.device)
        Yt = Yt.masked_fill(cm[None, :], 0.0)
    n_gt = Yt.sum(1).clamp(min=1e-9)
    if core is not None:
        keep = Yt.sum(1) > 0
        Yt, top, n_gt = Yt[keep], top[keep], n_gt[keep]
    out = {}
    for k in ks:
        hits = Yt.gather(1, top[:, :k]).sum(1)
        p = hits / k
        r = hits / n_gt
        f = torch.where(p + r > 0, 2 * p * r / (p + r).clamp(min=1e-9), torch.zeros_like(p))
        out[f"P@{k}"] = float(p.mean())
        out[f"R@{k}"] = float(r.mean())
        out[f"F1@{k}"] = float(f.mean())
    out["n"] = int(len(mask))
    return out


@torch.no_grad()
def evaluate_flow_aux(model, E, P, syn_Y, trt_Y):
    """评估中间任务与门控，不参与最佳 epoch 选择。"""
    out = model.flow_forward(P @ E[:model.n_users], E)

    def micro_f1(logits, target):
        pred = torch.sigmoid(logits) >= 0.5
        gold = target >= 0.5
        tp = (pred & gold).sum().float()
        fp = (pred & ~gold).sum().float()
        fn = (~pred & gold).sum().float()
        return float((2 * tp / (2 * tp + fp + fn).clamp(min=1.0)).item())

    syn_t = torch.from_numpy(syn_Y).to(E.device)
    trt_t = torch.from_numpy(trt_Y).to(E.device)
    return {"syn_micro_f1": micro_f1(out["syn_logits"], syn_t),
            "trt_micro_f1": micro_f1(out["trt_logits"], trt_t),
            "gate_mean": float(out["gate"].mean().item())}


def to_idx(records):
    return [torch.tensor(s, dtype=torch.long) for s, _ in records]


# ----------------------------------------------------------------------
# 单次运行
# ----------------------------------------------------------------------
def train_one_run(model_name, tr, val, test, n_users, n_items, R, w, seed, args, views=None,
                  med=None, core=None, flow=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    dev = DEVICE

    A_norm = build_norm_adj(R, self_loop=bool(args.self_loop)) if model_name in (
        "NGCF", "LightGCN", "MVGCNLite", "MVGCN", "MVGCNFlow",
        "SMGCN", "SMGCN_MVANA") else None
    kwargs = dict(n_users=n_users, n_items=n_items, dim=args.dim)
    kwargs["n_layers"] = args.layers
    if A_norm is not None:
        kwargs["A_norm"] = A_norm.to(dev)
    if model_name in ("MVGCNLite", "MVGCN", "MVGCNFlow", "SMGCN_MVANA") and views:
        kwargs["A_ss"] = views[0].to(dev) if views[0] is not None else None
        kwargs["A_hh"] = views[1].to(dev) if views[1] is not None else None
        kwargs["node_drop"] = getattr(args, "node_drop", 0.0)
        kwargs["gamma_init"] = getattr(args, "gamma_init", 0.0)
        kwargs["no_hh"] = bool(getattr(args, "no_hh", False))
        if model_name == "MVGCNLite":
            kwargs["mlp_head"] = bool(getattr(args, "mlp_head", False))
            kwargs["use_views"] = not bool(getattr(args, "no_views", False))
    if model_name == "SMGCN_MVANA":
        kwargs["sz_c"] = int(getattr(args, "sz_c", 2))
        kwargs["d_kv"] = int(getattr(args, "d_kv", 64))
        kwargs["mask_rate"] = float(getattr(args, "mask_rate", 0.2))
        kwargs["sana_alpha"] = float(getattr(args, "sana_alpha", 0.2))
        kwargs["mfm"] = getattr(args, "mfm", "MFM")
        kwargs["no_vlr"] = bool(getattr(args, "no_vlr", False))
        kwargs["no_sana"] = bool(getattr(args, "no_sana", False))
        kwargs["no_views"] = bool(getattr(args, "no_views", False))
        kwargs["direct_residual"] = bool(getattr(args, "direct_residual", False))
        kwargs["freeze_direct"] = bool(getattr(args, "freeze_direct", False))
        if kwargs["direct_residual"]:
            prior = build_direct_prior(
                tr, n_users, n_items,
                shrink=float(getattr(args, "direct_shrink", 5.0)))
            kwargs["direct_prior"] = torch.from_numpy(prior).to(dev)
            kwargs["direct_gate_init"] = float(
                getattr(args, "direct_gate_init", 0.01))
    if model_name == "SMGCN" and views:
        kwargs["A_ss"] = (views[0].to(dev) if views[0] is not None else None)
    if model_name in ("MVGCN", "MVGCNFlow"):  # 模块消融 / 改进开关
        kwargs["use_sana"] = not getattr(args, "no_sana", False)
        kwargs["use_views"] = not getattr(args, "no_views", False)
        kwargs["sana_calib"] = getattr(args, "sana_calib", "none")
        kwargs["sana_temp"] = getattr(args, "sana_temp", 1.0)
        kwargs["mlp_head"] = getattr(args, "mlp_head", False)
    if model_name in ("MVGCNLite", "MVGCN") and med is not None:
        kwargs["A_med"], kwargs["n_med"] = med[0].to(dev), med[1]
    if model_name == "MVGCNFlow":
        if flow is None:
            raise ValueError("MVGCNFlow requires aligned syndrome/treatment labels")
        kwargs.update(n_syn=flow["n_syn"], n_trt=flow["n_trt"],
                      A_sz=flow["A_sz"].to(dev), A_zt=flow["A_zt"].to(dev),
                      A_th=flow["A_th"].to(dev),
                      flow_gamma_init=float(getattr(args, "flow_gamma_init", 0.1)))
    if model_name == "GAT":
        kwargs["edge_src"], kwargs["edge_dst"] = build_edges(R)
    if model_name == "Popularity":
        pop = np.zeros(n_items, dtype=np.float32)
        for _syms, herbs in tr:
            pop[herbs] += 1.0
        kwargs["item_scores"] = torch.from_numpy(pop).to(dev)
    model = {"Popularity": Popularity, "MF": MF, "NGCF": NGCF,
             "LightGCN": LightGCN, "GAT": GAT,
             "MVGCNLite": MVGCNLite, "MVGCN": MVGCN, "MVGCNFlow": MVGCNFlow,
             "MLP": MLP, "SMGCN": SMGCN,
             "SMGCN_MVANA": SMGCN_MVANA}[model_name](**kwargs).to(dev)

    weighted = args.weighting in ("invfreq", "invfreq_all", "invfreq_pos")
    W = torch.from_numpy(w if weighted else np.ones(n_items, dtype=np.float32)).to(dev)
    Ytr = make_labels(tr, n_items)                # 全草药表多热标签
    P_tr = build_pool(tr, n_users).to(dev)        # 症状集合平均池化矩阵
    P_val, val_Y = build_pool(val, n_users).to(dev), make_labels(val, n_items)
    P_test, test_Y = build_pool(test, n_users).to(dev), make_labels(test, n_items)
    Ytr_t = torch.from_numpy(Ytr).to(dev)
    tail_core = None
    tail_n = int(getattr(args, "tail_exclude", 20))
    if tail_n > 0:
        train_freq = Ytr.sum(0)
        tail_core = np.zeros(n_items, dtype=bool)
        tail_core[np.argsort(-train_freq)[:min(tail_n, n_items)]] = True
    if model_name == "MVGCNFlow":
        syn_tr = torch.from_numpy(make_multilabel(flow["tr_syn"], flow["n_syn"])).to(dev)
        trt_tr = torch.from_numpy(make_multilabel(flow["tr_trt"], flow["n_trt"])).to(dev)
        syn_test_np = make_multilabel(flow["test_syn"], flow["n_syn"])
        trt_test_np = make_multilabel(flow["test_trt"], flow["n_trt"])

    n_train = len(tr)
    if model_name == "Popularity":
        model.eval()
        E = model.embeddings()
        vm = evaluate(E, P_val, val_Y, n_users, model=model, core=core)
        res = evaluate(E, P_test, test_Y, n_users, model=model, core=core)
        res.update({"_best_epoch": 0, "_val_R@20": vm["R@20"],
                    "_train_time_s": 0.0, "_n_train": n_train})
        if tail_core is not None:
            res[f"_tail_exclude_top{tail_n}"] = evaluate(
                E, P_test, test_Y, n_users, model=model, core=tail_core)
        tr_sets = {tuple(sorted(s)) for s, _ in tr}
        groups = {
            "_seen": np.array([i for i, (s, _) in enumerate(test)
                               if tuple(sorted(s)) in tr_sets], dtype=np.int64),
            "_unseen": np.array([i for i, (s, _) in enumerate(test)
                                 if tuple(sorted(s)) not in tr_sets], dtype=np.int64),
        }
        sizes = np.array([len(s) for s, _ in test], dtype=np.int64)
        groups.update(_s1=np.where(sizes == 1)[0].astype(np.int64),
                      _s2=np.where(sizes == 2)[0].astype(np.int64),
                      _s3p=np.where(sizes >= 3)[0].astype(np.int64))
        for tag, idx in groups.items():
            if len(idx):
                res[tag] = evaluate_masked(E, P_test, test_Y, n_users, idx,
                                           model=model, core=core)
        return res

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = {"val_R20": -1.0, "epoch": 0, "test": None, "val": None}
    no_improve = 0
    early_stopped = False
    t0 = time.time()
    last_loss = float("nan")
    convergence = []

    # 所有模型都记录 epoch 0，便于比较达到最终性能所需的轮次和时间。
    # 只有直连分支把 epoch 0 纳入选模：它由训练集条件概率初始化，允许
    # 后续神经残差训练回退到可靠先验，但不改变其他模型原有的选模语义。
    model.eval()
    with torch.no_grad():
        E_init = model.embeddings()
    vm0 = evaluate(E_init, P_val, val_Y, n_users, model=model, core=core)
    convergence.append({"epoch": 0, "elapsed_s": round(time.time() - t0, 4),
                        "val_P@5": vm0["P@5"], "val_R@20": vm0["R@20"]})
    if getattr(model, "use_direct_residual", False):
        tm0 = evaluate(E_init, P_test, test_Y, n_users, model=model, core=core)
        snap0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best = {"val_R20": vm0["R@20"], "epoch": 0, "test": tm0,
                "val": vm0, "E_best": E_init.clone(), "head_snap": snap0}

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(n_train)
        for st in range(0, n_train, args.batch):
            batch = torch.from_numpy(perm[st:st + args.batch]).long().to(dev)
            E = model.embeddings()
            if getattr(model, "use_vlr", False):
                set_repr = model.set_representation(P_tr[batch])            # [B, sz_c*d_kv]
            else:
                set_repr = P_tr[batch] @ E[:n_users]                        # [B, d]
            y = Ytr_t[batch]
            flow_out = None
            if model_name == "MVGCNFlow":
                teacher = float(getattr(args, "flow_teacher", 0.5)) * max(
                    0.0, 1.0 - float(epoch - 1) / max(args.epochs, 1))
                flow_out = model.flow_forward(
                    set_repr, E, syn_tr[batch], trt_tr[batch], teacher_forcing=teacher)
                logits = flow_out["logits"]
            else:
                logits = _scores(E, set_repr, n_users, model, P=P_tr[batch])
            bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            if args.weighting == "invfreq_pos":
                # 只放大稀有药物的正样本；若同时放大它们数量庞大的负样本，
                # 会反向把稀有药 logits 压低并退化成 popularity 排序。
                element_w = 1.0 + y * (W - 1.0)
            elif args.weighting in ("invfreq", "invfreq_all"):
                element_w = W                       # 旧协议，保留用于复现实验
            else:
                element_w = torch.ones_like(W)
            if args.loss == "bce":
                loss = (bce * element_w).sum() / (len(batch) * n_items)     # 加权 BCE
            elif args.loss == "focal":
                p = torch.sigmoid(logits)
                p_t = p * y + (1.0 - p) * (1.0 - y)                        # 正样本 p / 负样本 1-p
                focal_w = (1.0 - p_t).clamp(min=0.0).pow(args.focal_gamma)  # 聚焦难样本
                loss = (bce * focal_w * element_w).sum() / (len(batch) * n_items)
            elif args.loss == "hard_bce":                                   # top-K 难负样本 BCE
                k = min(args.topk, n_items - 1)
                neg_only = logits.masked_fill(y > 0.5, -1e9)                # 仅保留负样本
                topk_neg, _ = torch.topk(neg_only, k, dim=1)                # [B, k] 最难负样本
                pos = logits[y > 0.5]                                       # 全部正样本
                if pos.numel() > 0:
                    all_logits = torch.cat([pos, topk_neg.reshape(-1)])
                    all_labels = torch.cat([torch.ones_like(pos),
                                            torch.zeros_like(topk_neg.reshape(-1))])
                    loss = F.binary_cross_entropy_with_logits(all_logits, all_labels)
                else:
                    loss = (bce * element_w).sum() / (len(batch) * n_items)
            if flow_out is not None:
                loss = loss + float(getattr(args, "lambda_syn", 0.2)) * \
                    F.binary_cross_entropy_with_logits(flow_out["syn_logits"], syn_tr[batch])
                loss = loss + float(getattr(args, "lambda_trt", 0.2)) * \
                    F.binary_cross_entropy_with_logits(flow_out["trt_logits"], trt_tr[batch])
                lambda_con = float(getattr(args, "lambda_con", 0.05))
                if lambda_con > 0:
                    loss = loss + lambda_con * model.contrastive_loss(
                        flow_out["z_trt"], flow_out["herb_flow"], y,
                        temperature=float(getattr(args, "contrast_temp", 0.2)))
            # L2 正则：l2_target=e0 时加在第 0 层嵌入上（LightGCN / NGCF 的标准做法）。
            # 这一点对公平性至关重要：对称归一化会让 ||A^l E0||^2 收缩到 E0 的 ~3%，
            # 若把 L2 加在传播后的 E 上，图模型实际受到的惩罚只有 MF 的 8.5%，
            # 等于系统性让图模型欠正则化。
            if getattr(args, "l2_target", "e0") == "e0" and hasattr(model, "E0"):
                e0sq = model.E0.pow(2).sum()
                for pn, pp in model.named_parameters():   # 中介/语义节点嵌入同样计入
                    if (pn.startswith("E0_med") or pn.startswith("E_syn") or
                            pn.startswith("E_trt")):
                        e0sq = e0sq + pp.pow(2).sum()
                loss = loss + args.reg * 0.5 * e0sq / len(batch)
            else:
                loss = loss + args.reg * 0.5 * (E.pow(2).sum()) / len(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # 仅在 --trace 时才把 loss 拷回主机：GPU 上每次 float() 都会强制同步，
            # 打断异步流水线，对小模型来说是主要开销来源
            if args.trace:
                last_loss = float(loss.detach())

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                E = model.embeddings()
            vm = evaluate(E, P_val, val_Y, n_users, model=model, core=core)
            convergence.append({"epoch": epoch,
                                "elapsed_s": round(time.time() - t0, 4),
                                "val_P@5": vm["P@5"],
                                "val_R@20": vm["R@20"]})
            if args.trace:
                print(f"    ep{epoch:3d} loss={last_loss:.4f} val_P@5={vm['P@5']:.4f} "
                      f"val_R@20={vm['R@20']:.4f}", flush=True)
            if vm["R@20"] > best["val_R20"]:
                tm = evaluate(E, P_test, test_Y, n_users, model=model, core=core)
                # MLP 的评分头参数必须随嵌入一起快照，否则分层评估会
                # 用「最佳 epoch 的嵌入」配「最后 epoch 的 head」
                snap = None
                if (getattr(model, "use_mlp_head", False) or
                        getattr(model, "requires_state_snapshot", False)):
                    # 注意：model.head 是方法不是模块，必须快照整个 state_dict
                    snap = {k: v.detach().clone()
                            for k, v in model.state_dict().items()}
                best = {"val_R20": vm["R@20"], "epoch": epoch, "test": tm, "val": vm,
                        "E_best": E.clone(), "head_snap": snap}
                no_improve = 0
            else:
                no_improve += 1
                if args.patience and no_improve >= args.patience:
                    print(f"    [early-stop] ep{epoch} val_R@20 连续 {no_improve} 次评估无提升", flush=True)
                    early_stopped = True
                    break

    res = dict(best["test"])
    res["_best_epoch"] = best["epoch"]
    res["_val_R@20"] = best["val_R20"]
    res["_train_time_s"] = round(time.time() - t0, 1)
    res["_n_train"] = n_train
    res["_convergence"] = convergence

    # ---- 直连门控诊断：区分「训练末态」与「选模回退后的快照」 ----
    # 选模会把参数恢复到 best_epoch 的快照；若只看恢复后的值，会误判成
    # gate 从未更新。这里先记录训练末态，用于判断图分支是否被训练推开。
    if hasattr(model, "residual_gate_logit"):
        res["_direct_graph_gate_trained"] = float(
            torch.sigmoid(model.residual_gate_logit.detach()).item())
        res["_direct_scale_trained"] = float(
            torch.exp(model.direct_log_scale.detach().clamp(-3.0, 3.0)).item())

    # ---- seen / unseen 症状组合分层（检验多视图图是否帮助泛化到新组合）----
    # ---- 以及按症状个数 |S| 分层（检验图传播在多症状复杂病例上是否有价值）----
    if "E_best" in best:
        if best.get("head_snap") is not None:
            model.load_state_dict(best["head_snap"])
        tr_sets = {tuple(sorted(s)) for s, _ in tr}
        seen_idx = np.array([i for i, (s, _) in enumerate(test)
                             if tuple(sorted(s)) in tr_sets], dtype=np.int64)
        unseen_idx = np.array([i for i, (s, _) in enumerate(test)
                               if tuple(sorted(s)) not in tr_sets], dtype=np.int64)
        if len(seen_idx):
            res["_seen"] = evaluate_masked(best["E_best"], P_test, test_Y,
                                           n_users, seen_idx, model=model, core=core)
        if len(unseen_idx):
            res["_unseen"] = evaluate_masked(best["E_best"], P_test, test_Y,
                                             n_users, unseen_idx, model=model, core=core)
        sizes = np.array([len(s) for s, _ in test], dtype=np.int64)
        for tag, sel in (("_s1", sizes == 1), ("_s2", sizes == 2), ("_s3p", sizes >= 3)):
            idx = np.where(sel)[0].astype(np.int64)
            if len(idx):
                res[tag] = evaluate_masked(best["E_best"], P_test, test_Y,
                                           n_users, idx, model=model, core=core)
        if model_name == "MVGCNFlow":
            res["_flow_aux"] = evaluate_flow_aux(
                model, best["E_best"], P_test, syn_test_np, trt_test_np)
        if tail_core is not None:
            res[f"_tail_exclude_top{tail_n}"] = evaluate(
                best["E_best"], P_test, test_Y, n_users,
                model=model, core=tail_core)
    # ---- 结构参数探针：多视图残差 gamma 是否真的离开 0 ----
    if hasattr(model, "gamma_s"):
        res["_gamma_s"] = float(model.gamma_s.detach().item())
    if hasattr(model, "gamma_h"):
        res["_gamma_h"] = float(model.gamma_h.detach().item())
    if hasattr(model, "alpha"):
        res["_alpha"] = [float(x) for x in
                         torch.softmax(model.alpha.detach(), dim=0).cpu().numpy()]
    if hasattr(model, "gamma_m"):
        res["_gamma_m"] = float(model.gamma_m.detach().item())
    if hasattr(model, "gamma_sz"):
        res["_gamma_sz"] = float(model.gamma_sz.detach().item())
        res["_gamma_zt"] = float(model.gamma_zt.detach().item())
        res["_gamma_th"] = float(model.gamma_th.detach().item())
    if hasattr(model, "residual_gate_logit"):
        res["_direct_graph_gate"] = float(
            torch.sigmoid(model.residual_gate_logit.detach()).item())
        res["_direct_scale"] = float(
            torch.exp(model.direct_log_scale.detach().clamp(-3.0, 3.0)).item())
    return res


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    global DEVICE
    ap.add_argument("--dataset", default="smgcn", choices=["smgcn", "TCM_PD", "lung", "sdpr"])
    ap.add_argument("--model", default="all",
                    choices=sorted(set(MODEL_NAMES) | set(MODEL_GROUPS)),
                    help="单模型，或 basic=MLP,NGCF,SMGCN；"
                         "comparison=显式流行度+八个模型+完整移植版；all=保留原六模型组")
    ap.add_argument("--views", default="ss,hh",
                    help="MVGCNLite 启用的单侧视图: ss / hh / ss,hh / none")
    ap.add_argument("--sym_thresh", type=int, default=0,
                    help="S-S 共现阈值; 0=用默认值5。共现边始终仅从实际训练子集生成")
    ap.add_argument("--herb_thresh", type=int, default=0,
                    help="H-H 共现阈值; 0=用默认值40。共现边始终仅从实际训练子集生成")
    ap.add_argument("--layers", type=int, default=3, help="GCN 传播层数")
    ap.add_argument("--node_drop", type=float, default=0.0)
    ap.add_argument("--gamma_init", type=float, default=0.0,
                    help="多视图残差 gamma 的初值（0=从零开始学；1=一开始就信任视图）")
    ap.add_argument("--no_hh", action="store_true",
                    help="消融：去掉 H-H 中药-中药视图，只保留 S-S 症状视图")
    ap.add_argument("--no_sana", action="store_true", help="MV-GCN 消融：关掉逐节点 SANA，改用全局层组合")
    ap.add_argument("--no_views", action="store_true", help="MV-GCN 消融：关掉 S-S / H-H 视图残差")
    ap.add_argument("--sana_calib", default="layer", choices=["none", "layer", "node"],
                    help="SANA 尺度校准：layer=传播各层对齐 E0 全局范数；"
                         "node=融合后对齐均匀层组合的逐节点范数；none=旧协议")
    ap.add_argument("--sana_temp", type=float, default=1.0,
                    help="SANA softmax 温度，>1 使注意力更平（退向 Lite）")
    ap.add_argument("--mlp_head", action="store_true",
                    help="在图编码之上加轻量 MLP 评分头（FordNet 式解码器）")
    # ---- SMGCN_MVANA 忠实移植版专有超参（默认对齐 TF 重写版 run.sh）----
    ap.add_argument("--sz_c", type=int, default=2,
                    help="SMGCN_MVANA: 多通道 GCN 通道数")
    ap.add_argument("--d_kv", type=int, default=64,
                    help="SMGCN_MVANA: VLR KV 维度（输出 = sz_c*d_kv）")
    ap.add_argument("--mask_rate", type=float, default=0.2,
                    help="SMGCN_MVANA: DropNode 掩码率（MFM 下按通道线性铺开）")
    ap.add_argument("--sana_alpha", type=float, default=0.2,
                    help="SMGCN_MVANA: 门控 SANA 末两层融合权重")
    ap.add_argument("--mfm", default="MFM", choices=["MFM", "SFM", "TFM"],
                    help="SMGCN_MVANA: 多通道掩码策略")
    ap.add_argument("--no_vlr", action="store_true",
                    help="SMGCN_MVANA 消融：用多通道均值池化替代 VLR 注意力聚合")
    ap.add_argument("--direct_residual", action="store_true",
                    help="SMGCN_MVANA: 加入由训练集 P(药物|症状) 初始化的症状直连残差")
    ap.add_argument("--freeze_direct", action="store_true",
                    help="SMGCN_MVANA: 冻结症状—药物先验矩阵，只学 gate/scale 与图分支，"
                         "保证模型表现不低于该可解释先验")
    ap.add_argument("--direct_shrink", type=float, default=1.0,
                    help="症状直连先验向全局药物频率收缩的伪样本强度")
    ap.add_argument("--direct_gate_init", type=float, default=0.01,
                    help="症状直连残差中图分支的初始门控比例")
    ap.add_argument("--l2_target", default="e0", choices=["e0", "e"],
                    help="L2 加在第 0 层嵌入(e0，各模型强度可比) 还是传播后嵌入(e，图模型实际只有 8.5%%)")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--reg", type=float, default=7e-3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--split_mode", default="record", choices=["record", "combo"],
                    help="record=沿用数据集自带切分(随机, 存在症状组合泄漏); "
                         "combo=合并 train/test 后按症状组合分组再切, 保证 val/test 组合未见过")
    ap.add_argument("--view_src", default="cooccur", choices=["cooccur", "knowledge", "both"],
                    help="S-S/H-H 视图边来源: cooccur=训练集共现(与监督同源); "
                         "knowledge=SymMap 外部知识(性味归经/功效分类/病位病性); both=并集")
    ap.add_argument("--test_ratio", type=float, default=0.15,
                    help="仅 split_mode=combo 生效: 按组合数划给测试集的比例")
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--device", default="auto",
                    help="auto / cpu / cuda / cuda:0")
    ap.add_argument("--split_seed", type=int, default=2021, help="验证集划分的固定种子 (所有模型共享)")
    ap.add_argument("--tag", default="", help="结果文件后缀, 用于区分不同配置")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--self_loop", type=int, default=1,
                    help="二部图是否加自环 (1=加, 0=不加; LightGCN 原论文不加)")
    ap.add_argument("--interaction_weight", default="binary",
                    choices=["binary", "count", "ppmi"],
                    help="症状-药物边权：binary=旧版是否共现；count=共现次数；"
                         "ppmi=去边际流行度的正 PMI")
    ap.add_argument("--core_exclude", type=int, default=0,
                    help="评估时屏蔽训练集中频率最高的 N 味「核心药」(0=不屏蔽)。"
                         "核心药几乎每张方都开, 会稀释 P@K/R@K, 屏蔽后只评估差异化用药")
    ap.add_argument("--tail_exclude", type=int, default=20,
                    help="额外报告去掉训练频率最高 N 味药后的指标，但不参与选模；0=关闭")
    ap.add_argument("--mediator", default="none", choices=["none", "syn", "trt", "both"],
                    help="证型/治法中介视图: 把「辨证-立法-选药」编码成 S->M->H 两跳路径, "
                         "与 S-S/H-H 视图同样以残差注入 (仅 TCM-Lung 有标签)")
    ap.add_argument("--med_mu", type=float, default=1.0,
                    help="中介边权缩放系数")
    ap.add_argument("--med_score", default="lift", choices=["lift", "cond"],
                    help="中介边权: lift=提升度(减掉边际, 默认) / cond=条件概率")
    # ---- MVGCNFlow: 辨证论治四部图分支 ----
    ap.add_argument("--lambda_syn", type=float, default=0.2,
                    help="MVGCNFlow: 证候辅助损失权重")
    ap.add_argument("--lambda_trt", type=float, default=0.2,
                    help="MVGCNFlow: 治法辅助损失权重")
    ap.add_argument("--lambda_con", type=float, default=0.05,
                    help="MVGCNFlow: 治法-药物 InfoNCE 权重; 0=关闭")
    ap.add_argument("--contrast_temp", type=float, default=0.2,
                    help="MVGCNFlow: InfoNCE 温度")
    ap.add_argument("--flow_teacher", type=float, default=0.5,
                    help="MVGCNFlow: 首轮 teacher-forcing 比例，随 epoch 线性衰减到0")
    ap.add_argument("--flow_gamma_init", type=float, default=0.1,
                    help="MVGCNFlow: S-Z/Z-T/T-H 关系残差初值")
    ap.add_argument("--weighting", default="invfreq_pos",
                    choices=["invfreq_pos", "invfreq_all", "invfreq", "none"],
                    help="invfreq_pos=仅正样本逆频率加权（推荐）；"
                         "invfreq_all/invfreq=正负样本都加权（旧协议）；none=不加权")
    ap.add_argument("--loss", default="bce", choices=["bce", "focal", "hard_bce"],
                    help="损失函数: bce=频率加权多标签BCE(原); focal=focal BCE(聚焦难样本); "
                         "hard_bce=top-K难负样本BCE(直接优化P@K排序)")
    ap.add_argument("--topk", type=int, default=50,
                    help="hard_bce 的难负样本数 K (关注 top-K 负样本)")
    ap.add_argument("--focal_gamma", type=float, default=2.0,
                    help="focal loss 的 gamma")
    ap.add_argument("--patience", type=int, default=0,
                    help="早停耐心(以评估次数为单位): val_R20 连续 N 次评估无提升即停; 0=不早停")
    args = ap.parse_args()
    DEVICE = resolve_device(args.device)
    print(f"=== device: {DEVICE} "
          f"({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'}) ===",
          flush=True)

    ds_name = {"smgcn": "Set2Set", "TCM_PD": "TCM_PD", "lung": "TCM_Lung",
               "sdpr": "SDPR_Herb"}[args.dataset]
    print(f"=== dataset: {ds_name} (unified protocol) ===", flush=True)
    train_all, test_orig, n_users, n_items = load_dataset(ds_name)
    requested_models = resolve_models(args.model)
    needs_flow = "MVGCNFlow" in requested_models
    if needs_flow and ds_name != "TCM_Lung":
        raise SystemExit("[MVGCNFlow] 当前只支持具有逐处方证候/治法标签的 TCM_Lung")

    # 证候/治法标签必须与处方记录使用同一组索引切分。
    need_syn = needs_flow or args.mediator in ("syn", "both")
    need_trt = needs_flow or args.mediator in ("trt", "both")
    train_syn_all = load_label_lists(ds_name, "train_syn.txt") if need_syn else None
    test_syn_all = load_label_lists(ds_name, "test_syn.txt") if need_syn else None
    train_trt_all = load_label_lists(ds_name, "train_trt.txt") if need_trt else None
    test_trt_all = load_label_lists(ds_name, "test_trt.txt") if need_trt else None
    if need_syn and (train_syn_all is None or test_syn_all is None):
        raise SystemExit(f"[标签] 缺少 {ds_name}/train_syn.txt 或 test_syn.txt")
    if need_trt and (train_trt_all is None or test_trt_all is None):
        raise SystemExit(f"[标签] 缺少 {ds_name}/train_trt.txt 或 test_trt.txt")
    if train_syn_all is not None:
        train_syn_all, test_syn_all, syn_mapping, n_syn = remap_label_space(
            train_syn_all, test_syn_all)
    else:
        syn_mapping, n_syn = {}, 0
    if train_trt_all is not None:
        train_trt_all, test_trt_all, trt_mapping, n_trt = remap_label_space(
            train_trt_all, test_trt_all)
    else:
        trt_mapping, n_trt = {}, 0
    for records, labels, name in (
            (train_all, train_syn_all, "train_syn"), (test_orig, test_syn_all, "test_syn"),
            (train_all, train_trt_all, "train_trt"), (test_orig, test_trt_all, "test_trt")):
        if labels is not None and len(records) != len(labels):
            raise SystemExit(f"[标签错位] {name}: records={len(records)} labels={len(labels)}")

    # 固定验证集划分 (所有模型、所有 seed 共享同一划分, 保证可比)
    rng = np.random.RandomState(args.split_seed)
    if args.split_mode == "combo":
        # ---- 组合级切分 ----
        # 数据集按「记录」随机切分时, 82%~85% 的记录是重复症状组合, 测试集里 87% 的
        # 症状组合已在训练集出现过 -> 任务退化成查表。这里合并 train/test 后按症状
        # 组合分组再切分, 强制 val/test 的症状组合在训练集中从未出现。
        pool = list(train_all) + list(test_orig)
        pool_syn = ((list(train_syn_all) + list(test_syn_all))
                    if train_syn_all is not None else None)
        pool_trt = ((list(train_trt_all) + list(test_trt_all))
                    if train_trt_all is not None else None)
        groups = {}
        for i, (s, _h) in enumerate(pool):
            groups.setdefault(tuple(sorted(s)), []).append(i)
        keys = sorted(groups.keys())
        gperm = [keys[i] for i in rng.permutation(len(keys))]
        n_val_g = int(len(gperm) * args.val_ratio)
        n_test_g = int(len(gperm) * args.test_ratio)
        def _collect_indices(gs):
            out = []
            for g in gs:
                out.extend(groups[g])
            return out
        val_idx = _collect_indices(gperm[:n_val_g])
        test_idx = _collect_indices(gperm[n_val_g:n_val_g + n_test_g])
        tr_idx = _collect_indices(gperm[n_val_g + n_test_g:])
        tr, val, test = ([pool[i] for i in ids]
                         for ids in (tr_idx, val_idx, test_idx))
        tr_syn = [pool_syn[i] for i in tr_idx] if pool_syn is not None else None
        val_syn = [pool_syn[i] for i in val_idx] if pool_syn is not None else None
        test_syn = [pool_syn[i] for i in test_idx] if pool_syn is not None else None
        tr_trt = [pool_trt[i] for i in tr_idx] if pool_trt is not None else None
        val_trt = [pool_trt[i] for i in val_idx] if pool_trt is not None else None
        test_trt = [pool_trt[i] for i in test_idx] if pool_trt is not None else None
        print(f"[split=combo] unique combos={len(keys)} -> "
              f"#train={len(tr)} #val={len(val)} #test={len(test)} "
              f"(test 组合 100% 未见于 train)", flush=True)
    else:
        perm = rng.permutation(len(train_all))
        n_val = int(len(train_all) * args.val_ratio)
        val = [train_all[i] for i in perm[:n_val]]
        tr = [train_all[i] for i in perm[n_val:]]
        test = list(test_orig)
        tr_syn = [train_syn_all[i] for i in perm[n_val:]] if train_syn_all is not None else None
        val_syn = [train_syn_all[i] for i in perm[:n_val]] if train_syn_all is not None else None
        test_syn = list(test_syn_all) if test_syn_all is not None else None
        tr_trt = [train_trt_all[i] for i in perm[n_val:]] if train_trt_all is not None else None
        val_trt = [train_trt_all[i] for i in perm[:n_val]] if train_trt_all is not None else None
        test_trt = list(test_trt_all) if test_trt_all is not None else None
    print(f"n_users={n_users} n_items={n_items} #train={len(tr)} #val={len(val)} #test={len(test)}",
          flush=True)

    R = build_R(tr, n_users, n_items, weighting=args.interaction_weight)
    w, freq = item_weights(tr, n_items)

    # 「核心药」屏蔽掩码：少数高频药几乎每张方都出现，屏蔽后只评估差异化用药
    core = None
    if args.core_exclude > 0:
        top = np.argsort(-freq)[:args.core_exclude]
        core = np.zeros(n_items, dtype=bool)
        core[top] = True
        print(f"core_exclude={args.core_exclude}: 屏蔽草药覆盖 "
              f"{100 * freq[top].sum() / freq.sum():.1f}% 的开方量", flush=True)

    # 单侧视图图（S-S / H-H）。共现视图必须在 train/val 切分后，
    # 只用实际训练子集 tr 重建。旧实现在阈值为 0 时直接读取
    # 由完整 train.txt 预生成的 pair 文件，会把后来划入验证集的
    # 共现关系（包括草药标签关系）泄漏给模型。
    views = None
    if args.views.lower() != "none":
        # 0 表示使用项目默认阈值，不再表示加载切分前的边文件。
        co_sym, co_herb = [], []
        if args.view_src in ("cooccur", "both"):
            co_sym, co_herb = build_pairs_from_data(
                tr, n_users, n_items,
                args.sym_thresh if args.sym_thresh > 0 else 5,
                args.herb_thresh if args.herb_thresh > 0 else 40)

        if args.view_src == "cooccur":
            sym_pairs, herb_pairs = co_sym, co_herb
        elif args.view_src == "knowledge":
            sym_pairs = load_pairs(ds_name, "symPair-know.txt")
            herb_pairs = load_pairs(ds_name, "herbPair-know.txt")
        else:  # both: 训练子集共现边 ∪ 外部知识边
            sym_pairs = sorted(set(co_sym) |
                               set(load_pairs(ds_name, "symPair-know.txt")))
            herb_pairs = sorted(set(co_herb) |
                                set(load_pairs(ds_name, "herbPair-know.txt")))
        A_ss = build_view_adj(sym_pairs, n_users) if "ss" in args.views else None
        A_hh = build_view_adj(herb_pairs, n_items) if "hh" in args.views else None
        views = (A_ss, A_hh)
        print(f"views: A_ss={'yes' if A_ss is not None else 'no'} "
              f"({len(sym_pairs)} pairs) | A_hh={'yes' if A_hh is not None else 'no'} "
              f"({len(herb_pairs)} pairs)", flush=True)
    print(f"interaction graph={args.interaction_weight} nonzero={int((R > 0).sum())} "
          f"weight_sum={R.sum():.1f} | item weight range=[{w.min():.3f}, {w.max():.3f}]",
          flush=True)

    # 证型 / 治法中介视图（级联式辨证论治先验），仅用训练部分统计
    med = None
    if args.mediator != "none":
        A_med, n_med = build_mediator_adj(tr, tr_syn, tr_trt, n_users, n_items,
                                          mu=args.med_mu, score=args.med_score)
        if A_med is not None:
            med = (A_med, n_med)
            print(f"mediator: {args.mediator} | n_med={n_med} mu={args.med_mu} "
                  f"score={args.med_score} | edges={int((A_med > 0).sum())}", flush=True)

    flow = None
    if needs_flow:
        A_sz, A_zt, A_th = build_four_partite_relations(
            tr, tr_syn, tr_trt, n_users, n_items, n_syn, n_trt)
        flow = {"n_syn": n_syn, "n_trt": n_trt,
                "A_sz": A_sz, "A_zt": A_zt, "A_th": A_th,
                "tr_syn": tr_syn, "val_syn": val_syn, "test_syn": test_syn,
                "tr_trt": tr_trt, "val_trt": val_trt, "test_trt": test_trt,
                "syn_mapping": syn_mapping, "trt_mapping": trt_mapping}
        print(f"four-partite: S-Z={int((A_sz > 0).sum())} "
              f"Z-T={int((A_zt > 0).sum())} T-H={int((A_th > 0).sum())} "
              f"| n_syn={n_syn} n_trt={n_trt}", flush=True)

    models = requested_models
    os.makedirs(RESULT_DIR, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    for model_name in models:
        # 增量落盘：每个 seed 结束即写入 jsonl，任务被中断也不丢已完成的 seed
        jsonl = os.path.join(RESULT_DIR, f"v2_{ds_name}_{model_name}{suffix}_runs.jsonl")
        config_hash, run_config = experiment_fingerprint(args, model_name)
        done = {}
        if os.path.exists(jsonl):
            with open(jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        old_hash = r.get("_config_hash")
                        if old_hash is None:
                            raise SystemExit(
                                f"[resume blocked] {jsonl} 是旧版结果，没有配置指纹。"
                                "请使用新 --tag 重跑，不能将其与新实验混合。")
                        if old_hash != config_hash:
                            raise SystemExit(
                                f"[resume blocked] {jsonl} 配置不一致: "
                                f"existing={old_hash}, current={config_hash}。"
                                "请更换 --tag。")
                        done[r["_seed"]] = r
            if done:
                print(f"[resume] {jsonl} 已有 {len(done)} 个 seed: {sorted(done)}", flush=True)

        for seed in range(args.seed_start, args.seed_start + args.seeds):
            if seed in done:
                print(f"[{model_name} seed={seed}] 跳过 (已完成)", flush=True)
                continue
            res = train_one_run(model_name, tr, val, test, n_users, n_items, R, w, seed, args,
                                views=views, med=med, core=core, flow=flow)
            res["_seed"] = seed
            res["_config_hash"] = config_hash
            with open(jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            done[seed] = res
            print(f"[{model_name} seed={seed}] best_ep={res['_best_epoch']:3d} "
                  f"P@5={res['P@5']:.4f} R@20={res['R@20']:.4f} F1@20={res['F1@20']:.4f} "
                  f"({res['_train_time_s']}s)", flush=True)

        runs = [done[s] for s in sorted(done)]
        agg = {}
        for k in list(runs[0].keys()):
            if k.startswith("_"):
                continue
            vals = [r[k] for r in runs]
            # ddof=1 stdev is undefined for a single seed; report 0 rather than nan.
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            agg[k] = {"mean": float(np.mean(vals)), "std": sd,
                      "max": float(np.max(vals)), "min": float(np.min(vals))}
        out = {"model": model_name, "dataset": ds_name, "n_seeds": len(runs),
               "config_hash": config_hash, "run_config": run_config,
               "protocol": "unified_v2 (prescription-level weighted BCE, val-based selection)",
               "args": vars(args), "aggregate": agg, "runs": runs}
        fn = os.path.join(RESULT_DIR, f"v2_{ds_name}_{model_name}{suffix}_summary.json")
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n--- {model_name} on {ds_name} (mean±std over {len(runs)} seeds) ---")
        for k in ["P@5", "P@10", "P@20", "R@5", "R@10", "R@20", "F1@20"]:
            print(f"  {k:6s} {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}")
        print(f"  saved -> {fn}\n", flush=True)


if __name__ == "__main__":
    main()
