#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MLP / NGCF / SMGCN 的统一协议轻量回归测试。

该测试只验证三种基础方法在相同数据、损失、切分和评估代码下
可以完成训练，且所有输出指标有限。合成数据的指标不用于论文结论。

真实 Set2Set 比较请运行：
  python run_baselines_v2.py --dataset smgcn --model basic \
      --seeds 3 --epochs 200 --tag basic_p1
"""
from types import SimpleNamespace

import numpy as np
import torch

import run_baselines_v2 as rb


MODELS = ("MLP", "NGCF", "SMGCN")


def _records(n, offset=0):
    """生成可重现的小型处方集，草药数保持 >= 20 以覆盖 P/R@20。"""
    out = []
    for i in range(n):
        j = i + offset
        syms = sorted({j % 6, (j * 3 + 1) % 6})
        herbs = sorted({j % 25, (j + 3) % 25, (j * 2 + 7) % 25})
        out.append((syms, herbs))
    return out


def _args():
    return SimpleNamespace(
        self_loop=1, dim=8, layers=2, node_drop=0.0,
        gamma_init=0.0, no_hh=False, mlp_head=False,
        no_views=False, no_sana=False, sana_calib="none", sana_temp=1.0,
        lr=1e-2, weighting="invfreq", loss="bce", focal_gamma=2.0,
        topk=5, l2_target="e0", reg=1e-4, batch=12, epochs=3,
        eval_every=1, trace=False, patience=0,
    )


def _run_basic_models():
    torch.set_num_threads(1)
    rb.DEVICE = torch.device("cpu")
    n_users, n_items = 6, 25
    train = _records(36)
    val = _records(6, 100)
    test = _records(6, 200)
    R = rb.build_R(train, n_users, n_items)
    weights, _ = rb.item_weights(train, n_items)
    A_ss = rb.build_view_adj([(0, 1), (1, 2), (2, 3), (4, 5)], n_users)
    views = (A_ss, None)
    args = _args()

    results = {}
    for model_name in MODELS:
        result = rb.train_one_run(
            model_name, train, val, test, n_users, n_items,
            R, weights, seed=7, args=args, views=views)
        for metric in ("P@5", "P@10", "P@20", "R@5", "R@10", "R@20", "F1@20"):
            value = result[metric]
            assert np.isfinite(value), "%s %s is not finite" % (model_name, metric)
            assert 0.0 <= value <= 1.0, "%s %s out of range: %r" % (
                model_name, metric, value)
        assert 1 <= result["_best_epoch"] <= args.epochs
        results[model_name] = result
    return results


def test_basic_models_smoke():
    _run_basic_models()


def main():
    results = _run_basic_models()
    print("model       best_ep      P@5     R@20     F1@20")
    print("------------------------------------------------")
    for model_name in MODELS:
        r = results[model_name]
        print("%-10s  %7d  %8.4f  %8.4f  %8.4f" % (
            model_name, r["_best_epoch"], r["P@5"], r["R@20"], r["F1@20"]))
    print("BASIC MODEL SMOKE: PASS")


if __name__ == "__main__":
    main()
