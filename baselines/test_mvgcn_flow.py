#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MVGCNFlow 的最小可重复测试：关系构图、前后向和短训练。"""
from types import SimpleNamespace

import numpy as np
import torch

import run_baselines_v2 as runner


def _records(n_items=24):
    train = [
        ([0, 1], [0, 1, 2]), ([1, 2], [2, 3, 4]),
        ([2, 3], [4, 5, 6]), ([3, 4], [6, 7, 8]),
        ([0, 4], [1, 8, 9]), ([1, 3], [3, 7, 10]),
        ([0, 2, 4], [0, 5, 11]), ([1, 4], [2, 9, 12]),
    ]
    val = [([0, 3], [0, 7, 13]), ([2, 4], [5, 8, 14])]
    test = [([0, 1, 4], [1, 9, 15]), ([1, 2, 3], [3, 6, 16])]
    assert max(h for rows in train + val + test for h in rows[1]) < n_items
    return train, val, test


def _args():
    return SimpleNamespace(
        self_loop=1, dim=8, layers=2, node_drop=0.0, gamma_init=0.0,
        no_hh=False, mlp_head=False, no_views=True, no_sana=False,
        sana_calib="none", sana_temp=1.0, flow_gamma_init=0.1,
        lr=1e-3, weighting="none", loss="bce", focal_gamma=2.0,
        topk=10, l2_target="e0", reg=1e-4, batch=4, epochs=2,
        eval_every=1, trace=False, patience=0, lambda_syn=0.2,
        lambda_trt=0.2, lambda_con=0.05, contrast_temp=0.2,
        flow_teacher=0.5,
    )


def test_flow_smoke():
    runner.DEVICE = torch.device("cpu")
    n_users, n_items, n_syn, n_trt = 5, 24, 3, 4
    train, val, test = _records(n_items)
    tr_syn = [[0], [0, 1], [1], [2], [0, 2], [1, 2], [0, 1], [2]]
    tr_trt = [[0], [0, 1], [1], [2], [0, 2], [2, 3], [1, 3], [3]]
    val_syn, test_syn = [[0], [1, 2]], [[0, 2], [1]]
    val_trt, test_trt = [[0], [2]], [[0, 3], [1, 2]]

    A_sz, A_zt, A_th = runner.build_four_partite_relations(
        train, tr_syn, tr_trt, n_users, n_items, n_syn, n_trt)
    assert A_sz.shape == (n_users, n_syn)
    assert A_zt.shape == (n_syn, n_trt)
    assert A_th.shape == (n_trt, n_items)
    assert all(torch.isfinite(a).all() for a in (A_sz, A_zt, A_th))

    R = runner.build_R(train, n_users, n_items)
    weights, _ = runner.item_weights(train, n_items)
    flow = dict(n_syn=n_syn, n_trt=n_trt, A_sz=A_sz, A_zt=A_zt, A_th=A_th,
                tr_syn=tr_syn, val_syn=val_syn, test_syn=test_syn,
                tr_trt=tr_trt, val_trt=val_trt, test_trt=test_trt)
    result = runner.train_one_run(
        "MVGCNFlow", train, val, test, n_users, n_items, R, weights,
        seed=7, args=_args(), views=(None, None), flow=flow)
    for key in ("P@5", "R@20", "F1@20"):
        assert np.isfinite(result[key]), (key, result[key])
    for key in ("_gamma_sz", "_gamma_zt", "_gamma_th"):
        assert np.isfinite(result[key]), (key, result[key])
    aux = result["_flow_aux"]
    assert 0.0 <= aux["syn_micro_f1"] <= 1.0
    assert 0.0 <= aux["trt_micro_f1"] <= 1.0
    assert 0.0 <= aux["gate_mean"] <= 1.0


if __name__ == "__main__":
    test_flow_smoke()
    print("MVGCNFlow smoke test: PASS")
