"""
幾何共用函式（只需要 numpy）：正規化、關鍵點

從 constraint_dataset.py、eval_large.py 移出來，讓「使用模型」不必載入 SketchGraphs 套件
（constraint_dataset.py 一開頭就要 import sketchgraphs，只有整理訓練資料時才需要）。
"""
import numpy as np

import constraint_config as C

_EPS = 1e-8


def normalize(kinds, geom):
    """平移到外框中心、縮放到外框最長邊 = 2（和原本 utils.BoundingBoxShiftScale 相同的規則）。
    回傳 None 代表整張圖退化（大小為 0）。"""
    xs, ys = [], []
    for k, g in zip(kinds, geom):
        if k in (C.PRIM_POINT, C.PRIM_LINE, C.PRIM_ARC):
            xs += [g[0], g[2]]; ys += [g[1], g[3]]
        if k in (C.PRIM_CIRCLE, C.PRIM_ARC):
            xs += [g[4] - g[6], g[4] + g[6]]; ys += [g[5] - g[6], g[5] + g[6]]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    size = max(w, h)
    if not np.isfinite(size) or size < _EPS:
        return None
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    out = geom.copy()
    for i, k in enumerate(kinds):
        if k != C.PRIM_CIRCLE:                       # 圓沒有起終點，維持 0
            out[i, [0, 2]] -= cx; out[i, [1, 3]] -= cy
        out[i, 4] -= cx; out[i, 5] -= cy
    out[:, 0:6] *= 2.0 / size
    out[:, 6] *= 2.0 / size
    return out


def keypoints(kinds, geom):
    """每個圖元取 5 個代表點 (…,5,2)，用來算「離選定那一對有多近」；空格填無限大"""
    g = geom.astype(np.float64)
    s, e, c, r = g[..., 0:2], g[..., 2:4], g[..., 4:6], g[..., 6:7]
    pts = np.stack([s, e, c, (s + e) / 2, c], axis=-2)                           # 點/線/弧的預設
    circ = np.stack([c, c + r * [1, 0], c - r * [1, 0], c + r * [0, 1], c - r * [0, 1]], axis=-2)
    pts = np.where((kinds == C.PRIM_CIRCLE)[..., None, None], circ, pts)
    pts = np.where((kinds < 0)[..., None, None], np.inf, pts)
    return pts


def normalize_batch(kinds, geom):
    """和 constraint_dataset._normalize 相同的規則（外框中心移到原點、最長邊縮放成 2），一次處理一整批"""
    g = geom.astype(np.float64).copy()
    k = kinds[..., None]
    se = (k == C.PRIM_POINT) | (k == C.PRIM_LINE) | (k == C.PRIM_ARC)
    cr = (k == C.PRIM_CIRCLE) | (k == C.PRIM_ARC)
    big = np.inf
    xs_min = np.minimum(np.where(se, np.minimum(g[..., 0:1], g[..., 2:3]), big),
                        np.where(cr, g[..., 4:5] - g[..., 6:7], big)).min(axis=(1, 2))
    xs_max = np.maximum(np.where(se, np.maximum(g[..., 0:1], g[..., 2:3]), -big),
                        np.where(cr, g[..., 4:5] + g[..., 6:7], -big)).max(axis=(1, 2))
    ys_min = np.minimum(np.where(se, np.minimum(g[..., 1:2], g[..., 3:4]), big),
                        np.where(cr, g[..., 5:6] - g[..., 6:7], big)).min(axis=(1, 2))
    ys_max = np.maximum(np.where(se, np.maximum(g[..., 1:2], g[..., 3:4]), -big),
                        np.where(cr, g[..., 5:6] + g[..., 6:7], -big)).max(axis=(1, 2))
    size = np.maximum(xs_max - xs_min, ys_max - ys_min)
    size = np.where(np.isfinite(size) & (size > 1e-8), size, 2.0)
    cx, cy = (xs_max + xs_min) / 2, (ys_max + ys_min) / 2
    cx = np.where(np.isfinite(cx), cx, 0)[:, None]
    cy = np.where(np.isfinite(cy), cy, 0)[:, None]
    notc = (kinds != C.PRIM_CIRCLE)
    g[..., 0] -= cx * notc; g[..., 2] -= cx * notc
    g[..., 1] -= cy * notc; g[..., 3] -= cy * notc
    g[..., 4] -= cx; g[..., 5] -= cy
    g[..., 0:7] *= (2.0 / size)[:, None, None]
    g[kinds < 0] = 0
    return g.astype(np.float32)
