"""
階段 2-1：從階段 1 的資料建立訓練用的「圖元對標籤」

做的事：
  1. 每個圖元對，把它所有的約束整理成一個 128 格的「選項清單」（哪些選項是正確答案）
  2. 推導約束（傳遞性）：例如 A=B、A=C 就推出 B=C，另外記錄、可以開關
  3. 合法選項表：每種圖元組合（例如 線-圓）在訓練資料中實際出現過的選項
  4. 幾何驗證：檢查推導出來的約束在圖上是否真的成立（例如推導出的重合，兩點距離應為 0）

用法：
  python constraint_labels.py                 # 處理全部（約數分鐘）
  python constraint_labels.py --limit 50000   # 只處理前 5 萬張（測試用）
輸出：data/constraint/labels.npz、labels_stats.json
"""
import os
import json
import time
import argparse
import collections
import multiprocessing as mp

import numpy as np

import constraint_config as C

P_, L_, CI_, AR_ = C.PRIM_POINT, C.PRIM_LINE, C.PRIM_CIRCLE, C.PRIM_ARC
W_, S_, E_, CE_ = C.SUB_WHOLE, C.SUB_START, C.SUB_END, C.SUB_CENTER
CURVES = (L_, CI_, AR_)

RAW_COINCIDENT, RAW_HORIZONTAL, RAW_VERTICAL, RAW_PARALLEL = 0, 4, 6, 5
RAW_PERPENDICULAR, RAW_MIDPOINT, RAW_EQUAL = 9, 10, 11


def pointlike(kind, sub):
    """這個位置是不是一個「點」（點本身、線的端點、圓心、弧的端點與圓心）"""
    if kind == P_:
        return sub == W_
    if kind == L_:
        return sub in (S_, E_)
    if kind == CI_:
        return sub == CE_
    if kind == AR_:
        return sub in (S_, E_, CE_)
    return False


# 向量化版本：PL[類型, 位置] = 是否為點
PL = np.zeros((4, 4), dtype=bool)
for _k in range(4):
    for _s in range(4):
        PL[_k, _s] = pointlike(_k, _s)


class UF:
    """並查集（分組用）"""
    def __init__(self):
        self.p = {}

    def find(self, x):
        p = self.p
        p.setdefault(x, x)
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb

    def groups(self):
        g = collections.defaultdict(list)
        for x in list(self.p):
            g[self.find(x)].append(x)
        return g


# =====================================================================
# 一張草圖的推導
# =====================================================================
def derive_sketch(kinds, pairs, singles):
    """kinds: (P,) 圖元類型；pairs: (n,5) [a,b,原始類型,A位置,B位置]；singles: (m,3) [a,原始類型,位置]
    回傳推導出的約束 set{(a, b, 類型序號, A位置, B位置)}（a<b，不含原本就有的）"""
    coin, ygrp, xgrp = UF(), UF(), UF()
    par, eq_line, eq_curve = UF(), UF(), UF()
    on_curve, mids, perp = [], [], []
    direct = set()

    for a, b, t, sa, sb in pairs:
        a, b, t, sa, sb = int(a), int(b), int(t), int(sa), int(sb)
        if t in C.RAW_TO_TIDX:
            direct.add((a, b, C.RAW_TO_TIDX[t], sa, sb))
        ka, kb = int(kinds[a]), int(kinds[b])
        pa, pb = pointlike(ka, sa), pointlike(kb, sb)
        if t == RAW_COINCIDENT:
            if pa and pb:
                coin.union((a, sa), (b, sb))
            elif pa and sb == W_ and kb in CURVES:
                on_curve.append(((a, sa), b))
            elif pb and sa == W_ and ka in CURVES:
                on_curve.append(((b, sb), a))
        elif t == RAW_HORIZONTAL and pa and pb:
            ygrp.union((a, sa), (b, sb))
        elif t == RAW_VERTICAL and pa and pb:
            xgrp.union((a, sa), (b, sb))
        elif t == RAW_PARALLEL and ka == L_ and kb == L_ and sa == W_ and sb == W_:
            par.union(a, b)
        elif t == RAW_PERPENDICULAR and ka == L_ and kb == L_ and sa == W_ and sb == W_:
            perp.append((a, b))
        elif t == RAW_EQUAL and sa == W_ and sb == W_:
            if ka == L_ and kb == L_:
                eq_line.union(a, b)
            elif ka in (CI_, AR_) and kb in (CI_, AR_):
                eq_curve.union(a, b)
        elif t == RAW_MIDPOINT:
            if pa and kb == L_ and sb == W_:
                mids.append(((a, sa), b))
            elif pb and ka == L_ and sa == W_:
                mids.append(((b, sb), a))

    # 單一圖元約束：水平線 / 垂直線（含「同一條線的起點與終點水平」這種寫法）
    hlines, vlines = [], []
    for a, t, s in singles:
        a, t, s = int(a), int(t), int(s)
        if kinds[a] != L_ or s not in (W_, S_ * 4 + E_, E_ * 4 + S_):
            continue
        if t == RAW_HORIZONTAL:
            hlines.append(a)
        elif t == RAW_VERTICAL:
            vlines.append(a)
    for a in hlines:
        ygrp.union((a, S_), (a, E_))
        par.union(a, hlines[0])
    for a in vlines:
        xgrp.union((a, S_), (a, E_))
        par.union(a, vlines[0])
    if hlines and vlines:
        perp.append((hlines[0], vlines[0]))

    # 重合的點，x 和 y 也一定相同
    coin_groups = coin.groups()
    for members in coin_groups.values():
        for m in members[1:]:
            ygrp.union(members[0], m)
            xgrp.union(members[0], m)

    out = set()

    def add(a, sa, b, sb, t):
        if a == b:
            return
        if a > b:
            a, sa, b, sb = b, sb, a, sa
        key = (a, b, t, sa, sb)
        if key not in direct:
            out.add(key)

    def coin_members(kp):
        return coin_groups.get(coin.find(kp), [kp]) if kp in coin.p else [kp]

    # 1. 點對點重合
    for members in coin_groups.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                (a, sa), (b, sb) = members[i], members[j]
                add(a, sa, b, sb, C.T_COINCIDENT)
    # 2. 點在曲線上：與該點重合的其他點也在曲線上
    for kp, c in on_curve:
        for (a, sa) in coin_members(kp):
            add(a, sa, c, W_, C.T_COINCIDENT)
    # 3. 中點：與該點重合的其他點也是中點
    for kp, ln in mids:
        for (a, sa) in coin_members(kp):
            add(a, sa, ln, W_, C.T_MIDPOINT)
    # 4. 水平 / 垂直（點對點），已重合的不重複加
    for grp, t in ((ygrp, C.T_HORIZONTAL), (xgrp, C.T_VERTICAL)):
        for members in grp.groups().values():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    p, q = members[i], members[j]
                    if p in coin.p and q in coin.p and coin.find(p) == coin.find(q):
                        continue
                    add(p[0], p[1], q[0], q[1], t)
    # 5. 平行
    par_groups = par.groups()
    for members in par_groups.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                add(members[i], W_, members[j], W_, C.T_PARALLEL)
    # 6. 垂直相交：兩個平行組之間全部垂直
    seen = set()
    for a, b in perp:
        ra, rb = par.find(a), par.find(b)
        if (ra, rb) in seen:
            continue
        seen.add((ra, rb)); seen.add((rb, ra))
        for x in par_groups.get(ra, [a]):
            for y in par_groups.get(rb, [b]):
                add(x, W_, y, W_, C.T_PERPENDICULAR)
    # 7. 等長 / 等徑
    for grp in (eq_line, eq_curve):
        for members in grp.groups().values():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    add(members[i], W_, members[j], W_, C.T_EQUAL)
    return out


def _derive_chunk(job):
    start, kinds, p_off, p_rows, s_off, s_rows = job
    out = []
    for i in range(len(kinds)):
        pr = p_rows[p_off[i]:p_off[i + 1]]
        sr = s_rows[s_off[i]:s_off[i + 1]]
        for a, b, t, sa, sb in derive_sketch(kinds[i], pr, sr):
            out.append((start + i, a, b, t, sa, sb))
    return np.array(out, dtype=np.int32).reshape(-1, 6)


# =====================================================================
# 幾何驗證：算每個約束的「殘差」（成立時應接近 0）
# =====================================================================
def residuals(sk, a, b, t, sa, sb, kinds, geom):
    ga, gb = geom[sk, a].astype(np.float64), geom[sk, b].astype(np.float64)
    ka, kb = kinds[sk, a], kinds[sk, b]

    def kp(g, k, s):
        out = g[:, 0:2].copy()
        out[s == E_] = g[s == E_, 2:4]
        out[s == CE_] = g[s == CE_, 4:6]
        return out

    pa, pb = kp(ga, ka, sa), kp(gb, kb, sb)
    pla, plb = PL[ka, sa], PL[kb, sb]

    def unit(g):
        d = g[:, 2:4] - g[:, 0:2]
        n = np.linalg.norm(d, axis=1, keepdims=True)
        return d / np.maximum(n, 1e-12), n[:, 0]

    ua, la = unit(ga)
    ub, lb = unit(gb)
    res = np.full(len(sk), np.nan)

    def dist_to_carrier(p, g, k):
        u, _ = unit(g)
        d_line = np.abs((p[:, 0] - g[:, 0]) * u[:, 1] - (p[:, 1] - g[:, 1]) * u[:, 0])
        d_circ = np.abs(np.linalg.norm(p - g[:, 4:6], axis=1) - g[:, 6])
        return np.where(k == L_, d_line, d_circ)

    m = (t == C.T_COINCIDENT) & pla & plb
    res[m] = np.linalg.norm(pa[m] - pb[m], axis=1)
    m = (t == C.T_COINCIDENT) & pla & (sb == W_) & np.isin(kb, CURVES)
    res[m] = dist_to_carrier(pa[m], gb[m], kb[m])
    m = (t == C.T_COINCIDENT) & plb & (sa == W_) & np.isin(ka, CURVES)
    res[m] = dist_to_carrier(pb[m], ga[m], ka[m])
    m = (t == C.T_HORIZONTAL) & pla & plb
    res[m] = np.abs(pa[m, 1] - pb[m, 1])
    m = (t == C.T_VERTICAL) & pla & plb
    res[m] = np.abs(pa[m, 0] - pb[m, 0])
    ll = (ka == L_) & (kb == L_)
    m = (t == C.T_PARALLEL) & ll
    res[m] = np.abs(ua[m, 0] * ub[m, 1] - ua[m, 1] * ub[m, 0])
    m = (t == C.T_PERPENDICULAR) & ll
    res[m] = np.abs((ua[m] * ub[m]).sum(1))
    m = (t == C.T_EQUAL) & ll
    res[m] = np.abs(la[m] - lb[m])
    cc = np.isin(ka, (CI_, AR_)) & np.isin(kb, (CI_, AR_))
    m = (t == C.T_EQUAL) & cc
    res[m] = np.abs(ga[m, 6] - gb[m, 6])
    m = (t == C.T_MIDPOINT) & pla & (kb == L_) & (sb == W_)
    res[m] = np.linalg.norm(pa[m] - gb[m, 4:6], axis=1)
    m = (t == C.T_MIDPOINT) & plb & (ka == L_) & (sa == W_)
    res[m] = np.linalg.norm(pb[m] - ga[m, 4:6], axis=1)
    return res


def _residual_summary(r, t):
    out = {}
    for ti in range(C.NUM_TYPES):
        x = r[(t == ti) & np.isfinite(r)]
        if len(x) == 0:
            continue
        out[C.TIDX_NAMES_ZH[ti]] = {
            "可驗證筆數": int(len(x)),
            "中位數": float(np.median(x)),
            "99百分位": float(np.percentile(x, 99)),
            "殘差>0.001的比例": round(float((x > 1e-3).mean()), 5),
        }
    return out


# =====================================================================
# 主流程
# =====================================================================
def _pack(rows_group, opts, n_groups):
    """把 (組別, 選項) 打包成每組 16 bytes 的位元遮罩"""
    mask = np.zeros((n_groups, 16), dtype=np.uint8)
    np.bitwise_or.at(mask, (rows_group, opts // 8), (1 << (opts % 8)).astype(np.uint8))
    return mask


def build(limit=None, workers=None):
    t0 = time.time()
    print("讀取階段 1 資料 ...")
    d = np.load(C.DATA_FILE)
    kinds, geom, split = d["kinds"], d["geom"], d["split"]
    p_off, p_rows = d["pair_offsets"], d["pair_rows"]
    s_off, s_rows = d["single_offsets"], d["single_rows"]
    N = len(kinds) if limit is None else min(limit, len(kinds))
    kinds, geom, split = kinds[:N], geom[:N], split[:N]
    p_rows, s_rows = p_rows[:p_off[N]], s_rows[:s_off[N]]
    p_off, s_off = p_off[:N + 1], s_off[:N + 1]
    print(f"草圖 {N:,} 張，圖元對約束 {len(p_rows):,} 筆")

    # ---- 1. 原始（直接標註）的約束 → 選項 ----
    sk_of_row = np.repeat(np.arange(N, dtype=np.int64), np.diff(p_off))
    raw_t = p_rows[:, 2].astype(np.int64)
    lut = np.full(64, -1, dtype=np.int64)
    for raw, ti in C.RAW_TO_TIDX.items():
        lut[raw] = ti
    tidx = lut[np.clip(raw_t, 0, 63)]
    keep = tidx >= 0
    d_sk = sk_of_row[keep]
    d_a, d_b = p_rows[keep, 0].astype(np.int64), p_rows[keep, 1].astype(np.int64)
    d_sa, d_sb = p_rows[keep, 3].astype(np.int64), p_rows[keep, 4].astype(np.int64)
    d_t = tidx[keep]
    print(f"8 種目標約束：{keep.sum():,} 筆（其餘 {(~keep).sum():,} 筆是尺寸等其他類型，不使用）")

    # ---- 2. 推導約束 ----
    print("推導約束中 ...")
    chunk = 20000
    jobs = []
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        jobs.append((s, kinds[s:e], p_off[s:e + 1] - p_off[s], p_rows[p_off[s]:p_off[e]],
                     s_off[s:e + 1] - s_off[s], s_rows[s_off[s]:s_off[e]]))
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    from tqdm import tqdm
    if workers <= 1 or len(jobs) == 1:
        parts = [_derive_chunk(j) for j in tqdm(jobs, ncols=80)]
    else:
        with mp.Pool(min(workers, len(jobs))) as pool:
            parts = list(tqdm(pool.imap(_derive_chunk, jobs), total=len(jobs), ncols=80))
    der = np.concatenate(parts) if parts else np.zeros((0, 6), np.int32)
    der = der.astype(np.int64)
    print(f"推導出 {len(der):,} 筆新約束（{time.time() - t0:.0f} 秒）")

    # ---- 3. 幾何驗證 ----
    print("幾何驗證中 ...")
    r_direct = residuals(d_sk, d_a, d_b, d_t, d_sa, d_sb, kinds, geom)
    r_der = residuals(der[:, 0], der[:, 1], der[:, 2], der[:, 3], der[:, 4], der[:, 5], kinds, geom) \
        if len(der) else np.zeros(0)

    # ---- 4. 合併成「圖元對」表 ----
    all_sk = np.concatenate([d_sk, der[:, 0]])
    all_a = np.concatenate([d_a, der[:, 1]])
    all_b = np.concatenate([d_b, der[:, 2]])
    all_opt = np.concatenate([d_t * 16 + d_sa * 4 + d_sb, der[:, 3] * 16 + der[:, 4] * 4 + der[:, 5]])
    is_der = np.concatenate([np.zeros(len(d_sk), bool), np.ones(len(der), bool)])
    key = all_sk * 256 + all_a * 16 + all_b
    order = np.argsort(key, kind="stable")
    key, all_opt, is_der = key[order], all_opt[order], is_der[order]
    ukey, grp = np.unique(key, return_inverse=True)
    U = len(ukey)
    direct_mask = _pack(grp[~is_der], all_opt[~is_der], U)
    derived_mask = _pack(grp[is_der], all_opt[is_der], U)
    derived_mask &= ~direct_mask
    pair_sketch = (ukey // 256).astype(np.int32)
    pair_a = ((ukey // 16) % 16).astype(np.int8)
    pair_b = (ukey % 16).astype(np.int8)

    # ---- 5. 合法選項表（只用訓練集、只用直接標註）----
    tr = split[d_sk] == 0
    counts = np.zeros((4, 4, C.NUM_OPTIONS), dtype=np.int64)
    ka, kb = kinds[d_sk, d_a].astype(np.int64), kinds[d_sk, d_b].astype(np.int64)
    opt = d_t * 16 + d_sa * 4 + d_sb
    opt_m = d_t * 16 + d_sb * 4 + d_sa
    np.add.at(counts, (ka[tr], kb[tr], opt[tr]), 1)
    np.add.at(counts, (kb[tr], ka[tr], opt_m[tr]), 1)
    totals = counts.sum(-1, keepdims=True)
    legal = (counts >= C.LEGAL_MIN_COUNT) & (counts >= C.LEGAL_MIN_FRAC * totals)

    # 被合法表排除的標籤有多少
    lab_bits = np.unpackbits(direct_mask, axis=1, bitorder="little").astype(bool)
    pk_a = kinds[pair_sketch, pair_a].astype(np.int64)
    pk_b = kinds[pair_sketch, pair_b].astype(np.int64)
    lg = legal[pk_a, pk_b]
    dropped = int((lab_bits & ~lg).sum())
    total_lab = int(lab_bits.sum())
    pairs_all_dropped = int((((lab_bits & lg).sum(1) == 0) & lab_bits.any(1)).sum())
    der_bits = np.unpackbits(derived_mask, axis=1, bitorder="little").astype(bool)
    der_legal = int((der_bits & lg).sum())
    del lab_bits

    np.savez(C.LABELS_FILE, pair_sketch=pair_sketch, pair_a=pair_a, pair_b=pair_b,
             direct_mask=direct_mask, derived_mask=derived_mask,
             legal=legal, option_counts=counts, n_sketches=np.array(N))

    # ---- 統計 ----
    der_t = der[:, 3]
    n_direct_pairs = int((direct_mask.any(1)).sum())
    n_new_pairs = int(((~direct_mask.any(1)) & derived_mask.any(1)).sum())
    legal_names = {}
    for x in range(4):
        for y in range(x, 4):
            names = [C.option_name_zh(o, x, y) for o in np.argsort(-counts[x, y]) if legal[x, y, o]]
            legal_names[f"{C.PRIM_NAMES_ZH[x]}-{C.PRIM_NAMES_ZH[y]}"] = {"合法選項數": len(names), "選項(依出現次數)": names}
    stats = {
        "處理秒數": round(time.time() - t0, 1),
        "草圖數": int(N),
        "直接標註的約束(8種)": int(len(d_t)),
        "推導出的新約束": int(len(der)),
        "推導約束 各類型": {C.TIDX_NAMES_ZH[i]: int((der_t == i).sum()) for i in range(C.NUM_TYPES)},
        "直接標註 各類型": {C.TIDX_NAMES_ZH[i]: int((d_t == i).sum()) for i in range(C.NUM_TYPES)},
        "有直接標註的圖元對": n_direct_pairs,
        "只靠推導才有約束的新圖元對": n_new_pairs,
        "幾何驗證(直接標註)": _residual_summary(r_direct, d_t),
        "幾何驗證(推導)": _residual_summary(r_der, der_t) if len(der) else {},
        "被合法表排除的直接標籤": [dropped, round(dropped / max(total_lab, 1), 5)],
        "因此完全沒有標籤的圖元對": pairs_all_dropped,
        "落在合法表內的推導標籤": der_legal,
        "合法選項": legal_names,
    }
    with open(C.LABELS_STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print_stats(stats)
    print(f"\n已存到 {C.LABELS_FILE}（{os.path.getsize(C.LABELS_FILE) / 2**20:.1f} MB）、{C.LABELS_STATS_FILE}")
    return stats


def print_stats(s):
    print("\n========== 標籤統計 ==========")
    for k in ["處理秒數", "草圖數", "直接標註的約束(8種)", "推導出的新約束", "有直接標註的圖元對",
              "只靠推導才有約束的新圖元對", "被合法表排除的直接標籤", "因此完全沒有標籤的圖元對", "落在合法表內的推導標籤"]:
        v = s[k]
        print(f"{k}: {v:,}" if isinstance(v, int) else f"{k}: {v}")
    print("各類型：直接標註 / 推導新增")
    for n in C.TIDX_NAMES_ZH:
        print(f"    {n}: {s['直接標註 各類型'][n]:,} / {s['推導約束 各類型'][n]:,}")
    print("幾何驗證（殘差應接近 0；『>0.001 的比例』越小越好）：")
    for title in ["幾何驗證(直接標註)", "幾何驗證(推導)"]:
        print(f"  {title}")
        for n, v in s[title].items():
            print(f"    {n}: 筆數 {v['可驗證筆數']:,}，99百分位 {v['99百分位']:.2e}，>0.001 比例 {v['殘差>0.001的比例']:.3%}")
    print("各圖元組合的合法選項數：", {k: v["合法選項數"] for k, v in s["合法選項"].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="階段 2-1：建立圖元對標籤")
    ap.add_argument("--limit", type=int, default=None, help="只處理前 N 張草圖（測試用）")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    build(limit=args.limit, workers=args.workers)
