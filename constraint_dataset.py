"""
階段 1：把 SketchGraphs 資料整理成「圖元 + 圖元之間的約束」的精簡格式

和原本 dataset.py 的差別：
  * 直接解析 SketchGraphs 的序列格式（不繞經 Sketch 物件），速度較快
  * 同一對圖元的多個約束全部保留（原本只留第一個）
  * 約束只記錄實際存在的那幾筆，不配置 16x16x17 的大陣列（原本開約束會需要約 37 GB 記憶體）
  * 用多個 CPU 核心平行處理

用法（在專案資料夾、sketchdnn 環境下）：
  小量測試（每個原始檔只取前 3000 張，幾分鐘內完成）：
      python constraint_dataset.py build --limit 3000
  正式處理全部資料：
      python constraint_dataset.py build
  畫幾張檢查圖，確認約束有對到正確的圖元：
      python constraint_dataset.py show --n 6
"""
import os
import sys
import json
import time
import math
import argparse
import collections
import multiprocessing as mp

import numpy as np

import constraint_config as C

# 讓 Python 找得到 SketchGraphs 套件（原始資料是用 sketchgraphs 的類別存的，讀檔時一定要能 import）
_SG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SketchGraphs")
if _SG_PATH not in sys.path:
    sys.path.insert(0, _SG_PATH)

from sketchgraphs.data.sequence import NodeOp, EdgeOp                       # noqa: E402
from sketchgraphs.data._entity import EntityType, SubnodeType, ENTITY_TYPE_TO_CLASS  # noqa: E402
from sketchgraphs.data._constraint import ConstraintType                    # noqa: E402

_ENTITY_MAP = {EntityType.Point: C.PRIM_POINT, EntityType.Line: C.PRIM_LINE,
               EntityType.Circle: C.PRIM_CIRCLE, EntityType.Arc: C.PRIM_ARC}
_SUB_MAP = {SubnodeType.SN_Start: C.SUB_START, SubnodeType.SN_End: C.SUB_END,
            SubnodeType.SN_Center: C.SUB_CENTER}
from constraint_geom import _EPS, normalize as _normalize   # noqa: E402  正規化移到 constraint_geom.py（使用模型時不必載入 SketchGraphs）


# =====================================================================
# 一張草圖的解析
# =====================================================================
def _entity_geometry(kind, ent):
    """回傳 (7 個幾何數字, 是否退化)"""
    g = np.zeros(C.GEOM_DIM, dtype=np.float64)
    if kind == C.PRIM_POINT:
        p = np.array([ent.x, ent.y], dtype=np.float64)
        g[0:2] = g[2:4] = g[4:6] = p
        return g, False
    if kind == C.PRIM_LINE:
        s, e = np.asarray(ent.start_point, float), np.asarray(ent.end_point, float)
        g[0:2], g[2:4], g[4:6] = s, e, (s + e) / 2
        return g, bool(np.linalg.norm(e - s) < _EPS)
    if kind == C.PRIM_CIRCLE:
        g[4:6] = [ent.xCenter, ent.yCenter]
        g[6] = abs(ent.radius)
        return g, bool(abs(ent.radius) < _EPS)
    if kind == C.PRIM_ARC:
        s, e = np.asarray(ent.start_point, float), np.asarray(ent.end_point, float)
        g[0:2], g[2:4] = s, e
        g[4:6] = [ent.xCenter, ent.yCenter]
        g[6] = abs(ent.radius)
        return g, bool(abs(ent.radius) < _EPS or np.linalg.norm(e - s) < _EPS)
    raise ValueError(kind)




def parse_sequence(seq):
    """解析一張草圖的序列。
    回傳 (結果 dict 或 None, 略過原因字串或 None, 約束類型計數 Counter)"""
    kinds, geoms = [], []
    node_map = {}          # 序列中的節點編號 -> (圖元編號, 位置)
    has_unsupported = False
    n_degenerate = 0
    node_idx = -1
    last_main = None
    edges = []
    for op in seq:
        if isinstance(op, NodeOp):
            node_idx += 1
            label = op.label
            if isinstance(label, EntityType):
                last_main = node_idx
                if label in _ENTITY_MAP:
                    kind = _ENTITY_MAP[label]
                    ent = ENTITY_TYPE_TO_CLASS[label](str(node_idx))
                    for k, v in op.parameters.items():
                        setattr(ent, k, v)
                    g, degenerate = _entity_geometry(kind, ent)
                    if degenerate:
                        n_degenerate += 1          # 退化圖元直接丟掉（和原本作法相同），它的約束也跟著丟
                        continue
                    if not np.all(np.isfinite(g)):
                        return None, "數值異常(NaN/無限大)", collections.Counter()
                    node_map[node_idx] = (len(kinds), C.SUB_WHOLE)
                    kinds.append(kind)
                    geoms.append((g, bool(op.parameters.get("isConstruction", False))))
                elif label not in (EntityType.External, EntityType.Stop):
                    has_unsupported = True
            else:   # 子節點（起點/終點/中心）
                if last_main in node_map and label in _SUB_MAP:
                    node_map[node_idx] = (node_map[last_main][0], _SUB_MAP[label])
        elif isinstance(op, EdgeOp):
            edges.append(op)

    if has_unsupported and C.SKIP_UNSUPPORTED:
        return None, "含不支援的圖元(雲形線/橢圓等)", collections.Counter()
    n = len(kinds)
    if n > C.MAX_PRIMS:
        return None, f"圖元超過 {C.MAX_PRIMS} 個", collections.Counter()
    if n < C.MIN_PRIMS:
        return None, f"圖元少於 {C.MIN_PRIMS} 個", collections.Counter()

    geom = np.stack([g for g, _ in geoms])
    construct = np.array([c for _, c in geoms], dtype=bool)
    geom = _normalize(kinds, geom)
    if geom is None:
        return None, "整張圖大小為 0", collections.Counter()

    pairs, singles = set(), set()
    type_count = collections.Counter()         # 依原始 ConstraintType 統計（含沒用到的類型）
    for op in edges:
        if op.label == ConstraintType.Subnode:
            continue
        t = int(op.label)
        refs = op.references
        if 0 in refs:
            type_count[("外部參考", t)] += 1
            continue
        if any(r not in node_map for r in refs):
            type_count[("參考到被丟掉的圖元", t)] += 1
            continue
        mapped = [node_map[r] for r in refs]
        if len(mapped) == 1:
            (a, sa), = mapped
            singles.add((a, t, sa))
            type_count[("單一圖元", t)] += 1
        elif len(mapped) == 2:
            (a, sa), (b, sb) = mapped
            if a == b:      # 同一個圖元自己的兩個位置（例如線的起點和終點水平）→ 視為單一圖元約束
                singles.add((a, t, sa * 4 + sb))
                type_count[("同圖元兩位置", t)] += 1
                continue
            if a > b:
                a, sa, b, sb = b, sb, a, sa
            pairs.add((a, b, t, sa, sb))
            type_count[("圖元對", t)] += 1
        else:
            type_count[("三個以上圖元", t)] += 1

    if not pairs:
        return None, "沒有任何圖元對約束", type_count

    result = {
        "kinds": np.array(kinds, dtype=np.int8),
        "construct": construct,
        "geom": geom.astype(np.float32),
        "pairs": np.array(sorted(pairs), dtype=np.int16).reshape(-1, 5),
        "singles": np.array(sorted(singles), dtype=np.int16).reshape(-1, 3),
        "n_degenerate": n_degenerate,
    }
    return result, None, type_count


# =====================================================================
# 讀檔與平行處理
# =====================================================================
def load_sequences(path):
    """讀取 SketchGraphs 的序列檔（記憶體映射，不會整個讀進記憶體）"""
    from sketchgraphs.data import flat_array
    return flat_array.load_dictionary_flat(path)["sequences"]


_WORKER_SEQS = {}


def _worker_init(paths):
    if _SG_PATH not in sys.path:
        sys.path.insert(0, _SG_PATH)
    for p in paths:
        _WORKER_SEQS[p] = load_sequences(p)


def _process_chunk(job):
    path, start, end = job
    seqs = _WORKER_SEQS[path] if path in _WORKER_SEQS else load_sequences(path)
    return _process_range(seqs, start, end)


def _process_range(seqs, start, end):
    out = {"kinds": [], "construct": [], "geom": [], "pairs": [], "singles": []}
    reasons = collections.Counter()
    type_count = collections.Counter()
    n_degenerate = 0
    for i in range(start, end):
        try:
            res, reason, tc = parse_sequence(seqs[i])
        except Exception as e:                    # 單張壞掉不影響整體
            res, reason, tc = None, f"解析錯誤:{type(e).__name__}", collections.Counter()
        type_count.update(tc)
        if res is None:
            reasons[reason] += 1
            continue
        reasons["保留"] += 1
        n_degenerate += res["n_degenerate"]
        for k in out:
            out[k].append(res[k])
    return out, reasons, type_count, n_degenerate


def download(raw_dir):
    import requests
    from tqdm import tqdm
    os.makedirs(raw_dir, exist_ok=True)
    for name in C.RAW_FILE_NAMES:
        path = os.path.join(raw_dir, name)
        if os.path.exists(path):
            continue
        print(f"下載 {name} ...")
        tmp = path + ".part"
        with requests.get(C.DOWNLOAD_URL.format(name), stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=name, ncols=80) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
        os.replace(tmp, path)       # 下載完整才改名，避免半個檔案被當成已下載


def _dedup(kinds_pad, geom_pad):
    """去除重複草圖：把圖元類型和量化後的座標組成指紋，相同指紋只留第一張"""
    q = np.round(geom_pad * (C.DEDUP_LEVELS / 2)).astype(np.int32)
    seen = set()
    keep = []
    for i in range(len(kinds_pad)):
        key = kinds_pad[i].tobytes() + q[i].tobytes()
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return np.array(keep, dtype=np.int64)


def build(limit=None, workers=None, sources=None):
    """sources: 測試用，可直接給 [(名稱, 序列list), ...]；正式執行時為 None（讀原始檔）"""
    t_dl = time.time()
    if sources is None:
        download(C.RAW_DIR)
    t_dl = time.time() - t_dl
    t0 = time.time()
    if sources is None:
        paths = [os.path.join(C.RAW_DIR, n) for n in C.RAW_FILE_NAMES]
        sizes = []
        for p in paths:
            n = len(load_sequences(p))
            sizes.append(n if limit is None else min(n, limit))
            print(f"{os.path.basename(p)}: 共 {n:,} 張，本次處理 {sizes[-1]:,} 張")
    else:
        paths = [name for name, _ in sources]
        sizes = [len(s) if limit is None else min(len(s), limit) for _, s in sources]

    chunk = 5000
    jobs = [(p, s, min(s + chunk, n)) for p, n in zip(paths, sizes) for s in range(0, n, chunk)]
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)

    results = []
    from tqdm import tqdm
    if workers <= 1 or sources is not None:
        src = dict(sources) if sources is not None else {}
        for p, s, e in tqdm(jobs, desc="處理中", ncols=80):
            seqs = src[p] if p in src else load_sequences(p)
            results.append(_process_range(seqs, s, e))
    else:
        print(f"使用 {workers} 個 CPU 核心平行處理")
        with mp.Pool(workers, initializer=_worker_init, initargs=(paths,)) as pool:
            for r in tqdm(pool.imap(_process_chunk, jobs), total=len(jobs), desc="處理中", ncols=80):
                results.append(r)

    # ---- 合併 ----
    reasons, type_count = collections.Counter(), collections.Counter()
    kinds_l, cons_l, geom_l, pairs_l, singles_l = [], [], [], [], []
    n_degenerate = 0
    for out, rs, tc, nd in results:
        reasons.update(rs); type_count.update(tc); n_degenerate += nd
        kinds_l += out["kinds"]; cons_l += out["construct"]; geom_l += out["geom"]
        pairs_l += out["pairs"]; singles_l += out["singles"]
    N = len(kinds_l)
    if N == 0:
        raise RuntimeError("沒有任何草圖被保留，請檢查原始資料")
    P = C.MAX_PRIMS
    kinds = np.full((N, P), C.PRIM_NONE, dtype=np.int8)
    construct = np.zeros((N, P), dtype=bool)
    geom = np.zeros((N, P, C.GEOM_DIM), dtype=np.float32)
    for i, (k, c, g) in enumerate(zip(kinds_l, cons_l, geom_l)):
        kinds[i, :len(k)] = k; construct[i, :len(k)] = c; geom[i, :len(k)] = g

    keep = _dedup(kinds, geom)
    n_dup = N - len(keep)
    kinds, construct, geom = kinds[keep], construct[keep], geom[keep]
    pairs_l = [pairs_l[i] for i in keep]
    singles_l = [singles_l[i] for i in keep]

    def _flatten(lst, width):
        offsets = np.zeros(len(lst) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(x) for x in lst])
        rows = np.concatenate(lst).astype(np.int16) if offsets[-1] > 0 else np.zeros((0, width), np.int16)
        return offsets, rows.reshape(-1, width)

    pair_off, pair_rows = _flatten(pairs_l, 5)
    single_off, single_rows = _flatten(singles_l, 3)

    # ---- 切分 ----
    M = len(keep)
    perm = np.random.default_rng(C.SPLIT_SEED).permutation(M)
    n_tr = int(M * C.SPLIT_RATIO[0]); n_va = int(M * C.SPLIT_RATIO[1])
    split = np.zeros(M, dtype=np.int8)            # 0 訓練, 1 驗證, 2 測試
    split[perm[n_tr:n_tr + n_va]] = 1
    split[perm[n_tr + n_va:]] = 2

    os.makedirs(C.OUT_DIR, exist_ok=True)
    np.savez(C.DATA_FILE, kinds=kinds, construct=construct, geom=geom,
             pair_offsets=pair_off, pair_rows=pair_rows,
             single_offsets=single_off, single_rows=single_rows, split=split)

    stats = make_stats(kinds, pair_off, pair_rows, single_rows, reasons, type_count,
                       n_dup, n_degenerate, split, time.time() - t0)
    stats["下載秒數(已下載過則接近0)"] = round(t_dl, 1)
    stats["每秒處理草圖數"] = round(sum(reasons.values()) / max(stats["處理秒數"], 1e-9), 1)
    with open(C.STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print_stats(stats)
    print(f"\n資料已存到 {C.DATA_FILE}（{os.path.getsize(C.DATA_FILE) / 2**20:.1f} MB）")
    print(f"統計已存到 {C.STATS_FILE}")
    return stats


# =====================================================================
# 統計
# =====================================================================
def _cname(t):
    return C.CONSTRAINT_NAMES_ZH.get(int(t), str(t))


def make_stats(kinds, pair_off, pair_rows, single_rows, reasons, type_count,
               n_dup, n_degenerate, split, seconds):
    N = len(kinds)
    sketch_of_row = np.repeat(np.arange(N), np.diff(pair_off))
    a, b, t, sa, sb = (pair_rows[:, i].astype(np.int64) for i in range(5))
    ka = kinds[sketch_of_row, a].astype(np.int64)
    kb = kinds[sketch_of_row, b].astype(np.int64)

    # 同一對圖元有幾個約束
    pair_key = sketch_of_row * 256 + a * 16 + b
    _, per_pair = np.unique(pair_key, return_counts=True)
    multi = collections.Counter(per_pair.tolist())

    # 以「圖元類型組合」統計各種約束選項（類型, A 位置, B 位置）的次數 → 之後的基準方法與「選項夠不夠 5 個」的依據
    swap = ka > kb
    k1, k2 = np.where(swap, kb, ka), np.where(swap, ka, kb)
    s1, s2 = np.where(swap, sb, sa), np.where(swap, sa, sb)
    combo = collections.defaultdict(collections.Counter)
    for x in zip(k1.tolist(), k2.tolist(), t.tolist(), s1.tolist(), s2.tolist()):
        combo[(x[0], x[1])][(x[2], x[3], x[4])] += 1
    combo_out = {}
    for (x1, x2), cnt in sorted(combo.items()):
        name = f"{C.PRIM_NAMES_ZH[x1]}-{C.PRIM_NAMES_ZH[x2]}"
        total = sum(cnt.values())
        target = {k: v for k, v in cnt.items() if k[0] in C.TARGET_CONSTRAINTS}
        combo_out[name] = {
            "總數": total,
            "出現過的選項數(全部類型)": len(cnt),
            "出現過的選項數(8種目標約束)": len(target),
            "佔該組合 >=0.5% 的目標選項數": sum(1 for v in target.values() if v / total >= 0.005),
            "前10名": [[f"{_cname(k[0])}({C.SUB_NAMES_ZH[k[1]]}-{C.SUB_NAMES_ZH[k[2]]})", v, round(v / total, 4)]
                      for k, v in cnt.most_common(10)],
        }

    pair_types = collections.Counter(t.tolist())
    total_pairs = len(t)
    type_table = {}
    for (where, tt), v in sorted(type_count.items(), key=lambda x: -x[1]):
        type_table.setdefault(where, {})[_cname(tt)] = v

    prims_per = (kinds >= 0).sum(1)
    return {
        "處理秒數": round(seconds, 1),
        "草圖處理結果": dict(reasons.most_common()),
        "去除重複草圖": n_dup,
        "丟掉的退化圖元(長度或半徑為0)": n_degenerate,
        "最終草圖數": int(N),
        "切分(訓練/驗證/測試)": [int((split == i).sum()) for i in range(3)],
        "每張圖元數 平均/最小/最大": [round(float(prims_per.mean()), 2), int(prims_per.min()), int(prims_per.max())],
        "圖元類型數量": {C.PRIM_NAMES_ZH[k]: int((kinds == k).sum()) for k in C.PRIM_NAMES_ZH},
        "圖元對約束總數": int(total_pairs),
        "圖元對約束 各類型": {_cname(k): [v, round(v / total_pairs, 4)] for k, v in pair_types.most_common()},
        "8種目標約束佔圖元對約束比例": round(sum(v for k, v in pair_types.items() if k in C.TARGET_CONSTRAINTS) / total_pairs, 4),
        "同一對圖元的約束數 分布": {str(k): v for k, v in sorted(multi.items())},
        "單一圖元約束總數": int(len(single_rows)),
        "所有約束來源統計(含被排除的)": type_table,
        "各圖元類型組合的約束選項": combo_out,
    }


def print_stats(s):
    print("\n========== 統計 ==========")
    for k in ["下載秒數(已下載過則接近0)", "處理秒數", "每秒處理草圖數", "草圖處理結果", "去除重複草圖", "丟掉的退化圖元(長度或半徑為0)", "最終草圖數",
              "切分(訓練/驗證/測試)", "每張圖元數 平均/最小/最大", "圖元類型數量", "圖元對約束總數",
              "8種目標約束佔圖元對約束比例", "同一對圖元的約束數 分布", "單一圖元約束總數"]:
        print(f"{k}: {s[k]}")
    print("圖元對約束 各類型（次數, 比例）:")
    for k, v in s["圖元對約束 各類型"].items():
        print(f"    {k}: {v[0]:,} ({v[1]:.1%})")
    print("各圖元類型組合（總數 / 8種目標約束中出現過的選項數 / 佔比>=0.5%的選項數）:")
    for k, v in s["各圖元類型組合的約束選項"].items():
        print(f"    {k}: {v['總數']:,} / {v['出現過的選項數(8種目標約束)']} / {v['佔該組合 >=0.5% 的目標選項數']}")
    print("（完整內容請看 stats.json）")


# =====================================================================
# 檢查圖：把幾張草圖和它的約束畫出來，人工確認對得上
# =====================================================================
def show(n=6, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "Noto Sans CJK TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    d = np.load(C.DATA_FILE)
    kinds, geom, cons = d["kinds"], d["geom"], d["construct"]
    po, pr = d["pair_offsets"], d["pair_rows"]
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(kinds), size=min(n, len(kinds)), replace=False)
    colors = plt.get_cmap("tab10")
    for idx in ids:
        fig, (ax, tx) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1]})
        for i, (k, g) in enumerate(zip(kinds[idx], geom[idx])):
            if k < 0:
                continue
            ls = "--" if cons[idx, i] else "-"
            col = colors(i % 10)
            if k == C.PRIM_POINT:
                ax.plot(g[0], g[1], "o", color=col); lx, ly = g[0], g[1]
            elif k == C.PRIM_LINE:
                ax.plot([g[0], g[2]], [g[1], g[3]], ls, color=col)
                ax.plot(g[0], g[1], ">", color=col, ms=5)       # 三角形 = 起點
                lx, ly = g[4], g[5]
            elif k == C.PRIM_CIRCLE:
                ax.add_patch(plt.Circle((g[4], g[5]), g[6], fill=False, ls=ls, color=col)); lx, ly = g[4] + g[6], g[5]
            else:
                a0 = math.degrees(math.atan2(g[1] - g[5], g[0] - g[4]))
                a1 = math.degrees(math.atan2(g[3] - g[5], g[2] - g[4]))
                from matplotlib.patches import Arc as _Arc
                ax.add_patch(_Arc((g[4], g[5]), 2 * g[6], 2 * g[6], theta1=a0, theta2=a1, ls=ls, color=col))
                ax.plot(g[0], g[1], ">", color=col, ms=5)
                mid = math.radians((a0 + ((a1 - a0) % 360) / 2))
                lx, ly = g[4] + g[6] * math.cos(mid), g[5] + g[6] * math.sin(mid)
            ax.text(lx, ly, str(i), color=col, fontsize=12, weight="bold")
        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect("equal")
        ax.set_title(f"草圖 #{idx}（數字=圖元編號，三角形=起點，虛線=輔助線）")
        lines = []
        for a, b, t, sa, sb in pr[po[idx]:po[idx + 1]]:
            ka, kb = C.PRIM_NAMES_ZH[int(kinds[idx, a])], C.PRIM_NAMES_ZH[int(kinds[idx, b])]
            lines.append(f"{_cname(t)}： {ka}{a} 的{C.SUB_NAMES_ZH[int(sa)]} 與 {kb}{b} 的{C.SUB_NAMES_ZH[int(sb)]}")
        tx.axis("off")
        tx.text(0, 1, "圖元對約束：\n" + "\n".join(lines[:40]), va="top", fontsize=10)
        path = os.path.join(C.OUT_DIR, f"check_{idx}.png")
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print("已存", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="階段 1：整理約束資料")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="下載並處理資料")
    b.add_argument("--limit", type=int, default=None, help="每個原始檔只處理前 N 張（測試用）")
    b.add_argument("--workers", type=int, default=None, help="使用幾個 CPU 核心（預設：全部減 1）")
    s = sub.add_parser("show", help="畫檢查圖")
    s.add_argument("--n", type=int, default=6)
    s.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "build":
        build(limit=args.limit, workers=args.workers)
    else:
        show(n=args.n, seed=args.seed)
