"""
步驟 4：SketchDNN 輸出的後處理
  1. 把接近水平 / 垂直的線拉正
  2. 合併重疊的重複線段
  3. 刪掉太短的碎線
  4. 端點吸附：讓相鄰的線剛好接在角點上
  5. 合併重疊的點
輸出：samples/sample_N_clean.dxf 和 samples/sample_N_compare.png（灰色＝原始，黑色＝修正後）

執行方式（在 SketchDNN 資料夾、sketch_env 環境下）：
    python postprocess.py
"""
import os
import math
import numpy as np

# ===== 可調整的參數（座標單位：模型的 [-1, 1] 範圍）=====
ANGLE_TOL_DEG = 4.0   # 與水平/垂直相差幾度以內就拉正
DUP_DIST      = 0.06  # 兩條平行線相距多近算「重複」
DUP_OVERLAP   = 0.5   # 重疊比例超過多少算「重複」
MIN_LEN       = 0.12  # 比這短的線段刪掉（模型常產生的小碎線）
SNAP_DIST     = 0.08  # 端點相距多近就吸附在一起
CORNER_DIST   = 0.30  # 端點離兩線交點多近，就把兩條線延伸接成角
TRIM_DIST     = 0.08  # 線頭超出交點多少以內，才截掉接成角（超過就當成真的交叉）
POINT_MERGE   = 0.03  # 點相距多近就合併
SIZE_MM       = 100.0 # 匯出 DXF 時，外框邊長（mm）


# ---------- 幾何工具 ----------
def _dir(l):
    d = l[2:4] - l[0:2]
    n = np.linalg.norm(d)
    return d / n if n > 1e-12 else np.zeros(2)

def _length(l):
    return float(np.linalg.norm(l[2:4] - l[0:2]))

def _angle_between(a, b):
    c = abs(float(np.dot(_dir(a), _dir(b))))
    return math.degrees(math.acos(min(1.0, c)))

def _inside(l, p):
    """點 p（已知在直線上）是否落在線段 l 的兩端點之間"""
    d = l[2:4] - l[0:2]
    t = np.dot(p - l[0:2], d) / max(np.dot(d, d), 1e-12)
    return 0.0 < t < 1.0

def _intersect(a, b):
    """兩條無限延伸直線的交點；平行時回傳 None"""
    p, r = a[0:2], a[2:4] - a[0:2]
    q, s = b[0:2], b[2:4] - b[0:2]
    rxs = r[0] * s[1] - r[1] * s[0]
    if abs(rxs) < 1e-9:
        return None
    t = ((q - p)[0] * s[1] - (q - p)[1] * s[0]) / rxs
    return p + t * r


# ---------- 1. 拉正 ----------
def straighten(lines):
    out = []
    for l in lines:
        l = l.copy()
        d = l[2:4] - l[0:2]
        ang = math.degrees(math.atan2(abs(d[1]), abs(d[0])))  # 0=水平, 90=垂直
        if ang < ANGLE_TOL_DEG:
            y = (l[1] + l[3]) / 2; l[1] = l[3] = y
        elif ang > 90 - ANGLE_TOL_DEG:
            x = (l[0] + l[2]) / 2; l[0] = l[2] = x
        out.append(l)
    return out


# ---------- 2. 合併重複線 ----------
def merge_duplicates(lines):
    lines = sorted(lines, key=_length, reverse=True)   # 長的優先保留
    kept = []
    for l in lines:
        merged = False
        for i, k in enumerate(kept):
            if _angle_between(l, k) > ANGLE_TOL_DEG:
                continue
            u = _dir(k)
            nrm = np.array([-u[1], u[0]])
            dist = max(abs(np.dot(l[0:2] - k[0:2], nrm)), abs(np.dot(l[2:4] - k[0:2], nrm)))
            if dist > DUP_DIST:
                continue
            # 投影到 k 的方向上，看重疊多少
            k0, k1 = 0.0, _length(k)
            t = sorted([np.dot(l[0:2] - k[0:2], u), np.dot(l[2:4] - k[0:2], u)])
            overlap = min(k1, t[1]) - max(k0, t[0])
            if overlap > DUP_OVERLAP * min(_length(l), _length(k)):
                lo, hi = min(k0, t[0]), max(k1, t[1])          # 合併成聯集長度
                kept[i] = np.concatenate([k[0:2] + lo * u, k[0:2] + hi * u])
                merged = True
                break
        if not merged:
            kept.append(l.copy())
    return kept


# ---------- 3. 刪碎線 ----------
def remove_short(lines):
    return [l for l in lines if _length(l) >= MIN_LEN]


# ---------- 4. 端點吸附 ----------
def snap_endpoints(lines):
    lines = [l.copy() for l in lines]
    n = len(lines)
    used = set()

    # (a) 角點：兩條不平行的線，各自有一個端點靠近兩線交點 → 兩端都延伸/截到交點
    corners = []
    for i in range(n):
        for j in range(i + 1, n):
            if _angle_between(lines[i], lines[j]) < 15:
                continue
            p = _intersect(lines[i], lines[j])
            if p is None:
                continue
            ei = min((0, 2), key=lambda e: np.linalg.norm(lines[i][e:e+2] - p))
            ej = min((0, 2), key=lambda e: np.linalg.norm(lines[j][e:e+2] - p))
            di = np.linalg.norm(lines[i][ei:ei+2] - p)
            dj = np.linalg.norm(lines[j][ej:ej+2] - p)
            # 延伸量也不能超過線段本身長度的一半，避免短線被硬拉去接遠處
            # 交點若落在線段「裡面」代表要截短；只允許截掉一小段，
            # 否則就是真正的交叉（例如 T 字形），不該接成角
            lim_i = TRIM_DIST if _inside(lines[i], p) else min(CORNER_DIST, 0.5 * _length(lines[i]))
            lim_j = TRIM_DIST if _inside(lines[j], p) else min(CORNER_DIST, 0.5 * _length(lines[j]))
            if di < lim_i and dj < lim_j:
                corners.append((di + dj, i, ei, j, ej, p))
    corners.sort(key=lambda c: c[0])
    for _, i, ei, j, ej, p in corners:
        if (i, ei) in used or (j, ej) in used:
            continue
        lines[i][ei:ei+2] = p
        lines[j][ej:ej+2] = p
        used.add((i, ei)); used.add((j, ej))

    # (b) 其餘靠得很近的端點（例如平行線首尾相接）→ 合到中點
    pairs = []
    for i in range(n):
        for ei in (0, 2):
            for j in range(i + 1, n):
                for ej in (0, 2):
                    d = np.linalg.norm(lines[i][ei:ei+2] - lines[j][ej:ej+2])
                    if d < SNAP_DIST:
                        pairs.append((d, i, ei, j, ej))
    pairs.sort()
    for d, i, ei, j, ej in pairs:
        if (i, ei) in used or (j, ej) in used:
            continue
        m = (lines[i][ei:ei+2] + lines[j][ej:ej+2]) / 2
        lines[i][ei:ei+2] = m
        lines[j][ej:ej+2] = m
        used.add((i, ei)); used.add((j, ej))
    return lines


# ---------- 5. 合併點 ----------
def merge_points(points):
    out = []
    for p in points:
        if all(np.linalg.norm(p - q) >= POINT_MERGE for q in out):
            out.append(p.copy())
    return out


def clean_lines(lines):
    lines = straighten(lines)
    lines = merge_duplicates(lines)
    lines = remove_short(lines)
    lines = snap_endpoints(lines)
    lines = straighten(lines)       # 吸附後再拉正一次
    return lines


# ---------- 從模型輸出取出圖元 ----------
def extract(nodes):
    """nodes: 已經過 utils.ToNaive 的 (16, 21) numpy 陣列"""
    prims = {"line": [], "circle": [], "arc": [], "point": []}
    for p in nodes:
        kind = int(np.argmax(p[2:7]))
        constr = bool(p[0] < 0.5)
        if kind == 0:
            prims["line"].append((p[7:11].astype(float), constr))
        elif kind == 1:
            prims["circle"].append((p[11:14].astype(float), constr))
        elif kind == 2:
            prims["arc"].append((p[14:19].astype(float), constr))
        elif kind == 3:
            prims["point"].append((p[19:21].astype(float), constr))
    return prims


def clean_sketch(prims):
    out = dict(prims)
    for c in (False, True):   # 一般線和輔助線分開處理
        ls = [l for l, cc in prims["line"] if cc == c]
        out.setdefault("_lines", []).extend((l, c) for l in clean_lines(ls))
    out["line"] = out.pop("_lines")
    out["point"] = [(p, False) for p in merge_points([p for p, _ in prims["point"]])]
    return out


# ---------- 輸出 ----------
def save_compare_png(before, after, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, prims, title in ((axes[0], before, "Before"), (axes[1], after, "After")):
        for l, c in prims["line"]:
            ax.plot([l[0], l[2]], [l[1], l[3]], "k--" if c else "k-", lw=1.5)
        for (cx, cy, r), c in prims["circle"]:
            ax.add_patch(plt.Circle((cx, cy), r, fill=False, ls="--" if c else "-"))
        for p, _ in prims["point"]:
            ax.plot(p[0], p[1], "ko", ms=4)
        ax.set_title(title); ax.set_aspect("equal")
        ax.axis("off")
    axes[1].set_xlim(axes[0].get_xlim()); axes[1].set_ylim(axes[0].get_ylim())  # 左右用同一個比例尺
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_dxf(prims, path):
    import ezdxf
    S = SIZE_MM / 2
    doc = ezdxf.new("R2018", setup=True)
    doc.layers.add("GEOMETRY", color=7)
    doc.layers.add("CONSTRUCTION", color=8, linetype="DASHED")
    msp = doc.modelspace()
    lay = lambda c: {"layer": "CONSTRUCTION" if c else "GEOMETRY"}
    for l, c in prims["line"]:
        msp.add_line(tuple(l[0:2] * S), tuple(l[2:4] * S), dxfattribs=lay(c))
    for (cx, cy, r), c in prims["circle"]:
        if r > 1e-6:
            msp.add_circle((cx * S, cy * S), r * S, dxfattribs=lay(c))
    for (cx, cy, r, a0, a1), c in prims["arc"]:
        if r > 1e-6:
            msp.add_arc((cx * S, cy * S), r * S, a0 * 360, a1 * 360, dxfattribs=lay(c))
    for p, c in prims["point"]:
        msp.add_point(tuple(p * S), dxfattribs=lay(c))
    doc.saveas(path)


def main():
    import torch
    import utils
    nodes_all = torch.load(os.path.join("samples", "samples.pt"), weights_only=True)
    for i, nodes in enumerate(nodes_all):
        naive = utils.ToNaive(nodes.clone()).numpy()
        before = extract(naive)
        after = clean_sketch(before)
        save_dxf(after, os.path.join("samples", f"sample_{i}_clean.dxf"))
        save_compare_png(before, after, os.path.join("samples", f"sample_{i}_compare.png"))
        print(f"sample_{i}: 線 {len(before['line'])} → {len(after['line'])}，"
              f"點 {len(before['point'])} → {len(after['point'])}")


if __name__ == "__main__":
    main()
