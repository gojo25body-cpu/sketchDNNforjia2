"""
批次自動找約束：掃過整張圖的「相鄰圖元對」，請模型判斷每一對該加什麼約束，挑出值得加的列成清單

流程：
  1. 找候選圖元對：每個圖元和離它最近的 BATCH_NEIGHBORS 個圖元配對；另外「碰在一起」的一律納入
  2. 模型一次算出所有候選對的 128 個選項機率（Suggester.score_pairs）
  3. 每一對取機率 ≥ BATCH_PROB_MIN 的選項（最多 BATCH_MAX_PER_PAIR 個、類型不同）
  4. 幾何檢查：在目前的圖上算殘差，「幾乎成立」且機率 ≥ BATCH_AUTO_PROB → 標為高把握（預設勾選）
  5. 去掉多餘的：例如 A、B、C 三點已經用兩個「重合」連起來，第三個「重合」就是多餘的（AutoCAD 會判為過度約束）；
     已經透過介面加過的（套用記錄）也拿掉
  6. 依「先拓撲、後方向、最後大小」排序：重合 → 中點 → 相切 → 水平/垂直 → 平行/垂直相交 → 等長
  （選用）規則補強：rules=True 時，另外用純幾何規則補上模型範圍外、或模型把握偏低的約束：
     * 單一條線的水平／垂直：線的角度與水平或垂直相差 ≤ BATCH_GEOM_ANGLE 度
     * 圓角相切：直線與弧（或弧與弧）的端點已經接在一起，而且幾何上已經相切

這個檔案只用 numpy，不需要 AutoCAD 或 PyTorch，可以單獨測試。
"""
import math

import numpy as np

import constraint_config as C
from constraint_suggest import TYPE_KEYS, SUB_KEYS, raw_primitives, constraint_key
from autocad_interface import residual, expand_steps, build_command, is_pointlike

ORDER = {"coincident": 0, "midpoint": 1, "tangent": 2, "horizontal": 3, "vertical": 3,
         "horizontal_line": 3, "vertical_line": 3, "parallel": 4, "perpendicular": 4, "equal": 5}
ZH = {"point": "點", "line": "線", "circle": "圓", "arc": "弧"}
ANGLE_TYPES = ("parallel", "perpendicular")


def drawing_size(kinds, raw):
    from constraint_geom import keypoints
    pts = keypoints(kinds[None], raw[None])[0].reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(1)]
    return float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1e-9))


def neighbor_pairs(kinds, raw, k=None, touch=None):
    """回傳候選圖元對 [(a, b)]（a < b）"""
    from constraint_geom import keypoints
    k = C.BATCH_NEIGHBORS if k is None else k
    touch = C.BATCH_TOUCH if touch is None else touch
    n = len(kinds)
    pts = keypoints(kinds[None], raw[None])[0]                    # (n,5,2)
    size = drawing_size(kinds, raw)
    pairs = set()
    for s in range(0, n, 200):                                     # 分塊算距離，避免大圖吃光記憶體
        blk = pts[s:s + 200]
        d = np.linalg.norm(blk[:, :, None, None, :] - pts[None, None, :, :, :], axis=-1).min(axis=(1, 3))
        d[np.arange(len(blk)), np.arange(s, s + len(blk))] = np.inf
        near = np.argsort(d, axis=1)[:, :k]
        for r in range(len(blk)):
            i = s + r
            js = set(near[r].tolist()) | set(np.where(d[r] <= touch * size)[0].tolist())
            for j in js:
                if j != i:
                    pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[ra] = rb
        return True


def geom_score(ga, pa, gb, pb, c, size):
    """幾何誤差 ÷ 容許值：≤ 1 代表「幾乎成立」；None = 無法計算"""
    worst = 0.0
    for st in expand_steps(c, ga, gb):
        r = residual(ga, pa, gb, pb, st)
        if r is None:
            return None
        if st in ANGLE_TYPES:
            tol = math.sin(math.radians(C.BATCH_GEOM_ANGLE))
        else:
            tol = C.BATCH_GEOM_DIST * size
        worst = max(worst, r / tol)
    return worst


def make_constraint(prims, a, b, o):
    t, sa, sb = C.option_split(int(o))
    ka, kb = prims[a]["type"], prims[b]["type"]
    zh = {"point": "點", "line": "線", "circle": "圓", "arc": "弧"}
    return {
        "pair": [a, b], "type": TYPE_KEYS[t], "type_zh": C.TIDX_NAMES_ZH[t],
        "first": {"index": a, "position": SUB_KEYS[sa], "position_zh": C.SUB_NAMES_ZH[sa]},
        "second": {"index": b, "position": SUB_KEYS[sb], "position_zh": C.SUB_NAMES_ZH[sb]},
        "text_zh": f"{C.TIDX_NAMES_ZH[t]}：{zh[ka]}{a} 的{C.SUB_NAMES_ZH[sa]} 與 {zh[kb]}{b} 的{C.SUB_NAMES_ZH[sb]}",
    }


def single_line_constraint(prims, i, kind):
    zh = "水平" if kind == "horizontal_line" else "垂直"
    return {"single": True, "index": i, "pair": [i, i], "type": kind, "type_zh": f"{zh}（單線）",
            "first": {"index": i, "position": "whole", "position_zh": "整個"},
            "second": {"index": i, "position": "whole", "position_zh": "整個"},
            "text_zh": f"{zh}：線{i}（整條線本身）"}


def rule_constraints(prims, pairs, size):
    """規則補強（不靠模型）：回傳清單，格式和模型的建議相同，另外標 rule=True"""
    from autocad_interface import _unit, keypoint
    out = []
    sin_tol = math.sin(math.radians(C.BATCH_GEOM_ANGLE))
    for i, g in enumerate(prims):
        if g["type"] != "line":
            continue
        u = _unit(g)
        if abs(u[1]) <= sin_tol:
            out.append({"constraint": single_line_constraint(prims, i, "horizontal_line"),
                        "prob": None, "geom": abs(u[1]) / sin_tol, "auto": True, "rule": True})
        elif abs(u[0]) <= sin_tol:
            out.append({"constraint": single_line_constraint(prims, i, "vertical_line"),
                        "prob": None, "geom": abs(u[0]) / sin_tol, "auto": True, "rule": True})
    dist_tol = C.BATCH_GEOM_DIST * size
    for a, b in pairs:
        ka, kb = prims[a]["type"], prims[b]["type"]
        if not ({ka, kb} <= {"line", "arc"} and "arc" in (ka, kb)):
            continue
        ends = lambda g: [keypoint(g, "start"), keypoint(g, "end")]
        touch = min(math.dist(p, q) for p in ends(prims[a]) for q in ends(prims[b]))
        if touch > dist_tol:
            continue                               # 端點沒有接在一起，不是圓角接點
        r = residual(prims[a], "whole", prims[b], "whole", "tangent")
        if r is None or r > dist_tol:
            continue
        c = {"pair": [a, b], "type": "tangent", "type_zh": C.TIDX_NAMES_ZH[C.T_TANGENT],
             "first": {"index": a, "position": "whole", "position_zh": "整個"},
             "second": {"index": b, "position": "whole", "position_zh": "整個"},
             "text_zh": f"{C.TIDX_NAMES_ZH[C.T_TANGENT]}：{ZH[ka]}{a} 的整個 與 {ZH[kb]}{b} 的整個"}
        out.append({"constraint": c, "prob": None, "geom": r / dist_tol, "auto": True, "rule": True})
    return out


def item_key(c):
    a, b = c["pair"]
    return constraint_key(a, c["first"]["position"], b, c["second"]["position"], c["type"])


def select_constraints(prims, pairs, probs, legal, exclude=None, rules=False):
    """prims：介面格式的圖元（AutoCAD 座標）；pairs：[(a,b)]；probs、legal：(Q,128)
    exclude：已經加過的約束（constraint_key，用編號表示）；rules：是否加入規則補強
    回傳 (清單, 統計)。清單每項：{constraint, prob, geom, auto, rule}"""
    exclude = exclude or set()
    kinds, raw, _ = raw_primitives(prims)
    size = drawing_size(kinds, raw)
    stats = {"候選圖元對": len(pairs), "機率夠高的約束": 0, "已經加過": 0, "多餘（可由其他約束推得）": 0,
             "AutoCAD 沒有對應指令": 0, "列出": 0, "高把握": 0}
    rule_items = []
    if rules:
        for x in rule_constraints(prims, pairs, size):
            if item_key(x["constraint"]) in exclude:
                stats["已經加過"] += 1
            else:
                rule_items.append(x)
        stats["規則補強"] = len(rule_items)
    rule_keys = {item_key(x["constraint"]) for x in rule_items}
    cands = []
    for q, (a, b) in enumerate(pairs):
        order = np.argsort(-np.where(legal[q], probs[q], -1.0), kind="stable")
        used_types = set()
        for o in order:
            p = float(probs[q, o])
            if not legal[q, o] or p < C.BATCH_PROB_MIN or len(used_types) >= C.BATCH_MAX_PER_PAIR:
                break
            c = make_constraint(prims, a, b, o)
            if c["type"] in used_types:
                continue
            used_types.add(c["type"])
            stats["機率夠高的約束"] += 1
            pa, pb = c["first"]["position"], c["second"]["position"]
            if constraint_key(a, pa, b, pb, c["type"]) in exclude:
                stats["已經加過"] += 1
                continue
            if constraint_key(a, pa, b, pb, c["type"]) in rule_keys:
                continue                        # 規則已經列了同一個約束

            ga, gb = prims[a], prims[b]
            if any(build_command("A", ga, pa, "B", gb, pb, st, "near") is None for st in expand_steps(c, ga, gb)):
                stats["AutoCAD 沒有對應指令"] += 1
                continue
            gs = geom_score(ga, pa, gb, pb, c, size)
            auto = gs is not None and gs <= 1.0 and p >= C.BATCH_AUTO_PROB
            cands.append({"constraint": c, "prob": p, "geom": gs, "auto": auto, "rule": False})

    # 去掉多餘的約束：規則項（最確定）先佔位置，再來是高把握、機率高的
    cands.sort(key=lambda x: (not x["auto"], -x["prob"]))
    cands = rule_items + cands
    coin, hor, ver, par, eq = _UF(), _UF(), _UF(), _UF(), _UF()
    perp_seen = set()
    out = []
    for x in cands:
        c = x["constraint"]
        a, b = c["pair"]
        pa, pb = c["first"]["position"], c["second"]["position"]
        ka, kb = prims[a]["type"], prims[b]["type"]
        t = c["type"]
        redundant = False
        if t in ("horizontal_line", "vertical_line"):
            # 平行組裡已經有一條被定成水平（垂直），同組其他線就不用再定
            redundant = not par.union(a, "H" if t == "horizontal_line" else "V")
        elif t == "coincident" and is_pointlike(ka, pa) and is_pointlike(kb, pb):
            redundant = not coin.union((a, pa), (b, pb))
        elif t in ("horizontal", "vertical"):
            pa_key, pb_key = coin.find((a, pa)), coin.find((b, pb))
            if pa_key == pb_key:                     # 兩點已經重合，水平/垂直自然成立
                redundant = True
            else:
                redundant = not (hor if t == "horizontal" else ver).union(pa_key, pb_key)
        elif t == "parallel":
            ga_, gb_ = par.find(a), par.find(b)
            hv = {par.find("H"), par.find("V")}
            if ga_ in hv and gb_ in hv and ga_ != gb_:
                redundant = True                     # 一條水平、一條垂直，不可能平行（矛盾），不列
            else:
                redundant = not par.union(a, b)
        elif t == "perpendicular":
            key = tuple(sorted((str(par.find(a)), str(par.find(b)))))
            hv = tuple(sorted((str(par.find("H")), str(par.find("V")))))
            redundant = key in perp_seen or key == hv    # 一條水平、一條垂直，自然垂直相交
            perp_seen.add(key)
        elif t == "equal":
            redundant = not eq.union(a, b)
        if redundant:
            stats["多餘（可由其他約束推得）"] += 1
            continue
        out.append(x)
    out.sort(key=lambda x: (ORDER[x["constraint"]["type"]], not x.get("rule"), not x["auto"], -(x["prob"] or 0)))
    stats["列出"] = len(out)
    stats["高把握"] = sum(x["auto"] for x in out)
    return out, stats


def describe_item(x):
    g = "—" if x["geom"] is None else (f"誤差為容許值的 {x['geom']:.2g} 倍" if x["geom"] > 1 else "幾乎成立")
    if x.get("rule"):
        return f"【規則】{x['constraint']['text_zh']}　{g}"
    tag = "【高把握】" if x["auto"] else "【請確認】"
    return f"{tag}{x['constraint']['text_zh']}　機率 {x['prob']:.0%}　{g}"
