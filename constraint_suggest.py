"""
階段 3：產生約束建議（給 AutoCAD 介面使用）

輸入：一張圖的圖元 + 使用者選的 1～2 對圖元
輸出：5 個建議（依機率由高到低），每個建議彼此至少有 2 處不同；湊不滿時放寬或補滿，補上的標記「補充」

三種用法：
  1. 介面程式直接呼叫（推薦）：
         from constraint_suggest import Suggester
         s = Suggester("runs_constraint/main/best.pt")
         result = s.suggest(primitives, pairs)        # 回傳 dict（格式見 suggest() 說明）
  2. 用 JSON 檔：
         python constraint_suggest.py --ckpt runs_constraint/main/best.pt --input example_input.json --output out.json
  3. 示範：從驗證資料抽幾張圖，產生建議並畫圖（✓ = 資料中的正確答案）
         python constraint_suggest.py --ckpt runs_constraint/main/best.pt --demo 6

輸入 JSON 格式（座標用圖面的實際單位即可，程式會自動縮放）：
  {
    "primitives": [
      {"type": "line",   "start": [0, 0], "end": [100, 0]},
      {"type": "circle", "center": [50, 30], "radius": 10},
      {"type": "arc",    "center": [0, 0], "radius": 20, "start_angle": 0, "end_angle": 1.5708},
      {"type": "point",  "position": [10, 10]},
      ...（可加 "construction": true 表示輔助線）
      圖元數 ≤ 64 個時整張圖直接給模型；超過時只取選定的圖元與離它們最近的圖元共 16 個（見 constraint_config）
    ],
    "pairs": [[0, 1]]                 ← 1 對；2 對寫成 [[0, 1], [1, 2]]，第 2 對會參考第 1 對的選擇
  }
  弧的角度是弧度、逆時針從 start_angle 到 end_angle（和 AutoCAD 相同）；也可以改給 "start"/"end" 兩個端點座標。
"""
import os
import json
import math
import argparse

import numpy as np

import constraint_config as C

TYPE_KEYS = ["coincident", "horizontal", "vertical", "parallel", "perpendicular", "tangent", "midpoint", "equal"]
SUB_KEYS = {C.SUB_WHOLE: "whole", C.SUB_START: "start", C.SUB_END: "end", C.SUB_CENTER: "center"}
KIND_OF = {"point": C.PRIM_POINT, "line": C.PRIM_LINE, "circle": C.PRIM_CIRCLE, "arc": C.PRIM_ARC}
# 每種圖元有哪些「位置」
VALID_SUB = {C.PRIM_POINT: {C.SUB_WHOLE},
             C.PRIM_LINE: {C.SUB_WHOLE, C.SUB_START, C.SUB_END},
             C.PRIM_CIRCLE: {C.SUB_WHOLE, C.SUB_CENTER},
             C.PRIM_ARC: {C.SUB_WHOLE, C.SUB_START, C.SUB_END, C.SUB_CENTER}}
N_PICK = 5
MIN_DIFF = 2


# =====================================================================
# 輸入轉換
# =====================================================================
def raw_primitives(prims):
    """把介面給的圖元轉成 (kinds (n,), 原始座標 geom (n,7), construct (n,))，座標還沒正規化"""
    if not 2 <= len(prims) <= C.SUGGEST_HARD_MAX:
        raise ValueError(f"圖元數量要在 2～{C.SUGGEST_HARD_MAX} 個之間，目前 {len(prims)} 個")
    kinds, geoms, cons = [], [], []
    for i, p in enumerate(prims):
        t = str(p.get("type", "")).lower()
        if t not in KIND_OF:
            raise ValueError(f"第 {i} 個圖元的類型「{t}」不支援（只能是 point / line / circle / arc）")
        k = KIND_OF[t]
        g = np.zeros(C.GEOM_DIM)
        if k == C.PRIM_POINT:
            g[0:2] = g[2:4] = g[4:6] = p["position"]
        elif k == C.PRIM_LINE:
            s, e = np.asarray(p["start"], float), np.asarray(p["end"], float)
            if np.linalg.norm(e - s) < 1e-9:
                raise ValueError(f"第 {i} 個圖元（線）長度為 0")
            g[0:2], g[2:4], g[4:6] = s, e, (s + e) / 2
        elif k == C.PRIM_CIRCLE:
            g[4:6], g[6] = p["center"], abs(float(p["radius"]))
            if g[6] < 1e-9:
                raise ValueError(f"第 {i} 個圖元（圓）半徑為 0")
        else:
            c, r = np.asarray(p["center"], float), abs(float(p["radius"]))
            if r < 1e-9:
                raise ValueError(f"第 {i} 個圖元（弧）半徑為 0")
            if "start" in p and "end" in p:
                s, e = np.asarray(p["start"], float), np.asarray(p["end"], float)
            else:
                a0, a1 = float(p["start_angle"]), float(p["end_angle"])
                s = c + r * np.array([math.cos(a0), math.sin(a0)])
                e = c + r * np.array([math.cos(a1), math.sin(a1)])
            g[0:2], g[2:4], g[4:6], g[6] = s, e, c, r
        kinds.append(k); geoms.append(g); cons.append(bool(p.get("construction", False)))
    return np.array(kinds, np.int64), np.stack(geoms), np.array(cons, bool)


def prepare(kinds, geom, cons):
    """正規化成模型用的格式：kinds (n,)、geom (n,7)、construct (n,)（和訓練資料用同一套正規化規則）"""
    from constraint_geom import normalize
    g = normalize(list(kinds), geom)
    if g is None:
        raise ValueError("整張圖的大小為 0")
    return kinds.astype(np.int64), g.astype(np.float32), cons.astype(bool)


def parse_primitives(prims):
    """（相容舊程式）整張圖轉成模型格式"""
    return prepare(*raw_primitives(prims))


def nearest_subset(kinds, geom, must, k):
    """must（選定的圖元）一定保留，再補上離它們最近的圖元，共 k 個。回傳原始編號陣列（must 在最前面）"""
    from constraint_geom import keypoints
    n = len(kinds)
    if n <= k:
        return np.arange(n)
    pts = keypoints(kinds[None], geom[None])[0]                       # (n,5,2)
    target = pts[list(must)].reshape(-1, 2)
    d = np.linalg.norm(pts[:, :, None, :] - target[None, None], axis=-1).min(axis=(1, 2))
    d[list(must)] = -np.arange(len(must), 0, -1)                      # 依 must 的順序排在最前面
    return np.argsort(d, kind="stable")[:k]


def check_pairs(pairs, n):
    if not 1 <= len(pairs) <= 2:
        raise ValueError("只能選 1 或 2 對圖元")
    out = []
    for p in pairs:
        i, j = int(p[0]), int(p[1])
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"圖元對 {p} 的編號超出範圍（共有 {n} 個圖元，編號 0～{n - 1}）")
        if i == j:
            raise ValueError(f"圖元對 {p} 是同一個圖元")
        out.append((i, j))
    if len(out) == 2 and set(out[0]) == set(out[1]):
        raise ValueError("兩對圖元不能相同")
    return out


# =====================================================================
# 挑選 5 個建議（純 numpy，可單獨測試）
# =====================================================================
def sub_valid_mask(ka, kb):
    """(128,) 這兩種圖元在「位置」上說得通的選項（不看類型合不合理）"""
    m = np.zeros(C.NUM_OPTIONS, bool)
    for o in range(C.NUM_OPTIONS):
        _, sa, sb = C.option_split(o)
        m[o] = sa in VALID_SUB[ka] and sb in VALID_SUB[kb]
    return m


def option_order(prob, logit, legal, ka, kb):
    """回傳 (合法選項依機率排序, 補充選項依模型分數排序)"""
    legal_ids = np.where(legal)[0]
    legal_ids = legal_ids[np.argsort(-prob[legal_ids], kind="stable")]
    filler = np.where(~legal & sub_valid_mask(ka, kb))[0]
    filler = filler[np.argsort(-logit[filler], kind="stable")]
    return legal_ids, filler


def _fields(combo):
    """一個建議的欄位：1 對 = (類型, A位置, B位置)；2 對 = 6 個欄位"""
    out = []
    for o in combo:
        out.extend(C.option_split(int(o)))
    return tuple(out)


def _n_diff(f1, f2):
    return sum(x != y for x, y in zip(f1, f2))


def constraint_key(i, pos_i, j, pos_j, type_key):
    """一個約束的識別碼（和兩個圖元的先後順序無關）：(圖元, 位置, 圖元, 位置, 類型)"""
    a, b = (i, pos_i), (j, pos_j)
    if b < a:
        a, b = b, a
    return (a[0], a[1], b[0], b[1], type_key)


def combo_keys(combo, canon):
    out = []
    for (a, b), o in zip(canon, combo):
        t, sa, sb = C.option_split(int(o))
        out.append(constraint_key(a, SUB_KEYS[sa], b, SUB_KEYS[sb], TYPE_KEYS[t]))
    return out


def drop_existing(cands, canon, exclude):
    """拿掉「所有約束都已經加過」的候選（例如之前透過介面套用過的），由後面的候選遞補。回傳 (剩下的, 拿掉幾個)"""
    if not exclude:
        return cands, 0
    keep = [c for c in cands if not all(k in exclude for k in combo_keys(c[0], canon))]
    return keep, len(cands) - len(keep)


def greedy_pick(candidates, n=N_PICK, min_diff=MIN_DIFF):
    """candidates：依優先順序排好的 [(combo, score, is_filler)]
    依序挑：先要求和已選的每一個都差 min_diff 處；不夠就放寬到差 1 處；再不夠就用補充選項"""
    chosen, chosen_f = [], []
    for need, allow_filler in ((min_diff, False), (1, False), (1, True)):
        for combo, score, is_filler in candidates:
            if len(chosen) >= n:
                break
            if is_filler and not allow_filler:
                continue
            f = _fields(combo)
            if any(_n_diff(f, g) < need for g in chosen_f):
                continue
            chosen.append({"combo": combo, "score": score, "filler": is_filler,
                           "relaxed": need < min_diff})
            chosen_f.append(f)
        if len(chosen) >= n:
            break
    return chosen


def candidates_single(prob, logit, legal, ka, kb):
    legal_ids, filler = option_order(prob, logit, legal, ka, kb)
    return [((int(o),), float(prob[o]), False) for o in legal_ids] + \
           [((int(o),), None, True) for o in filler]


def candidates_double(p1, logit1, legal1, k1, p2_given, logit2_given, legal2, k2, max_first=None):
    """p1 (128,)：第 1 對的機率；p2_given (L1,128)：在第 1 對選了 legal1 第 i 個選項的條件下，第 2 對的機率
    聯合分數 = p1 × p2|1"""
    l1, f1 = option_order(p1, logit1, legal1, *k1)
    cands = []
    for i, o1 in enumerate(l1):
        for o2 in np.where(legal2)[0]:
            cands.append(((int(o1), int(o2)), float(p1[o1] * p2_given[i, o2]), False))
    cands.sort(key=lambda x: -x[1])
    # 補充：至少一對用到不合法的選項，依「兩對各自排名的和」排序
    order2 = np.argsort(-p2_given.max(0) * legal2 + (~legal2) * -1e9, kind="stable")
    l2 = [o for o in order2 if legal2[o]]
    _, f2 = option_order(p2_given.max(0), logit2_given.max(0), legal2, *k2)
    list1 = [(int(o), False) for o in l1] + [(int(o), True) for o in f1]
    list2 = [(int(o), False) for o in l2] + [(int(o), True) for o in f2]
    fill = []
    for r1, (o1, a) in enumerate(list1[:20]):
        for r2, (o2, b) in enumerate(list2[:20]):
            if a or b:
                fill.append((r1 + r2, (o1, o2)))
    fill.sort(key=lambda x: x[0])
    cands += [(combo, None, True) for _, combo in fill]
    return cands


# =====================================================================
# 模型推論
# =====================================================================
class Suggester:
    def __init__(self, ckpt, device=None):
        import torch
        from constraint_model import ConstraintModel
        self.torch = torch
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(ckpt, map_location=self.dev, weights_only=False)
        a = ck.get("args", {})
        self.model = ConstraintModel(d=a.get("dim", C.MODEL_DIM), layers=a.get("layers", C.MODEL_LAYERS),
                                     heads=a.get("heads", C.MODEL_HEADS)).to(self.dev)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.ckpt = ckpt
        self.epoch = ck.get("state", {}).get("epoch")

    def _run(self, K, G, S, qa, qb, ca=None, cb=None, copt=None, h=None):
        """回傳 (機率, 原始分數, 合法遮罩, 編碼)，形狀 (Q,128)"""
        torch = self.torch
        t = lambda x, dt: torch.as_tensor(np.asarray(x), dtype=dt, device=self.dev)
        kinds, geom, cons = t(K[None], torch.long), t(G[None], torch.float32), t(S[None], torch.bool)
        with torch.no_grad():
            if h is None:
                h = self.model.encode(kinds, geom, cons)
            kw = {}
            if ca is not None:
                kw = dict(ca=t([ca], torch.long), cb=t([cb], torch.long), copt=t([copt], torch.long),
                          has_cond=torch.ones(1, len(ca), dtype=torch.bool, device=self.dev))
            logits, legal = self.model(kinds, geom, cons, t([qa], torch.long), t([qb], torch.long), h=h, **kw)
        lg = logits.float()[0].cpu().numpy()
        lm = legal[0].cpu().numpy()
        return 1 / (1 + np.exp(-np.clip(lg, -50, 50))) * lm, lg, lm, h

    def score_pairs(self, primitives, pairs, batch=512):
        """批次功能用：一次算很多圖元對。回傳 (canon 圖元對 [(a,b)] a<b, 機率 (Q,128), 合法遮罩 (Q,128))
        圖元數 ≤ SUGGEST_DIRECT_MAX：整張圖編碼一次；超過：每一對各自裁出附近的 SUGGEST_CROP_K 個圖元"""
        torch = self.torch
        kinds, raw, cons = raw_primitives(primitives)
        n = len(kinds)
        canon = sorted({(min(int(a), int(b)), max(int(a), int(b))) for a, b in pairs if a != b})
        if not canon:
            return [], np.zeros((0, C.NUM_OPTIONS)), np.zeros((0, C.NUM_OPTIONS), bool)
        t = lambda x, dt: torch.as_tensor(np.asarray(x), dtype=dt, device=self.dev)
        probs, legals = [], []
        with torch.no_grad():
            if n <= C.SUGGEST_DIRECT_MAX:
                K, G, S = prepare(kinds, raw, cons)
                kk, gg, ss = t(K[None], torch.long), t(G[None], torch.float32), t(S[None], torch.bool)
                h = self.model.encode(kk, gg, ss)
                for s0 in range(0, len(canon), batch):
                    part = canon[s0:s0 + batch]
                    qa = t([[a for a, _ in part]], torch.long)
                    qb = t([[b for _, b in part]], torch.long)
                    logits, legal = self.model(kk, gg, ss, qa, qb, h=h)
                    probs.append(torch.sigmoid(logits.float())[0].cpu().numpy())
                    legals.append(legal[0].cpu().numpy())
            else:
                Kc = C.SUGGEST_CROP_K
                for s0 in range(0, len(canon), batch):
                    part = canon[s0:s0 + batch]
                    KB = np.full((len(part), Kc), C.PRIM_NONE, np.int64)
                    GB = np.zeros((len(part), Kc, C.GEOM_DIM), np.float32)
                    SB = np.zeros((len(part), Kc), bool)
                    for r, (a, b) in enumerate(part):
                        sel = nearest_subset(kinds, raw, [a, b], Kc)          # a、b 會排在第 0、1 個
                        k_, g_, s_ = prepare(kinds[sel], raw[sel], cons[sel])
                        KB[r, :len(sel)], GB[r, :len(sel)], SB[r, :len(sel)] = k_, g_, s_
                    zeros = t(np.zeros((len(part), 1)), torch.long)
                    logits, legal = self.model(t(KB, torch.long), t(GB, torch.float32), t(SB, torch.bool),
                                               zeros, zeros + 1)
                    probs.append(torch.sigmoid(logits.float())[:, 0].cpu().numpy())
                    legals.append(legal[:, 0].cpu().numpy())
        return canon, np.concatenate(probs), np.concatenate(legals).astype(bool)

    def suggest(self, primitives, pairs, exclude=None):
        """primitives：圖元清單（格式見檔案開頭）；pairs：[[i, j]] 或 [[i, j], [k, l]]
        回傳 dict：{"pairs": ..., "suggestions": [ {rank, probability, filler, relaxed, constraints:[...]} ×5 ], ...}
          probability：1 對 = 該約束成立的機率；2 對 = 兩個約束同時成立的機率（p1 × p2|1）
          filler=true：合理選項不夠，為了補滿 5 個而加入的「補充」選項（資料中幾乎沒出現過，probability 為 null）
          relaxed=true：為了湊滿 5 個，這個建議和其他建議只差 1 處
        exclude：已經加過的約束（constraint_key 的集合，編號用 primitives 的編號）；所有約束都已加過的建議不會出現"""
        kinds, raw, cons = raw_primitives(primitives)
        n = len(kinds)
        pairs = check_pairs(pairs, n)
        canon = [(min(i, j), max(i, j)) for i, j in pairs]
        kk = [(int(kinds[a]), int(kinds[b])) for a, b in canon]
        # 圖不大就整張給模型；太大就只取選定的圖元 + 最近的幾個（大草圖測試顯示兩者成績幾乎相同）
        if n <= C.SUGGEST_DIRECT_MAX:
            sel = np.arange(n)
            context = {"mode": "whole", "mode_zh": f"整張圖（{n} 個圖元）", "used": n}
        else:
            must = list(dict.fromkeys([x for pr in canon for x in pr]))
            sel = nearest_subset(kinds, raw, must, C.SUGGEST_CROP_K)
            context = {"mode": "crop", "mode_zh": f"只取選定圖元附近的 {len(sel)} 個（全圖 {n} 個）",
                       "used": int(len(sel)), "used_indices": [int(x) for x in sel]}
        K, G, S = prepare(kinds[sel], raw[sel], cons[sel])
        loc = {int(g): l for l, g in enumerate(sel)}
        lc = [(loc[a], loc[b]) for a, b in canon]      # 模型裡的編號；保持「第一個/第二個」的角色不變
        p1, lg1, lm1, h = self._run(K, G, S, [lc[0][0]], [lc[0][1]])
        p1, lg1, lm1 = p1[0], lg1[0], lm1[0].astype(bool)
        if len(pairs) == 1:
            cands = candidates_single(p1, lg1, lm1, *kk[0])
        else:
            l1, _ = option_order(p1, lg1, lm1, *kk[0])
            a2, b2 = lc[1]
            Q = len(l1)
            p2, lg2, lm2, _ = self._run(K, G, S, [a2] * Q, [b2] * Q, [lc[0][0]] * Q, [lc[0][1]] * Q,
                                        [int(o) for o in l1], h=h)
            cands = candidates_double(p1, lg1, lm1, kk[0], p2, lg2, lm2[0].astype(bool), kk[1])
        cands, n_excluded = drop_existing(cands, canon, exclude)
        picked = greedy_pick(cands)
        res = format_result(picked, pairs, canon, kinds, self)
        res["context"] = context
        res["excluded"] = n_excluded
        return res


def format_result(picked, pairs, canon, K, suggester=None):
    out = []
    for r, item in enumerate(picked):
        cons = []
        for (i, j), (a, b), o in zip(pairs, canon, item["combo"]):
            t, sa, sb = C.option_split(o)
            if (i, j) != (a, b):              # 換回使用者給的順序
                sa, sb = sb, sa
            ki, kj = int(K[i]), int(K[j])
            cons.append({
                "pair": [i, j], "type": TYPE_KEYS[t], "type_zh": C.TIDX_NAMES_ZH[t],
                "first": {"index": i, "position": SUB_KEYS[sa], "position_zh": C.SUB_NAMES_ZH[sa]},
                "second": {"index": j, "position": SUB_KEYS[sb], "position_zh": C.SUB_NAMES_ZH[sb]},
                "text_zh": f"{C.TIDX_NAMES_ZH[t]}：{C.PRIM_NAMES_ZH[ki]}{i} 的{C.SUB_NAMES_ZH[sa]} 與 "
                           f"{C.PRIM_NAMES_ZH[kj]}{j} 的{C.SUB_NAMES_ZH[sb]}",
            })
        out.append({"rank": r + 1, "probability": None if item["score"] is None else round(item["score"], 4),
                    "filler": item["filler"], "relaxed": item["relaxed"], "constraints": cons})
    return {"pairs": [list(p) for p in pairs], "suggestions": out,
            "model": None if suggester is None else {"checkpoint": suggester.ckpt, "epoch": suggester.epoch}}


def print_result(res):
    for s in res["suggestions"]:
        p = "  補充  " if s["probability"] is None else f"{s['probability']:7.2%}"
        tag = "（補充）" if s["filler"] else ("（放寬：只差 1 處）" if s["relaxed"] else "")
        print(f"  第{s['rank']}名 {p}  " + "；".join(c["text_zh"] for c in s["constraints"]) + tag)


# =====================================================================
# 示範：從驗證資料抽圖
# =====================================================================
def demo(sg, n, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc as _Arc
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "Noto Sans CJK TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    d = np.load(C.DATA_FILE)
    L = np.load(C.LABELS_FILE)
    kinds_all, geom_all, cons_all, split = d["kinds"], d["geom"], d["construct"], d["split"]
    ps, pa, pb, dm = L["pair_sketch"], L["pair_a"], L["pair_b"], L["direct_mask"]
    rng = np.random.default_rng(seed)
    test_ids = np.where(split == 2)[0]
    out_dir = os.path.join(C.RUNS_DIR, "suggest_demo")
    os.makedirs(out_dir, exist_ok=True)
    done = 0
    while done < n:
        sk = int(rng.choice(test_ids))
        lo, hi = np.searchsorted(ps, sk), np.searchsorted(ps, sk + 1)
        if hi - lo < 2:
            continue
        idx = rng.choice(np.arange(lo, hi), size=2 if done % 2 else 1, replace=False)
        prims = to_primitives(kinds_all[sk], geom_all[sk], cons_all[sk])
        pairs = [[int(pa[i]), int(pb[i])] for i in idx]
        truth = [set(np.where(np.unpackbits(dm[i], bitorder="little"))[0].tolist()) for i in idx]
        res = sg.suggest(prims, pairs)
        print(f"\n測試草圖 #{sk}，選了 {len(pairs)} 對：{pairs}")
        print("  資料中的正確答案：" + " ／ ".join(
            "；".join(C.option_name_zh(o, int(kinds_all[sk, a]), int(kinds_all[sk, b]), a, b) for o in sorted(t))
            for t, (a, b) in zip(truth, pairs)))
        lines = []
        for s in res["suggestions"]:
            ok = all(C.option_id(C.T_COINCIDENT + TYPE_KEYS.index(c["type"]),
                                 [k for k, v in SUB_KEYS.items() if v == c["first"]["position"]][0],
                                 [k for k, v in SUB_KEYS.items() if v == c["second"]["position"]][0]) in t
                     for c, t in zip(s["constraints"], truth))
            s["_ok"] = ok
            p = "補充" if s["probability"] is None else f"{s['probability']:.1%}"
            lines.append(f"{'✓' if ok else '  '} 第{s['rank']}名 {p}  " + "\n        ".join(c["text_zh"] for c in s["constraints"])
                         + ("（補充）" if s["filler"] else ""))
        print_result(res)
        fig, (ax, tx) = plt.subplots(1, 2, figsize=(14, 6))
        sel = {x for p in pairs for x in p}
        for i, (k, g) in enumerate(zip(kinds_all[sk], geom_all[sk])):
            if k < 0:
                continue
            col = "red" if i in sel else "gray"
            lw = 2.5 if i in sel else 1
            ls = "--" if cons_all[sk, i] else "-"
            if k == C.PRIM_POINT:
                ax.plot(g[0], g[1], "o", color=col); lx, ly = g[0], g[1]
            elif k == C.PRIM_LINE:
                ax.plot([g[0], g[2]], [g[1], g[3]], ls, color=col, lw=lw)
                ax.plot(g[0], g[1], ">", color=col, ms=5); lx, ly = g[4], g[5]
            elif k == C.PRIM_CIRCLE:
                ax.add_patch(plt.Circle((g[4], g[5]), g[6], fill=False, color=col, lw=lw, ls=ls)); lx, ly = g[4] + g[6], g[5]
            else:
                a0 = math.degrees(math.atan2(g[1] - g[5], g[0] - g[4])); a1 = math.degrees(math.atan2(g[3] - g[5], g[2] - g[4]))
                ax.add_patch(_Arc((g[4], g[5]), 2 * g[6], 2 * g[6], theta1=a0, theta2=a1, color=col, lw=lw, ls=ls))
                ax.plot(g[0], g[1], ">", color=col, ms=5); lx, ly = g[0], g[1]
            ax.text(lx, ly, str(i), color=col, fontsize=12, weight="bold")
        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect("equal")
        ax.set_title(f"測試草圖 #{sk}（紅色 = 選取的圖元，三角形 = 起點）")
        tx.axis("off")
        tx.text(0, 1, f"選取：{pairs}\n\n模型建議（✓ = 資料中的正確答案）：\n" + "\n".join(lines), va="top", fontsize=10)
        path = os.path.join(out_dir, f"demo_{sk}.png")
        fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
        print("  圖已存到", path)
        done += 1


def to_primitives(kinds, geom, cons):
    """把資料集裡的一張圖轉成介面的輸入格式（示範與測試用）"""
    out = []
    for k, g, c in zip(kinds, geom, cons):
        if k < 0:
            continue
        g = g.astype(float)
        if k == C.PRIM_POINT:
            out.append({"type": "point", "position": g[0:2].tolist()})
        elif k == C.PRIM_LINE:
            out.append({"type": "line", "start": g[0:2].tolist(), "end": g[2:4].tolist()})
        elif k == C.PRIM_CIRCLE:
            out.append({"type": "circle", "center": g[4:6].tolist(), "radius": float(g[6])})
        else:
            out.append({"type": "arc", "center": g[4:6].tolist(), "radius": float(g[6]),
                        "start": g[0:2].tolist(), "end": g[2:4].tolist()})
        out[-1]["construction"] = bool(c)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="產生約束建議")
    ap.add_argument("--ckpt", required=True, help="模型檔，例如 runs_constraint/main/best.pt")
    ap.add_argument("--input", help="輸入 JSON 檔")
    ap.add_argument("--output", help="輸出 JSON 檔（不給就只印在畫面上）")
    ap.add_argument("--demo", type=int, default=0, help="從測試資料抽幾張圖示範")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="強制用 CPU")
    args = ap.parse_args()
    sg = Suggester(args.ckpt, "cpu" if args.cpu else None)
    print(f"模型：{args.ckpt}（訓練了 {sg.epoch} 個 epoch）")
    if args.input:
        with open(args.input, encoding="utf-8") as f:
            req = json.load(f)
        res = sg.suggest(req["primitives"], req["pairs"])
        print_result(res)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            print("已存到", args.output)
    if args.demo:
        demo(sg, args.demo, args.seed)
    if not args.input and not args.demo:
        print("請給 --input 或 --demo")
