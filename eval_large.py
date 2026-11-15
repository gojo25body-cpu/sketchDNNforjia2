"""
階段 6：大草圖測試——圖元超過 16 個時，模型還能用嗎？

背景：訓練用的 sg_t16 只收錄 16 個圖元以內的草圖（處理統計裡「圖元超過 16 個」被略過的數量是 0）。
所以大草圖要從 SketchGraphs 的「完整版」序列檔 sg_all.npy（約 15 GB）裡抽。
這些大草圖從來沒進過訓練，天然就是沒看過的測試資料。

比較的做法（同一批圖元對、同一個模型，不需要重新訓練）：
  A      ：整張草圖直接餵給模型（模型結構允許，但訓練時沒看過這麼多圖元）
  B16    ：只取「選定那一對 + 離它們最近的 14 個圖元」共 16 個，重新縮放到外框邊長 2 再餵
  B16n   ：同 B16，但不重新縮放（用來分辨「裁掉遠處圖元」和「放大」各自的影響）
  B8     ：同 B16，但只取 8 個圖元

用法（在專案資料夾、sketchdnn 環境下）：
  1. 建立大草圖測試集（第一次會下載 sg_all.npy，約 15 GB，可中斷續傳）：
       python eval_large.py build
       python eval_large.py build --target 500      每個大小區間只抽 500 張（較快）
  2. 評估：
       python eval_large.py eval --ckpt runs_constraint/noise5/best.pt
       python eval_large.py eval --ckpt runs_constraint/noise5/best.pt --noise 0,1,2.5
輸出：data/constraint/large_test.npz、large_test_stats.json
      runs_constraint/eval_large_<模型資料夾名稱>/large_eval.json
"""
import os
import sys
import json
import time
import shutil
import argparse
import collections
import multiprocessing as mp

import numpy as np

import constraint_config as C
from constraint_geom import keypoints, normalize_batch   # noqa: F401  （移到 constraint_geom.py，這裡保留名稱給舊程式用）

MAXP = 64                                            # 這次測試接受的最多圖元數
MIN_N = 17                                           # 16 個以內是訓練資料的範圍
BUCKETS = [(17, 24), (25, 32), (33, 48), (49, 64)]   # 依圖元數分成 4 個區間，各抽一樣多張
ALL_FILE = "sg_all.npy"
ALL_BYTES_APPROX = 15 * 2**30
LARGE_FILE = os.path.join(C.OUT_DIR, "large_test.npz")
LARGE_STATS = os.path.join(C.OUT_DIR, "large_test_stats.json")
MODES_ALL = {"A": (None, True), "B16": (16, True), "B16n": (16, False), "B8": (8, True)}


def bucket_of(n):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= n <= hi:
            return i
    return -1


def bucket_name(i):
    lo, hi = BUCKETS[i]
    return f"{lo}~{hi} 個圖元"


# =====================================================================
# 1. 建立測試集
# =====================================================================
def download_all():
    """下載 sg_all.npy；中斷後重新執行會從斷掉的地方接著下載"""
    import requests
    from tqdm import tqdm
    os.makedirs(C.RAW_DIR, exist_ok=True)
    path = os.path.join(C.RAW_DIR, ALL_FILE)
    if os.path.exists(path):
        return path
    tmp = path + ".part"
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    free = shutil.disk_usage(C.RAW_DIR).free
    if free < ALL_BYTES_APPROX - have + 2**30:
        sys.exit(f"硬碟空間不足：需要約 {(ALL_BYTES_APPROX - have) / 2**30 + 1:.0f} GB，"
                 f"目前只剩 {free / 2**30:.1f} GB。請清出空間後再執行。")
    url = C.DOWNLOAD_URL.format(ALL_FILE)
    print(f"下載 {url}" + (f"（從 {have / 2**30:.2f} GB 處接續）" if have else ""))
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, stream=True, timeout=60, headers=headers) as r:
        r.raise_for_status()
        if have and r.status_code != 206:       # 伺服器不支援續傳 → 從頭下載
            print("伺服器不支援續傳，從頭下載。")
            have = 0
        total = have + int(r.headers.get("content-length", 0))
        with open(tmp, "ab" if have else "wb") as f, \
                tqdm(total=total, initial=have, unit="B", unit_scale=True, desc=ALL_FILE, ncols=80) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    os.replace(tmp, path)
    return path


_W = {}


def _init_worker(path, min_n=None):
    global MIN_N
    import constraint_dataset as D
    C.MAX_PRIMS = MAXP               # 子行程會重新載入設定，要在這裡再改一次上限
    if min_n is not None:
        MIN_N = min_n
    _W["seqs"] = D.load_sequences(path)
    _W["parse"] = D.parse_sequence


def _labels(kinds, pairs, singles):
    """回傳 {(a,b): (直接標註 128 格, 推導 128 格)}，算法與 constraint_labels.py 相同"""
    from constraint_labels import derive_sketch
    out = collections.defaultdict(lambda: (np.zeros(C.NUM_OPTIONS, bool), np.zeros(C.NUM_OPTIONS, bool)))
    for a, b, t, sa, sb in pairs:
        t = int(t)
        if t in C.RAW_TO_TIDX:
            out[(int(a), int(b))][0][C.RAW_TO_TIDX[t] * 16 + int(sa) * 4 + int(sb)] = True
    for a, b, t, sa, sb in derive_sketch(kinds, pairs, singles):
        out[(a, b)][1][t * 16 + sa * 4 + sb] = True
    res = {}
    for k, (dm, vm) in out.items():
        vm = vm & ~dm
        if dm.any() or vm.any():
            res[k] = (dm, vm)
    return res


def _work(idx):
    seqs, parse = _W["seqs"], _W["parse"]
    kept, reasons = [], collections.Counter()
    for i in idx:
        try:
            res, reason, _ = parse(seqs[int(i)])
        except Exception as e:
            res, reason = None, f"解析錯誤:{type(e).__name__}"
        if res is None:
            reasons[reason] += 1
            continue
        n = len(res["kinds"])
        if n < MIN_N:
            reasons["16 個以內（訓練資料的範圍，不用）"] += 1
            continue
        lab = _labels(res["kinds"], res["pairs"], res["singles"])
        if not any(dm.any() for dm, _ in lab.values()):
            reasons["沒有 8 種目標約束"] += 1
            continue
        reasons["符合"] += 1
        kept.append((int(i), res["kinds"], res["construct"], res["geom"], lab))
    return kept, reasons


def build(args):
    import constraint_dataset as D
    t0 = time.time()
    path = os.path.join(C.RAW_DIR, args.source) if args.source else download_all()
    t_dl = time.time() - t0
    print(f"讀取 {path} ...")
    from sketchgraphs.data import flat_array            # constraint_dataset 已設定好路徑
    data = flat_array.load_dictionary_flat(path)
    seqs = data["sequences"]
    N = len(seqs)
    lens = np.asarray(data["sequence_lengths"]) if "sequence_lengths" in data else None
    print(f"共 {N:,} 張草圖")
    # 先用序列長度粗篩：17 個圖元至少要 18 個步驟；太長的（極少數）略過以免卡住
    order = np.random.default_rng(args.seed).permutation(N)
    if lens is not None:
        ln = lens[order]
        order = order[(ln >= 18) & (ln <= 4000)]
        print(f"序列長度粗篩後剩 {len(order):,} 張候選")

    need = {i: args.target for i in range(len(BUCKETS))}
    got = {i: [] for i in range(len(BUCKETS))}
    reasons = collections.Counter()
    seen = set()
    chunk = 2000
    jobs = [order[s:s + chunk] for s in range(0, min(len(order), args.max_scan), chunk)]
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"開始抽樣（每個區間 {args.target} 張，最多檢查 {min(len(order), args.max_scan):,} 張，{workers} 個行程）...")
    scanned = 0
    t1 = time.time()
    # 分批送工作：每一批做完才決定要不要繼續。
    # （之前一次把幾千個工作排進佇列、收集夠了再強制結束行程池，在 Windows 上會卡死）
    finished = False
    pool = mp.Pool(workers, initializer=_init_worker, initargs=(path, MIN_N))
    try:
        per_round = workers * 2
        for r0 in range(0, len(jobs), per_round):
            for kept, rs in pool.map(_work, jobs[r0:r0 + per_round], chunksize=1):
                scanned += chunk
                reasons.update(rs)
                for item in kept:
                    _, kinds, _, geom, _ = item
                    b = bucket_of(len(kinds))
                    if b < 0 or len(got[b]) >= need[b]:
                        continue
                    key = kinds.tobytes() + np.round(geom * 128).astype(np.int16).tobytes()   # 去除重複草圖
                    if key in seen:
                        reasons["重複草圖"] += 1
                        continue
                    seen.add(key)
                    got[b].append(item)
            print(f"  已檢查 {scanned:,} 張，已收集 " +
                  "、".join(f"{bucket_name(i)} {len(got[i])}" for i in got) + f"（{time.time() - t1:.0f} 秒）",
                  flush=True)
            if sum(len(v) for v in got.values()) >= sum(need.values()):
                finished = True
                break
    finally:
        pool.close()          # 佇列已經空了，正常關閉即可，不用強制結束
        pool.join()
    print("收集完成，整理存檔中 ..." if finished else "已達檢查上限，整理存檔中 ...", flush=True)

    items = [it for b in range(len(BUCKETS)) for it in got[b]]
    M = len(items)
    if M == 0:
        sys.exit("一張符合的草圖都沒有找到，請檢查資料檔。")
    kinds = np.full((M, MAXP), C.PRIM_NONE, np.int8)
    construct = np.zeros((M, MAXP), bool)
    geom = np.zeros((M, MAXP, C.GEOM_DIM), np.float32)
    nprim = np.zeros(M, np.int16)
    src = np.zeros(M, np.int64)
    ps, pa, pb, dms, vms = [], [], [], [], []
    for m, (i, k, c, g, lab) in enumerate(items):
        n = len(k)
        kinds[m, :n], construct[m, :n], geom[m, :n] = k, c, g
        nprim[m], src[m] = n, i
        for (a, b), (dm, vm) in sorted(lab.items()):
            ps.append(m); pa.append(a); pb.append(b)
            dms.append(np.packbits(dm, bitorder="little")); vms.append(np.packbits(vm, bitorder="little"))
    np.savez_compressed(LARGE_FILE, kinds=kinds, construct=construct, geom=geom, nprim=nprim, src=src,
                        pair_sketch=np.array(ps, np.int32), pair_a=np.array(pa, np.int16),
                        pair_b=np.array(pb, np.int16), direct_mask=np.array(dms, np.uint8),
                        derived_mask=np.array(vms, np.uint8))
    stats = {
        "來源": path, "下載秒數": round(t_dl, 1), "處理秒數": round(time.time() - t1, 1),
        "檢查的草圖數": scanned, "各區間收集到": {bucket_name(i): len(got[i]) for i in got},
        "篩選結果": dict(reasons.most_common()),
        "圖元對數": len(ps), "有直接標註的圖元對": int(sum(d.any() for d in dms)),
    }
    with open(LARGE_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n已存到 {LARGE_FILE}")
    if any(len(got[i]) < need[i] for i in got):
        print("注意：有區間沒有收滿（大草圖比較少見，或檢查上限 --max-scan 太小），結果仍可用，但該區間樣本較少。")


# =====================================================================
# 2. 評估用的純 numpy 工具（裁切、重新縮放、計分）
# =====================================================================




def crop(kinds, construct, geom, qa, qb, K, renorm):
    """每個查詢：保留 qa、qb 與離它們最近的 K-2 個圖元。回傳 (kinds, construct, geom, 新qa, 新qb)，qa→0、qb→1"""
    Q, P = kinds.shape
    pts = keypoints(kinds, geom)                                     # (Q,P,5,2)
    ar = np.arange(Q)
    pair_pts = np.concatenate([pts[ar, qa], pts[ar, qb]], axis=1)    # (Q,10,2)
    d = np.linalg.norm(pts[:, :, :, None, :] - pair_pts[:, None, None, :, :], axis=-1)   # (Q,P,5,10)
    d = np.where(np.isfinite(d), d, np.inf).min(axis=(2, 3))
    d[kinds < 0] = np.inf
    d[ar, qa], d[ar, qb] = -2.0, -1.0                                # 一定排在最前面，順序 qa、qb
    K = min(K, P)
    sel = np.argsort(d, axis=1, kind="stable")[:, :K]
    ck = np.take_along_axis(kinds, sel, 1).copy()
    cc = np.take_along_axis(construct, sel, 1)
    cg = np.take_along_axis(geom, sel[..., None], 1)
    gone = ~np.isfinite(np.take_along_axis(d, sel, 1))               # 原圖不到 K 個圖元時，多出來的是空格
    ck[gone] = C.PRIM_NONE
    cg = np.where(gone[..., None], 0, cg).astype(np.float32)
    cc = cc & ~gone
    if renorm:
        cg = normalize_batch(ck, cg)
    return ck, cc, cg, np.zeros(Q, np.int64), np.ones(Q, np.int64)


def score(prob, legal, dm):
    """prob、legal、dm：(Q,128)。回傳每個查詢的 (有效, 前1名對, 前5名對, 非重合)"""
    strict = dm & legal
    valid = strict.any(-1)
    p = np.where(legal, prob, -1.0)
    top5 = np.argsort(-p, axis=-1, kind="stable")[:, :5]
    hit = np.take_along_axis(strict, top5, -1)
    nc = valid & ~strict[:, :16].any(-1)          # 正確答案裡沒有「重合」的圖元對
    return valid, hit[:, 0] & valid, hit.any(-1) & valid, nc


# =====================================================================
# 3. 評估
# =====================================================================
def run_eval(args):
    import torch
    from constraint_model import ConstraintModel
    from eval_constraint import add_noise

    dev = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    a = ck.get("args", {})
    model = ConstraintModel(d=a.get("dim", C.MODEL_DIM), layers=a.get("layers", C.MODEL_LAYERS),
                            heads=a.get("heads", C.MODEL_HEADS)).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    epoch = ck.get("state", {}).get("epoch")
    amp = dev.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"模型：{args.ckpt}（epoch {epoch}），裝置 {dev}")

    def predict(kinds, construct, geom, qa, qb):
        """回傳 (機率, 合法遮罩)，都是 (Q,128) numpy；每個查詢自己一列"""
        probs, legals = [], []
        for s in range(0, len(kinds), args.batch):
            sl = slice(s, s + args.batch)
            k = kinds[sl]
            P = int(max(1, (k >= 0).sum(1).max()))          # 只留到這一批最多的圖元數，省計算
            t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits, legal = model(t(k[:, :P]).long(), t(geom[sl][:, :P]).float(),
                                      t(construct[sl][:, :P]).bool(),
                                      t(qa[sl]).long().view(-1, 1), t(qb[sl]).long().view(-1, 1))
            probs.append(torch.sigmoid(logits.float()).squeeze(1).cpu().numpy())
            legals.append(legal.squeeze(1).cpu().numpy())
        return np.concatenate(probs), np.concatenate(legals)

    evaluate(args, predict, model_name=f"{args.ckpt}（epoch {epoch}）", add_noise=add_noise)


def evaluate(args, predict, model_name, add_noise):
    """和模型無關的部分（抽圖元對、加雜訊、裁切、計分），predict 由呼叫端提供"""
    if not os.path.exists(LARGE_FILE):
        sys.exit(f"找不到 {LARGE_FILE}，請先執行：python eval_large.py build")
    D = np.load(LARGE_FILE)
    kinds, construct, geom, nprim = D["kinds"], D["construct"], D["geom"], D["nprim"]
    ps, pa, pb = D["pair_sketch"], D["pair_a"].astype(np.int64), D["pair_b"].astype(np.int64)
    dm_all = np.unpackbits(D["direct_mask"], axis=1, bitorder="little").astype(bool)
    has = dm_all.any(1)
    # 每張草圖固定抽 args.pairs 個有直接標註的圖元對（不重複），所有做法、雜訊等級都用同一批
    rng = np.random.default_rng(999)
    order = np.argsort(ps, kind="stable")
    starts = np.searchsorted(ps[order], np.arange(len(kinds) + 1))
    qidx = []
    for m in range(len(kinds)):
        cand = order[starts[m]:starts[m + 1]]
        cand = cand[has[cand]]
        if len(cand):
            qidx.append(rng.choice(cand, min(args.pairs, len(cand)), replace=False))
    qidx = np.sort(np.concatenate(qidx))
    qs, qa, qb, qdm = ps[qidx], pa[qidx], pb[qidx], dm_all[qidx]
    qbucket = np.array([bucket_of(int(n)) for n in nprim[qs]])
    print(f"大草圖 {len(kinds):,} 張，評估圖元對 {len(qs):,} 個")

    levels = [float(x) for x in args.noise.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]
    results = []
    for li, pct in enumerate(levels):
        sigma = pct / 100 * 2                                # 草圖外框邊長 = 2
        g_noisy = add_noise(kinds, geom, sigma, np.random.default_rng(1000 + li))
        for mode in modes:
            t0 = time.time()
            K, renorm = MODES_ALL[mode]
            k, c, g = kinds[qs], construct[qs], g_noisy[qs]
            a_, b_ = qa, qb
            if K is not None:
                k, c, g, a_, b_ = crop(k, c, g, qa, qb, K, renorm)
            prob, legal = predict(k, c, g, a_, b_)
            valid, h1, h5, nc = score(prob, legal, qdm)
            r = {"雜訊佔外框邊長%": pct, "做法": mode, "秒數": round(time.time() - t0, 1), "各區間": {}}
            for bi in list(range(len(BUCKETS))) + [None]:
                m = (qbucket == bi) if bi is not None else np.ones(len(qs), bool)
                n = int(valid[m].sum())
                entry = {"圖元對數": n,
                         "前1名命中率": float(h1[m].sum() / max(n, 1)),
                         "前5名命中率": float(h5[m].sum() / max(n, 1)),
                         "前5名命中率(非重合)": float((h5 & nc)[m].sum() / max(int((nc & m).sum()), 1))}
                if bi is None:
                    r["全部"] = entry
                else:
                    r["各區間"][bucket_name(bi)] = entry
            results.append(r)
            print(f"雜訊 {pct:>4}%  {mode:<5} 全部：前1名 {r['全部']['前1名命中率']:.2%}  "
                  f"前5名 {r['全部']['前5名命中率']:.2%}  非重合前5名 {r['全部']['前5名命中率(非重合)']:.2%}"
                  f"（{r['秒數']} 秒）")

    # 表格：每個雜訊等級一張，列 = 做法，欄 = 圖元數區間
    print("\n前5名命中率（前1名命中率）")
    for pct in levels:
        print(f"\n雜訊 {pct}% 外框邊長")
        print(f"{'做法':<6}" + "".join(f"{bucket_name(i):>18}" for i in range(len(BUCKETS))) + f"{'全部':>16}")
        for r in results:
            if r["雜訊佔外框邊長%"] != pct:
                continue
            cells = [r["各區間"][bucket_name(i)] for i in range(len(BUCKETS))] + [r["全部"]]
            print(f"{r['做法']:<7}" + "".join(f"{e['前5名命中率']:>9.1%}（{e['前1名命中率']:.1%}）" for e in cells))

    ref = None
    ref_path = os.path.join(C.RUNS_DIR, f"eval_{args.name}", "noise_eval.json")
    if os.path.exists(ref_path):
        with open(ref_path, encoding="utf-8") as f:
            ref = json.load(f)
        print(f"\n對照（16 個以內的一般測試集，{ref_path}）：")
        for r in ref["結果"]:
            if r["雜訊佔外框邊長%"] in levels or (r["雜訊佔外框邊長%"] == 0 and 0 in levels):
                print(f"  雜訊 {r['雜訊佔外框邊長%']}%：前1名 {r['前1名命中率']:.1%}  前5名 {r['前5名命中率']:.1%}")

    out = os.path.join(C.RUNS_DIR, f"eval_large_{args.name}")
    os.makedirs(out, exist_ok=True)
    report = {"模型": model_name, "大草圖數": int(len(kinds)), "評估圖元對數": int(len(qs)),
              "做法說明": {"A": "整張直接餵", "B16": "選定那對+最近14個，重新縮放",
                       "B16n": "同 B16 但不重新縮放", "B8": "選定那對+最近6個，重新縮放"},
              "結果": results,
              "對照(16個以內)": [r for r in ref["結果"] if r["雜訊佔外框邊長%"] in levels] if ref else None}
    with open(os.path.join(out, "large_eval.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n結果已存到 {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="大草圖（17～64 個圖元）測試")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="下載完整版資料並抽出大草圖測試集")
    b.add_argument("--target", type=int, default=1500, help="每個圖元數區間抽幾張")
    b.add_argument("--max-scan", type=int, default=6_000_000, help="最多檢查幾張草圖")
    b.add_argument("--workers", type=int, default=None)
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--source", default=None, help="改用 data/raw 裡的其他檔案（測試用）")
    e = sub.add_parser("eval", help="評估")
    e.add_argument("--ckpt", default="runs_constraint/noise5/best.pt")
    e.add_argument("--name", default=None, help="結果資料夾名稱（預設：模型資料夾名稱）")
    e.add_argument("--noise", default="0,1", help="雜訊等級（佔外框邊長的 %%，逗號分隔）")
    e.add_argument("--modes", default="A,B16,B16n,B8")
    e.add_argument("--pairs", type=int, default=C.PAIRS_PER_SKETCH)
    e.add_argument("--batch", type=int, default=512)
    e.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args)
    else:
        args.name = args.name or os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))
        run_eval(args)
