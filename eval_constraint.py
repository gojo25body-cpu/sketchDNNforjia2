"""
階段 5：評估——「圖畫得不準」時，模型還準不準？

訓練資料（SketchGraphs）的草圖都已經求解完成：該重合的點本來就疊在一起、該平行的線角度差剛好是 0。
但使用者在 AutoCAD 畫的圖還沒加約束、本來就不精準。這支程式在「測試集」上把圖元座標加上不同程度的隨機偏移，
模擬沒畫準的圖，量測命中率會掉多少。

用法（在專案資料夾、sketchdnn 環境下）：
  python eval_constraint.py                                  評估 runs_constraint/main/best.pt
  python eval_constraint.py --ckpt runs_constraint/xxx/best.pt --name xxx
  python eval_constraint.py --cpu                            用 CPU（不跟訓練搶顯示卡，較慢）
輸出：runs_constraint/eval_<名稱>/noise_eval.json、noise_eval.png

雜訊怎麼加（每個圖元各自獨立地偏移，所以原本疊在一起的端點會分開，就像手畫時沒對準）：
  點、線的端點：x、y 各加常態分布雜訊（標準差 σ）
  圓：圓心偏移 σ，半徑增減 σ
  弧：圓心偏移、半徑增減，起訖點沿弧移動約 σ 的距離（起訖點仍落在弧上）
  σ 以「草圖外框邊長」的百分比表示：草圖縮放到邊長 2，所以 σ = 0.02 就是外框的 1%
"""
import os
import json
import time
import argparse

import numpy as np

import constraint_config as C

LEVELS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]     # σ（正規化座標）；草圖外框邊長 = 2


def add_noise(kinds, geom, sigma, rng):
    """kinds (B,P)、geom (B,P,7) → 加了雜訊的 geom（不改原陣列）
    sigma 可以是一個數字（整批同樣大小），或長度 B 的陣列（每張草圖各自的大小，0 = 不加）"""
    sig = np.asarray(sigma, dtype=np.float64)
    if not np.any(sig > 0):
        return geom
    S = np.broadcast_to(sig.reshape(-1, 1) if sig.ndim == 1 else sig, kinds.shape)   # 每個圖元的 σ
    g = geom.astype(np.float64).copy()
    N = lambda m, k: rng.standard_normal((int(m.sum()), k)) * S[m][:, None]

    m = (kinds == C.PRIM_POINT) & (S > 0)      # σ = 0 的草圖完全不動
    if m.any():
        g[m, 0:2] += N(m, 2)
        g[m, 2:4] = g[m, 0:2]; g[m, 4:6] = g[m, 0:2]
    m = (kinds == C.PRIM_LINE) & (S > 0)      # σ = 0 的草圖完全不動
    if m.any():
        g[m, 0:2] += N(m, 2); g[m, 2:4] += N(m, 2)
        g[m, 4:6] = (g[m, 0:2] + g[m, 2:4]) / 2
    m = (kinds == C.PRIM_CIRCLE) & (S > 0)      # σ = 0 的草圖完全不動
    if m.any():
        g[m, 4:6] += N(m, 2)
        r = g[m, 6]
        g[m, 6] = np.maximum(r + N(m, 1)[:, 0], 0.2 * r)
    m = (kinds == C.PRIM_ARC) & (S > 0)      # σ = 0 的草圖完全不動
    if m.any():
        c, r = g[m, 4:6], g[m, 6]
        a0 = np.arctan2(g[m, 1] - c[:, 1], g[m, 0] - c[:, 0])
        a1 = np.arctan2(g[m, 3] - c[:, 1], g[m, 2] - c[:, 0])
        c2 = c + N(m, 2)
        r2 = np.maximum(r + N(m, 1)[:, 0], 0.2 * r)
        da = np.clip(N(m, 2) / np.maximum(r2, 1e-6)[:, None], -0.5, 0.5)     # 沿弧移動約 σ 的距離
        a0, a1 = a0 + da[:, 0], a1 + da[:, 1]
        g[m, 0:2] = c2 + r2[:, None] * np.stack([np.cos(a0), np.sin(a0)], 1)
        g[m, 2:4] = c2 + r2[:, None] * np.stack([np.cos(a1), np.sin(a1)], 1)
        g[m, 4:6], g[m, 6] = c2, r2
    g[kinds < 0] = 0
    return g.astype(np.float32)


def run(args):
    import torch
    import torch.nn.functional as F
    from constraint_model import ConstraintModel
    from train_constraint import Data, to_device

    dev = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    a = ck.get("args", {})
    model = ConstraintModel(d=a.get("dim", C.MODEL_DIM), layers=a.get("layers", C.MODEL_LAYERS),
                            heads=a.get("heads", C.MODEL_HEADS)).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    epoch = ck.get("state", {}).get("epoch")
    derived = a.get("derived", "off")
    print(f"模型：{args.ckpt}（訓練了 {epoch} 個 epoch，推導約束 {derived}），裝置 {dev}")

    data = Data(derived, val_sketches=1)
    d = np.load(C.DATA_FILE)
    split = d["split"]
    test_all = np.where((split == 2) & (data.npairs > 0))[0]
    test_ids = np.sort(np.random.default_rng(321).choice(test_all, min(args.sketches, len(test_all)), replace=False))
    print(f"測試草圖 {len(test_ids):,} 張（從 {len(test_all):,} 張測試集中隨機抽），每張 {args.pairs} 個圖元對")

    # 先把要評估的圖元對固定下來，每個雜訊等級都用同一批，比較才公平
    rng = np.random.default_rng(999)
    batches = [data.make_batch(test_ids[s:s + args.batch], args.pairs, rng, 0.0)
               for s in range(0, len(test_ids), args.batch)]
    counts = torch.from_numpy(data.counts).to(dev).float()
    amp = dev.type == "cuda" and torch.cuda.is_bf16_supported()

    results = []
    for li, sigma in enumerate(LEVELS):
        t0 = time.time()
        nrng = np.random.default_rng(1000 + li)
        acc = {k: 0.0 for k in ["n", "h1", "h5", "n_nc", "h5_nc", "n_l", "h5_l", "b1", "b5"]}
        per_n, per_h = np.zeros(C.NUM_TYPES), np.zeros(C.NUM_TYPES)
        with torch.no_grad():
            for b in batches:
                b2 = dict(b)
                b2["geom"] = add_noise(b["kinds"], b["geom"], sigma, nrng)
                bt = to_device(b2, dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    logits, legal = model(bt["kinds"], bt["geom"], bt["construct"], bt["qa"], bt["qb"])
                prob = torch.sigmoid(logits.float()) * legal
                dm, vm = bt["dbits"], bt["vbits"]
                strict, loose = dm & legal, (dm | vm) & legal
                valid = strict.any(-1)
                top5 = prob.topk(5, dim=-1).indices
                in5 = torch.zeros_like(strict).scatter_(-1, top5, torch.ones_like(top5, dtype=torch.bool))
                hit5 = (in5 & strict).any(-1)
                acc["n"] += float(valid.sum())
                acc["h1"] += float((strict.gather(-1, top5[..., :1]).squeeze(-1) & valid).sum())
                acc["h5"] += float((hit5 & valid).sum())
                nc = valid & ~strict[..., :16].any(-1)
                acc["n_nc"] += float(nc.sum()); acc["h5_nc"] += float((hit5 & nc).sum())
                lv = loose.any(-1)
                acc["n_l"] += float(lv.sum()); acc["h5_l"] += float(((in5 & loose).any(-1) & lv).sum())
                for t in range(C.NUM_TYPES):
                    sl = slice(t * 16, (t + 1) * 16)
                    has_t = strict[..., sl].any(-1)
                    per_n[t] += float(has_t.sum())
                    per_h[t] += float(((in5 & strict)[..., sl].any(-1) & has_t).sum())
                ka = bt["kinds"].gather(1, bt["qa"]); kb = bt["kinds"].gather(1, bt["qb"])
                base = counts[ka, kb] * legal + torch.rand_like(prob) * 1e-3
                bt5 = base.topk(5, dim=-1).indices
                bh = strict.gather(-1, bt5)
                acc["b1"] += float((bh[..., 0] & valid).sum()); acc["b5"] += float((bh.any(-1) & valid).sum())
        n = max(acc["n"], 1)
        r = {
            "雜訊σ": sigma, "雜訊佔外框邊長%": round(sigma / 2 * 100, 3),
            "前1名命中率": acc["h1"] / n, "前5名命中率": acc["h5"] / n,
            "前5名命中率(非重合)": acc["h5_nc"] / max(acc["n_nc"], 1),
            "前5名命中率(寬鬆)": acc["h5_l"] / max(acc["n_l"], 1),
            "基準 前1名": acc["b1"] / n, "基準 前5名": acc["b5"] / n,
            "各類型 前5名找回率": {C.TIDX_NAMES_ZH[t]: (round(per_h[t] / per_n[t], 4) if per_n[t] else None)
                             for t in range(C.NUM_TYPES)},
            "各類型 樣本數": {C.TIDX_NAMES_ZH[t]: int(per_n[t]) for t in range(C.NUM_TYPES)},
            "圖元對數": int(acc["n"]), "秒數": round(time.time() - t0, 1),
        }
        results.append(r)
        print(f"σ={sigma:<6}（外框 {r['雜訊佔外框邊長%']:>5}%）前1名 {r['前1名命中率']:.2%}  前5名 {r['前5名命中率']:.2%}  "
              f"非重合前5名 {r['前5名命中率(非重合)']:.2%}  基準前5名 {r['基準 前5名']:.2%}  （{r['秒數']} 秒）")

    out = os.path.join(C.RUNS_DIR, f"eval_{args.name}")
    os.makedirs(out, exist_ok=True)
    report = {"模型": args.ckpt, "epoch": epoch, "推導約束": derived, "測試草圖數": int(len(test_ids)),
              "每張圖元對數": args.pairs, "雜訊說明": __doc__.split("雜訊怎麼加")[1].strip(), "結果": results}
    with open(os.path.join(out, "noise_eval.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    plot(results, os.path.join(out, "noise_eval.png"), f"{args.name}（epoch {epoch}）")
    print("\n各類型 前5名找回率（正確答案含該類型時，前 5 名裡有沒有找到它）：")
    names = C.TIDX_NAMES_ZH
    print("外框%   " + " ".join(f"{n:>6}" for n in names))
    for r in results:
        print(f"{r['雜訊佔外框邊長%']:>5}  " + " ".join(
            f"{(r['各類型 前5名找回率'][n] or 0):>7.1%}" for n in names))
    print(f"\n結果已存到 {out}")


def plot(results, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "Noto Sans CJK TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    x = [r["雜訊佔外框邊長%"] for r in results]
    xs = [max(v, 0.05) for v in x]           # 對數座標放不下 0，第一點畫在 0.05 的位置並標註
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, style in [("前5名命中率", "o-"), ("前1名命中率", "s-"), ("前5名命中率(非重合)", "^-"), ("基準 前5名", "x--")]:
        ax.plot(xs, [r[key] * 100 for r in results], style, label=key)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(["0（無雜訊）" if v == 0 else f"{v:g}%" for v in x])
    ax.set_xlabel("座標雜訊（標準差，佔草圖外框邊長的百分比）")
    ax.set_ylabel("命中率（%）")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(f"圖畫得越不準，模型還準不準？ {title}")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="雜訊強度 vs 命中率評估")
    ap.add_argument("--ckpt", default="runs_constraint/main/best.pt")
    ap.add_argument("--name", default=None, help="結果資料夾名稱（預設：模型資料夾名稱）")
    ap.add_argument("--sketches", type=int, default=20000, help="評估幾張測試草圖")
    ap.add_argument("--pairs", type=int, default=C.PAIRS_PER_SKETCH)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    args.name = args.name or os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))
    run(args)
