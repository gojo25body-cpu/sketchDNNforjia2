"""
階段 2-3：訓練約束建議模型（含測速與冒煙測試）

用法（在專案資料夾、sketchdnn 環境下）：
  python train_constraint.py --smoke        冒煙測試：用 2 萬張草圖跑 2 個 epoch，確認整條流程能動（約幾分鐘）
  python train_constraint.py --benchmark    測速：產生 runs_constraint/<名稱>/benchmark_report.json，推算到 epoch 1000 的時間
  python train_constraint.py                正式訓練（預設 1000 epoch；可以隨時按 Ctrl+C 中斷）
  python train_constraint.py --resume       從上次中斷的地方接續

常用參數：
  --derived off|soft|on   推導約束怎麼用（off=不用；soft=當作「半個正確答案」；on=當作完整正確答案）
  --batch 512             一步訓練幾張草圖（顯示記憶體不夠時調小）
  --name 名稱             這次實驗的名稱（輸出放在 runs_constraint/名稱/）
"""
import os
import sys
import csv
import json
import math
import time
import queue
import argparse
import platform
import threading

import numpy as np
import torch
import torch.nn.functional as F

import constraint_config as C
from constraint_model import ConstraintModel
from eval_constraint import add_noise

DERIVED_TARGET = {"off": 0.0, "soft": 0.5, "on": 1.0}


# =====================================================================
# 資料
# =====================================================================
class Data:
    def __init__(self, derived="off", val_sketches=C.VAL_SKETCHES, train_limit=None, seed=0):
        if not os.path.exists(C.LABELS_FILE):
            print("找不到標籤檔，先執行 constraint_labels.py ...")
            import constraint_labels
            constraint_labels.build()
        t0 = time.time()
        print("讀取資料 ...")
        d = np.load(C.DATA_FILE)
        self.kinds, self.construct, self.geom = d["kinds"], d["construct"], d["geom"]
        split = d["split"]
        L = np.load(C.LABELS_FILE)
        ps, pa, pb = L["pair_sketch"], L["pair_a"], L["pair_b"]
        dm, vm = L["direct_mask"], L["derived_mask"]
        self.legal, self.counts = L["legal"], L["option_counts"]
        keep = dm.any(1) if derived == "off" else (dm.any(1) | vm.any(1))
        self.pair_sketch, self.pa, self.pb = ps[keep], pa[keep].astype(np.int64), pb[keep].astype(np.int64)
        self.dmask, self.vmask = dm[keep], vm[keep]
        self.derived = derived
        self.soft = DERIVED_TARGET[derived]
        self.noise_max = 0.0          # 0 = 不加雜訊；由 main() 依參數設定
        self.noise_min = 0.002
        self.noise_clean = 0.2
        N = len(self.kinds)
        self.offsets = np.searchsorted(self.pair_sketch, np.arange(N + 1)).astype(np.int64)
        self.npairs = np.diff(self.offsets)
        has = self.npairs > 0
        self.train_ids = np.where((split == 0) & has)[0]
        if train_limit:
            self.train_ids = np.sort(np.random.default_rng(seed).choice(self.train_ids, min(train_limit, len(self.train_ids)), replace=False))
        val_all = np.where((split == 1) & has)[0]
        self.val_ids = np.sort(np.random.default_rng(123).choice(val_all, min(val_sketches, len(val_all)), replace=False))
        print(f"訓練草圖 {len(self.train_ids):,} 張，驗證草圖 {len(self.val_ids):,} 張，"
              f"圖元對 {len(self.pa):,} 個（推導約束：{derived}），讀取 {time.time() - t0:.0f} 秒")

    @staticmethod
    def _bits(m):
        return np.unpackbits(m, axis=-1, bitorder="little")

    def make_batch(self, ids, K, rng, cond_prob):
        off, n = self.offsets[ids][:, None], self.npairs[ids][:, None]
        B = len(ids)
        pick = off + np.minimum((rng.random((B, K)) * n).astype(np.int64), n - 1)
        has_cond = (rng.random((B, K)) < cond_prob) & (n > 1)
        local = pick - off
        r2 = (rng.random((B, K)) * np.maximum(n - 1, 1)).astype(np.int64)
        c_local = r2 + (r2 >= local)
        cpick = np.where(has_cond, off + np.minimum(c_local, n - 1), pick)
        cm = self.dmask[cpick] | self.vmask[cpick] if self.derived != "off" else self.dmask[cpick]
        kinds, geom = self.kinds[ids], self.geom[ids]
        if self.noise_max > 0:
            # 每張草圖隨機抽一個雜訊大小：noise_clean 的比例保持精準，其餘在 [noise_min, noise_max] 之間對數均勻
            lo, hi = math.log(self.noise_min), math.log(self.noise_max)
            sig = np.exp(rng.uniform(lo, hi, B))
            sig[rng.random(B) < self.noise_clean] = 0.0
            geom = add_noise(kinds, geom, sig, rng)
        return {
            "kinds": kinds, "construct": self.construct[ids], "geom": geom,
            "qa": self.pa[pick], "qb": self.pb[pick],
            "dbits": self._bits(self.dmask[pick]), "vbits": self._bits(self.vmask[pick]),
            "ca": self.pa[cpick], "cb": self.pb[cpick], "cbits": self._bits(cm), "has_cond": has_cond,
        }


class Prefetch:
    """背景執行緒先準備好下幾批資料，讓顯示卡不用等 CPU"""
    def __init__(self, fn, n, depth=6):
        self.q = queue.Queue(depth)
        self.n = n

        def run():
            try:
                for i in range(n):
                    self.q.put(fn(i))
            except Exception as e:      # 把錯誤轉交給主執行緒
                self.q.put(e)
        self.t = threading.Thread(target=run, daemon=True)
        self.t.start()

    def __iter__(self):
        for _ in range(self.n):
            x = self.q.get()
            if isinstance(x, Exception):
                raise x
            yield x


def to_device(b, dev):
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev, non_blocking=True)
    return {
        "kinds": t(b["kinds"]).long(), "construct": t(b["construct"]).bool(), "geom": t(b["geom"]).float(),
        "qa": t(b["qa"]).long(), "qb": t(b["qb"]).long(),
        "dbits": t(b["dbits"]).bool(), "vbits": t(b["vbits"]).bool(),
        "ca": t(b["ca"]).long(), "cb": t(b["cb"]).long(), "cbits": t(b["cbits"]).bool(),
        "has_cond": t(b["has_cond"]).bool(),
    }


# =====================================================================
# 訓練 / 驗證的單步
# =====================================================================
def loss_fn(model, bt, soft, amp):
    copt = (torch.rand(bt["cbits"].shape, device=bt["cbits"].device) * bt["cbits"]).argmax(-1)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        logits, legal = model(bt["kinds"], bt["geom"], bt["construct"], bt["qa"], bt["qb"],
                              bt["ca"], bt["cb"], copt, bt["has_cond"])
    d, v = bt["dbits"].float(), bt["vbits"].float()
    target = d + (1 - d) * v * soft
    el = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    lm = legal.float()
    return (el * lm).sum() / lm.sum().clamp(min=1)


@torch.no_grad()
def evaluate(model, data, dev, amp, batch=1024, K=C.PAIRS_PER_SKETCH):
    """前 1 / 前 5 名命中率：模型給的前 k 個選項裡，有沒有至少一個是正確答案
    嚴格 = 只算資料裡直接標註的約束；寬鬆 = 推導出的也算對
    基準 = 不看圖，只依「這種圖元組合最常出現的選項」排序"""
    model.eval()
    rng = np.random.default_rng(999)
    counts = torch.from_numpy(data.counts).to(dev).float()
    acc = {k: 0.0 for k in ["n", "n_nc", "loss", "hit1", "hit5", "hit5_nc", "base1", "base5", "n_loose", "hit5_loose"]}
    n_batches = 0
    for s in range(0, len(data.val_ids), batch):
        ids = data.val_ids[s:s + batch]
        bt = to_device(data.make_batch(ids, K, rng, 0.0), dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits, legal = model(bt["kinds"], bt["geom"], bt["construct"], bt["qa"], bt["qb"])
        logits = logits.float()
        d, v = bt["dbits"], bt["vbits"]
        target = d.float() + (~d).float() * v.float() * data.soft
        el = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        acc["loss"] += float((el * legal).sum() / legal.sum().clamp(min=1))
        n_batches += 1
        prob = torch.sigmoid(logits) * legal
        strict, loose = d & legal, (d | v) & legal
        valid = strict.any(-1)
        top5 = prob.topk(5, dim=-1).indices
        hit = strict.gather(-1, top5)
        acc["n"] += float(valid.sum())
        acc["hit1"] += float((hit[..., 0] & valid).sum())
        acc["hit5"] += float((hit.any(-1) & valid).sum())
        nc = valid & ~strict[..., :16].any(-1)                    # 正確答案裡沒有「重合」的圖元對
        acc["n_nc"] += float(nc.sum())
        acc["hit5_nc"] += float((hit.any(-1) & nc).sum())
        lv = loose.any(-1)
        acc["n_loose"] += float(lv.sum())
        acc["hit5_loose"] += float((loose.gather(-1, top5).any(-1) & lv).sum())
        ka = bt["kinds"].gather(1, bt["qa"]); kb = bt["kinds"].gather(1, bt["qb"])
        base = counts[ka, kb] * legal + torch.rand_like(prob) * 1e-3   # 加一點雜訊打破平手
        bt5 = base.topk(5, dim=-1).indices
        bh = strict.gather(-1, bt5)
        acc["base1"] += float((bh[..., 0] & valid).sum())
        acc["base5"] += float((bh.any(-1) & valid).sum())
    model.train()
    n = max(acc["n"], 1)
    return {
        "val_loss": acc["loss"] / max(n_batches, 1),
        "前1名命中率": acc["hit1"] / n, "前5名命中率": acc["hit5"] / n,
        "前5名命中率(非重合)": acc["hit5_nc"] / max(acc["n_nc"], 1),
        "前5名命中率(寬鬆)": acc["hit5_loose"] / max(acc["n_loose"], 1),
        "基準 前1名": acc["base1"] / n, "基準 前5名": acc["base5"] / n,
    }


# =====================================================================
# 建立模型、存檔
# =====================================================================
def build_model(args, data, dev):
    model = ConstraintModel(d=args.dim, layers=args.layers, heads=args.heads).to(dev)
    model.legal.copy_(torch.from_numpy(data.legal).to(dev))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=C.WEIGHT_DECAY)
    return model, opt


def make_sched(opt, total_steps):
    warm_steps = min(C.WARMUP_STEPS, max(1, total_steps // 10))   # 冒煙測試步數很少時，暖身也跟著縮短

    def f(s):
        warm = min(1.0, (s + 1) / warm_steps)
        prog = min(1.0, s / max(total_steps, 1))
        return warm * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def save_ckpt(path, model, opt, sched, state, args):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "state": state, "args": vars(args)}, tmp)
    os.replace(tmp, path)       # 先寫暫存檔再改名：存到一半當機也不會弄壞舊的存檔


def fmt_time(sec):
    sec = float(sec)
    if sec < 120:
        return f"{sec:.1f} 秒"
    if sec < 7200:
        return f"{sec / 60:.1f} 分鐘"
    if sec < 172800:
        return f"{sec / 3600:.1f} 小時"
    return f"{sec / 86400:.1f} 天（{sec / 3600:.0f} 小時）"


# =====================================================================
# 正式訓練 / 冒煙測試
# =====================================================================
def train(args, data, dev):
    out = os.path.join(C.RUNS_DIR, args.name)
    os.makedirs(out, exist_ok=True)
    model, opt = build_model(args, data, dev)
    B, K = args.batch, args.pairs
    steps_per_epoch = math.ceil(len(data.train_ids) / B)
    sched = make_sched(opt, args.epochs * steps_per_epoch)
    state = {"epoch": 0, "step_in_epoch": 0, "global_step": 0, "best": -1.0}
    last_path, best_path = os.path.join(out, "last.pt"), os.path.join(out, "best.pt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        state = ck["state"]
        print(f"從 epoch {state['epoch']} 第 {state['step_in_epoch']} 步接續")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型參數 {n_params / 1e6:.2f} M，每個 epoch {steps_per_epoch:,} 步（每步 {B} 張草圖 × {K} 對）")
    with open(os.path.join(out, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    csv_path = os.path.join(out, "epochs.csv")
    new_csv = not os.path.exists(csv_path)
    csv_f = open(csv_path, "a", newline="", encoding="utf-8-sig")
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(out, "logs"))
    except Exception:
        pass

    first_losses, last_losses = [], []
    log_every = 5 if args.smoke else 50
    try:
        while state["epoch"] < args.epochs:
            ep = state["epoch"]
            perm = np.random.default_rng(args.seed + ep).permutation(data.train_ids)
            start = state["step_in_epoch"]
            rng = np.random.default_rng(args.seed * 100003 + ep * 7919 + start)
            fn = lambda i: to_device(data.make_batch(perm[(start + i) * B:(start + i + 1) * B], K, rng, C.COND_PROB), dev)
            t_ep = time.time(); t_save = time.time(); run_loss = 0.0; run_n = 0
            for i, bt in enumerate(Prefetch(fn, steps_per_epoch - start)):
                loss = loss_fn(model, bt, data.soft, args.amp)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
                opt.step(); sched.step()
                state["global_step"] += 1
                state["step_in_epoch"] = start + i + 1
                if state["global_step"] % log_every == 0:
                    lv = float(loss)
                    run_loss += lv; run_n += 1
                    if len(first_losses) < 4:
                        first_losses.append(lv)
                    last_losses = (last_losses + [lv])[-4:]
                    if writer:
                        writer.add_scalar("train/loss", lv, state["global_step"])
                if state["global_step"] % 500 == 0 or (args.smoke and state["global_step"] % 20 == 0):
                    done = state["step_in_epoch"] / steps_per_epoch
                    el = time.time() - t_ep
                    eta = el / max(state["step_in_epoch"] - start, 1) * (steps_per_epoch - state["step_in_epoch"])
                    print(f"epoch {ep} {done:6.1%} 步 {state['step_in_epoch']}/{steps_per_epoch} "
                          f"loss {run_loss / max(run_n, 1):.4f} 學習率 {sched.get_last_lr()[0]:.2e} 本 epoch 剩約 {fmt_time(eta)}")
                    run_loss = 0.0; run_n = 0
                if time.time() - t_save > C.SAVE_EVERY_MINUTES * 60:
                    save_ckpt(last_path, model, opt, sched, state, args); t_save = time.time()
            train_sec = time.time() - t_ep
            m = evaluate(model, data, dev, args.amp)
            ep_sec = time.time() - t_ep
            state["epoch"] += 1; state["step_in_epoch"] = 0
            is_best = m["前5名命中率"] > state["best"]
            if is_best:
                state["best"] = m["前5名命中率"]
            save_ckpt(last_path, model, opt, sched, state, args)
            if is_best:
                save_ckpt(best_path, model, opt, sched, state, args)
            row = {"epoch": ep, "訓練秒數": round(train_sec, 1), "epoch總秒數": round(ep_sec, 1),
                   "學習率": sched.get_last_lr()[0], **{k: round(v, 5) for k, v in m.items()},
                   "時間": time.strftime("%Y-%m-%d %H:%M:%S")}
            if new_csv:
                csv.DictWriter(csv_f, fieldnames=list(row)).writeheader(); new_csv = False
            csv.DictWriter(csv_f, fieldnames=list(row)).writerow(row); csv_f.flush()
            if writer:
                for k, v in m.items():
                    writer.add_scalar(f"val/{k}", v, ep)
            print(f"== epoch {ep} 完成（{fmt_time(ep_sec)}）" + "，".join(f"{k} {v:.4f}" for k, v in m.items())
                  + ("  ★ 目前最佳" if is_best else ""))
            remain = args.epochs - state["epoch"]
            if remain > 0:
                print(f"   照這個速度，剩下 {remain} 個 epoch 約需 {fmt_time(ep_sec * remain)}")
    except KeyboardInterrupt:
        print("\n手動中斷，存檔中 ...")
        save_ckpt(last_path, model, opt, sched, state, args)
        print(f"已存到 {last_path}，之後用 --resume 接續")
    finally:
        csv_f.close()
        if writer:
            writer.close()
    return model, first_losses, last_losses


def show_examples(model, data, dev, n=3):
    rng = np.random.default_rng(7)
    ids = data.val_ids[:n]
    bt = to_device(data.make_batch(ids, 1, rng, 0.0), dev)
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        prob = model.probs(bt["kinds"], bt["geom"], bt["construct"], bt["qa"], bt["qb"])
    for i in range(len(ids)):
        a, b = int(bt["qa"][i, 0]), int(bt["qb"][i, 0])
        ka, kb = int(bt["kinds"][i, a]), int(bt["kinds"][i, b])
        truth = torch.nonzero(bt["dbits"][i, 0]).flatten().tolist()
        print(f"\n驗證草圖 #{ids[i]}，圖元對 ({C.PRIM_NAMES_ZH[ka]}{a}, {C.PRIM_NAMES_ZH[kb]}{b})")
        print("  正確答案：" + "；".join(C.option_name_zh(o, ka, kb, a, b) for o in truth))
        p = prob[i, 0]
        top = torch.topk(p, 5)
        for r, (v, o) in enumerate(zip(top.values.tolist(), top.indices.tolist())):
            mark = "✓" if o in truth else " "
            print(f"  第{r + 1}名 {v:6.1%} {mark} {C.option_name_zh(o, ka, kb, a, b)}")


# =====================================================================
# 測速
# =====================================================================
def benchmark(args, data, dev):
    out = os.path.join(C.RUNS_DIR, args.name)
    os.makedirs(out, exist_ok=True)
    model, opt = build_model(args, data, dev)
    B, K = args.batch, args.pairs
    steps_per_epoch = math.ceil(len(data.train_ids) / B)
    sched = make_sched(opt, args.target_epochs * steps_per_epoch)
    rng = np.random.default_rng(0)
    perm = np.random.default_rng(0).permutation(data.train_ids)
    nb = max(1, len(perm) // B)
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)

    # 1. 只量 CPU 準備資料的時間
    t = time.time()
    for i in range(30):
        data.make_batch(perm[(i % nb) * B:((i % nb) + 1) * B], K, rng, C.COND_PROB)
    prep_sec = (time.time() - t) / 30

    def one_step(bt):
        loss = loss_fn(model, bt, data.soft, args.amp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
        opt.step(); sched.step()
        return loss

    warm, blocks, per_block = 30, 6, max(1, args.bench_steps // 6)
    total = warm + blocks * per_block
    fn = lambda i: to_device(data.make_batch(perm[(i % nb) * B:((i % nb) + 1) * B], K, rng, C.COND_PROB), dev)
    print(f"測速中：先預熱 {warm} 步，再計時 {blocks * per_block} 步 ...")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    it = iter(Prefetch(fn, total))
    for _ in range(warm):
        one_step(next(it))
    sync()
    block_times = []
    losses = []
    for _ in range(blocks):
        t = time.time()
        for _ in range(per_block):
            loss = one_step(next(it))
        sync()
        block_times.append((time.time() - t) / per_block)
        losses.append(float(loss))
    step_mean = float(np.mean(block_times))

    print("測量驗證時間 ...")
    sync(); t = time.time()
    m = evaluate(model, data, dev, args.amp)
    sync(); val_sec = time.time() - t

    print("測量存檔時間 ...")
    p = os.path.join(out, "bench_tmp.pt")
    t = time.time()
    save_ckpt(p, model, opt, sched, {"epoch": 0}, args)
    save_sec = time.time() - t
    try:
        os.remove(p)
    except OSError:
        pass

    train_epoch = step_mean * steps_per_epoch
    saves_per_epoch = 1 + train_epoch / (C.SAVE_EVERY_MINUTES * 60)
    epoch_sec = train_epoch + val_sec + saves_per_epoch * save_sec
    fast = min(block_times) * steps_per_epoch + val_sec + saves_per_epoch * save_sec
    slow = max(block_times) * steps_per_epoch + val_sec + saves_per_epoch * save_sec
    E = args.target_epochs
    gpu = torch.cuda.get_device_properties(0) if dev.type == "cuda" else None
    report = {
        "說明": "由 train_constraint.py --benchmark 產生，請把這個檔案交回去推算訓練時間",
        "時間": time.strftime("%Y-%m-%d %H:%M:%S"),
        "環境": {
            "顯示卡": gpu.name if gpu else "無（CPU）",
            "顯示記憶體總量GB": round(gpu.total_memory / 2**30, 2) if gpu else None,
            "測速時顯示記憶體峰值GB": round(torch.cuda.max_memory_allocated() / 2**30, 3) if gpu else None,
            "PyTorch": torch.__version__, "Python": platform.python_version(),
            "作業系統": platform.platform(), "CPU核心數": os.cpu_count(),
        },
        "設定": {"每步草圖數": B, "每張圖元對數": K, "混合精度": args.amp, "推導約束": args.derived,
                 "模型維度": args.dim, "層數": args.layers, "注意力頭數": args.heads,
                 "模型參數(百萬)": round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
                 "驗證草圖數": len(data.val_ids)},
        "資料": {"訓練草圖數": int(len(data.train_ids)), "每個epoch步數": steps_per_epoch},
        "量測": {
            "預熱步數": warm, "計時步數": blocks * per_block,
            "每步秒數(各區段平均)": [round(x, 5) for x in block_times],
            "每步秒數(平均)": round(step_mean, 5),
            "其中CPU準備資料每批秒數(背景執行，大部分可被遮蓋)": round(prep_sec, 5),
            "驗證一次秒數": round(val_sec, 2), "存檔一次秒數": round(save_sec, 2),
            "測速期間loss": [round(x, 4) for x in losses],
        },
        "推算": {
            "一個epoch秒數": round(epoch_sec, 1),
            f"{E}個epoch秒數": round(epoch_sec * E, 0),
            f"{E}個epoch(易讀)": fmt_time(epoch_sec * E),
            f"{E}個epoch範圍(最快~最慢區段)": [fmt_time(fast * E), fmt_time(slow * E)],
            "注意": "以短時間量測推算；長時間訓練可能因顯示卡溫度、同時使用電腦而變慢，實際值以正式訓練的 epochs.csv 為準",
        },
        "測速後（只訓練了預熱+計時這幾百步）的驗證結果(參考)": {k: round(v, 4) for k, v in m.items()},
    }
    path = os.path.join(out, "benchmark_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n========== 測速結果 ==========")
    print(f"顯示卡：{report['環境']['顯示卡']}，顯示記憶體峰值 {report['環境']['測速時顯示記憶體峰值GB']} GB")
    print(f"每步 {step_mean * 1000:.1f} 毫秒（CPU 準備一批 {prep_sec * 1000:.1f} 毫秒），每個 epoch {steps_per_epoch:,} 步")
    print(f"驗證一次 {fmt_time(val_sec)}，存檔一次 {fmt_time(save_sec)}")
    print(f"→ 一個 epoch 約 {fmt_time(epoch_sec)}")
    print(f"→ {E} 個 epoch 約 {fmt_time(epoch_sec * E)}（範圍 {fmt_time(fast * E)} ~ {fmt_time(slow * E)}）")
    print(f"報告已存到 {path}")


# =====================================================================
def main():
    ap = argparse.ArgumentParser(description="訓練約束建議模型")
    ap.add_argument("--name", default=None, help="實驗名稱（預設：smoke / bench / main）")
    ap.add_argument("--smoke", action="store_true", help="冒煙測試")
    ap.add_argument("--benchmark", action="store_true", help="測速")
    ap.add_argument("--resume", action="store_true", help="從 last.pt 接續")
    ap.add_argument("--derived", choices=["off", "soft", "on"], default="off")
    ap.add_argument("--epochs", type=int, default=C.NUM_EPOCHS)
    ap.add_argument("--target-epochs", type=int, default=1000, help="測速時要推算幾個 epoch 的時間")
    ap.add_argument("--bench-steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=C.BATCH_SKETCHES)
    ap.add_argument("--pairs", type=int, default=C.PAIRS_PER_SKETCH)
    ap.add_argument("--lr", type=float, default=C.LEARNING_RATE)
    ap.add_argument("--dim", type=int, default=C.MODEL_DIM)
    ap.add_argument("--layers", type=int, default=C.MODEL_LAYERS)
    ap.add_argument("--heads", type=int, default=C.MODEL_HEADS)
    ap.add_argument("--val-sketches", type=int, default=C.VAL_SKETCHES)
    ap.add_argument("--train-limit", type=int, default=None, help="只用部分訓練草圖（測試用）")
    ap.add_argument("--no-amp", dest="amp", action="store_false", help="關閉混合精度")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="雜訊增強的上限（佔草圖外框邊長的百分比，例如 5 = 5%%）；0 = 不加。驗證也會用同樣的雜訊")
    ap.add_argument("--noise-clean", type=float, default=0.2, help="雜訊增強時，保持精準的草圖比例")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        print("警告：沒有偵測到 GPU，會用 CPU 跑（非常慢）")
        args.amp = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if args.amp and not torch.cuda.is_bf16_supported():
            print("這張顯示卡不支援 bfloat16，關閉混合精度")
            args.amp = False
    torch.manual_seed(args.seed)

    if args.smoke:
        args.name = args.name or "smoke"
        args.epochs = min(args.epochs, 2)
        args.train_limit = args.train_limit or 20000
        args.val_sketches = min(args.val_sketches, 2000)
    elif args.benchmark:
        args.name = args.name or "bench"
    else:
        args.name = args.name or "main"

    data = Data(args.derived, args.val_sketches, args.train_limit, args.seed)
    if args.noise > 0:
        data.noise_max = args.noise / 100 * 2       # 百分比 → 正規化座標（外框邊長 = 2）
        data.noise_clean = args.noise_clean
        print(f"雜訊增強：{args.noise_clean:.0%} 草圖保持精準，其餘雜訊 0.1%～{args.noise:g}%（佔外框邊長，對數均勻）")
    if args.benchmark:
        benchmark(args, data, dev)
        return
    model, first, last = train(args, data, dev)
    if args.smoke:
        print("\n========== 冒煙測試結果 ==========")
        if first and last:
            ok = np.mean(last) < np.mean(first)
            print(f"訓練 loss：一開始 {np.mean(first):.4f} → 最後 {np.mean(last):.4f}（{'有下降 ✓' if ok else '沒有下降，請回報'}）")
        show_examples(model, data, dev)
        print("\n冒煙測試完成。結果在", os.path.join(C.RUNS_DIR, args.name))


if __name__ == "__main__":
    main()
