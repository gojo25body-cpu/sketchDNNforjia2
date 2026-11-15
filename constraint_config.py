"""
約束建議功能的共用設定（階段 1～5 都會用到）
"""
import os

# ===== 路徑 =====
RAW_DIR = os.path.join("data", "raw")                 # SketchGraphs 原始檔（和原本 dataset.py 共用，已下載過就不會重抓）
OUT_DIR = os.path.join("data", "constraint")          # 階段 1 的輸出
DATA_FILE = os.path.join(OUT_DIR, "constraint_data.npz")
STATS_FILE = os.path.join(OUT_DIR, "stats.json")
RAW_FILE_NAMES = ["sg_t16_train.npy", "sg_t16_validation.npy", "sg_t16_test.npy"]
DOWNLOAD_URL = "https://sketchgraphs.cs.princeton.edu/sequence/{}"

# ===== 資料篩選 =====
MAX_PRIMS = 16            # 一張草圖最多幾個圖元（超過就整張略過）——只影響「訓練資料」
# 產生建議時（階段 6 大草圖測試的結論：17～64 個圖元整張直接餵，前 5 名仍有 99.3～99.5%；
# 裁成 16 個的成績和整張幾乎相同，所以更大的圖改用裁切，計算量固定）
SUGGEST_DIRECT_MAX = 64   # 圖元數 ≤ 這個值：整張圖直接給模型
SUGGEST_CROP_K = 16       # 超過時：只取選定的圖元 + 離它們最近的圖元，共這麼多個
SUGGEST_HARD_MAX = 5000   # 一次最多讀幾個圖元（避免不小心全選整張大圖時卡很久）

# ===== 批次自動找約束 =====
BATCH_MAX_PRIMS = 1500     # 批次功能一次最多處理幾個圖元
BATCH_NEIGHBORS = 6        # 每個圖元和離它最近的幾個圖元配成候選對（另外「碰在一起」的一律納入）
BATCH_TOUCH = 0.002        # 兩個圖元的關鍵點距離 < 圖形大小 × 這個比例，視為碰在一起
BATCH_PROB_MIN = 0.5       # 機率低於這個值的約束不列出
BATCH_AUTO_PROB = 0.9      # 機率 ≥ 這個值、而且幾何上幾乎成立 → 標為「高把握」，預設勾選
BATCH_GEOM_DIST = 0.005    # 幾乎成立：距離誤差 ≤ 圖形大小 × 0.5%
BATCH_GEOM_ANGLE = 1.0     # 幾乎成立：角度誤差 ≤ 1 度
BATCH_MAX_PER_PAIR = 2     # 同一對圖元最多列幾個約束（不同類型）
MIN_PRIMS = 2             # 至少幾個圖元（少於 2 個就沒有「圖元對」可言）
SKIP_UNSUPPORTED = True   # 草圖裡有雲形線/橢圓等不支援的圖元時，整張略過（否則模型會看到殘缺的圖）
DEDUP_LEVELS = 256        # 去除重複草圖時，座標量化的精細程度（[-1,1] 切成幾格）
SPLIT_SEED = 4            # 訓練/驗證/測試切分的亂數種子（和原本 config.DATASET_SPLIT_SEED 相同）
SPLIT_RATIO = (0.90, 0.05, 0.05)

# ===== 圖元類型（本專案自己的編號）=====
PRIM_NONE, PRIM_POINT, PRIM_LINE, PRIM_CIRCLE, PRIM_ARC = -1, 0, 1, 2, 3
PRIM_NAMES_ZH = {PRIM_POINT: "點", PRIM_LINE: "線", PRIM_CIRCLE: "圓", PRIM_ARC: "弧"}

# 每個圖元存 7 個幾何數字（已正規化到 [-1, 1]）
#   0,1 起點 x,y   2,3 終點 x,y   4,5 中心 x,y   6 半徑
#   點：起點=終點=中心=該點        線：中心=中點，半徑=0
#   圓：只有中心與半徑有意義（起終點填 0）  弧：全部有意義
GEOM_DIM = 7

# ===== 約束的「位置」編號 =====
SUB_WHOLE, SUB_START, SUB_END, SUB_CENTER = 0, 1, 2, 3
SUB_NAMES_ZH = {SUB_WHOLE: "整個", SUB_START: "起點", SUB_END: "終點", SUB_CENTER: "中心"}

# ===== 約束類型 =====
# 資料檔裡存的是 SketchGraphs 原始的 ConstraintType 編號（不先篩），
# 這樣之後要增減類型（例如要不要加「同心」）不必重新處理資料。
CONSTRAINT_NAMES_ZH = {
    0: "重合", 1: "投影", 2: "鏡射", 3: "距離", 4: "水平", 5: "平行", 6: "垂直",
    7: "相切", 8: "長度", 9: "垂直相交", 10: "中點", 11: "等長/等徑", 12: "直徑",
    13: "偏移", 14: "半徑", 15: "同心", 16: "固定", 17: "角度", 18: "環形陣列",
    19: "穿刺", 20: "線性陣列", 21: "中心線尺寸", 22: "相交", 23: "輪廓", 24: "象限點",
    25: "法向", 26: "短軸直徑", 27: "長軸直徑", 28: "Rho", 29: "未知",
}
# 目前需求確認的 8 種幾何約束（階段 2 會用到；尺寸類約束不在範圍內）
TARGET_CONSTRAINTS = [0, 4, 6, 5, 9, 7, 10, 11]

# =====================================================================
# 階段 2：約束「選項」的編號
#   一個選項 = (約束類型, A 的位置, B 的位置)，A 是編號較小的圖元
#   選項編號 = 類型序號 * 16 + A位置 * 4 + B位置，共 8 * 4 * 4 = 128 種
# =====================================================================
NUM_TYPES = len(TARGET_CONSTRAINTS)
NUM_OPTIONS = NUM_TYPES * 16
RAW_TO_TIDX = {raw: i for i, raw in enumerate(TARGET_CONSTRAINTS)}
TIDX_NAMES_ZH = [CONSTRAINT_NAMES_ZH[raw] for raw in TARGET_CONSTRAINTS]
T_COINCIDENT, T_HORIZONTAL, T_VERTICAL, T_PARALLEL, T_PERPENDICULAR, T_TANGENT, T_MIDPOINT, T_EQUAL = range(8)


def option_id(tidx, sa, sb):
    return tidx * 16 + sa * 4 + sb


def option_split(opt):
    return opt // 16, (opt // 4) % 4, opt % 4


def option_name_zh(opt, kind_a=None, kind_b=None, ia=None, ib=None):
    t, sa, sb = option_split(int(opt))
    na = (PRIM_NAMES_ZH.get(kind_a, "") + (str(ia) if ia is not None else "A"))
    nb = (PRIM_NAMES_ZH.get(kind_b, "") + (str(ib) if ib is not None else "B"))
    return f"{TIDX_NAMES_ZH[t]}：{na} 的{SUB_NAMES_ZH[sa]} 與 {nb} 的{SUB_NAMES_ZH[sb]}"


LABELS_FILE = os.path.join(OUT_DIR, "labels.npz")
LABELS_STATS_FILE = os.path.join(OUT_DIR, "labels_stats.json")
# 「合法選項」：某種圖元組合下，訓練資料中出現至少這麼多次的選項才會被模型考慮
LEGAL_MIN_COUNT = 50
LEGAL_MIN_FRAC = 1e-4

# =====================================================================
# 階段 2：模型與訓練（依 RTX 3050 6GB 設計，可用參數覆蓋）
# =====================================================================
MODEL_DIM = 256
MODEL_LAYERS = 6
MODEL_HEADS = 8
MODEL_DROPOUT = 0.1

BATCH_SKETCHES = 512        # 一步訓練幾張草圖
PAIRS_PER_SKETCH = 8        # 每張草圖抽幾個圖元對（一個 epoch = 每張訓練草圖各被抽到一次）
COND_PROB = 0.5             # 有多少比例的圖元對，會附上「第 1 對已選的約束」當條件（訓練 2 對圖元的情況）
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 1000
GRAD_CLIP = 1.0
NUM_EPOCHS = 1000
VAL_SKETCHES = 20000        # 每個 epoch 結束時用幾張驗證草圖評估（全部 13 萬張太慢）
SAVE_EVERY_MINUTES = 20     # 每隔幾分鐘存一次進度（中途當機可以接續）
RUNS_DIR = "runs_constraint"
