"""
尺寸約束：加尺寸、改數值、固定、等比例縮放（第二種、第一種做法）

指令寫法來自 dim_test.py 的實測（10/10 成功）：
  建立：(command "_.DCDIAMETER" 圓上點 尺寸線位置 "名字=數值")，數值處輸入「名字=數值」可同時命名與設值
  改值：(command "_.-PARAMETERS" "_Edit" "名字" "數值或運算式")
DCRADIUS、DCVERTICAL、DCALIGNED、SCALE 已實測（dim_test 5～9）；GCFIX 改成「固定點」後需再測（dim_test 8）

尺寸清單（名字、類型、哪個圖元、設定的運算式）存在圖檔旁邊的 <圖檔名>.ai_dims.json；
圖檔還沒存過時存在 runs_constraint/dims_<圖檔名>.json。
「目前的值」一律直接從圖形量測，不依賴這份清單，所以就算在 AutoCAD 裡改過，顯示的值仍是正確的。
"""
import os
import re
import json
import math

import constraint_config as C

# 類型 → (中文, 適用的圖元, AutoCAD 指令, 名字前綴, 指令是否先按 Enter 選「物件」)
DIM_KINDS = {
    "diameter":   ("直徑", ("circle",), "_.DCDIAMETER", "dimD", False),
    "radius":     ("半徑", ("arc", "circle"), "_.DCRADIUS", "dimR", False),
    "length":     ("長度（斜線也可）", ("line",), "_.DCALIGNED", "dimL", True),
    "horizontal": ("水平距離", ("line",), "_.DCHORIZONTAL", "dimH", True),
    "vertical":   ("垂直距離", ("line",), "_.DCVERTICAL", "dimV", True),
}
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def kinds_for(g):
    """這個圖元可以加哪些尺寸：[(類型, 中文)]"""
    return [(k, v[0]) for k, v in DIM_KINDS.items() if g["type"] in v[1]]


def measure(g, kind):
    """從圖形量出目前的尺寸值"""
    if kind == "diameter":
        return 2 * g["radius"]
    if kind == "radius":
        return g["radius"]
    s, e = g["start"], g["end"]
    if kind == "length":
        return math.dist(s, e)
    if kind == "horizontal":
        return abs(e[0] - s[0])
    if kind == "vertical":
        return abs(e[1] - s[1])
    raise ValueError(kind)


def fmt(v):
    """數值 → AutoCAD 可接受的文字（保留足夠位數，不用科學記號）"""
    s = f"{v:.8f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def check_expr(expr):
    """運算式只允許數字、名字、+ - * / ( ) . 與空白；回傳錯誤訊息或 None"""
    expr = expr.strip()
    if not expr:
        return "請輸入數值或運算式"
    if not re.fullmatch(r"[A-Za-z0-9_+\-*/(). ]+", expr):
        return "只能使用數字、尺寸名字和 + - * / ( )"
    try:
        if float(expr) <= 0:
            return "尺寸必須大於 0"
    except ValueError:
        pass                                   # 是運算式，交給 AutoCAD 計算
    return None


def pick_and_place(g, kind, size):
    """(要點在圖元的哪裡, 尺寸線放在哪裡)；尺寸線放在圖元外側一點，避免點選時選到別的東西"""
    from autocad_interface import pick_point
    off = 0.15 * size
    if g["type"] == "line":
        s, e = g["start"], g["end"]
        m = ((s[0] + e[0]) / 2, (s[1] + e[1]) / 2)
        dx, dy = e[0] - s[0], e[1] - s[1]
        n = math.hypot(dx, dy) or 1.0
        if kind == "horizontal":
            loc = (m[0], m[1] - max(off, 0.15 * n))
        elif kind == "vertical":
            loc = (m[0] + max(off, 0.15 * n), m[1])
        else:
            loc = (m[0] - dy / n * max(off, 0.15 * n), m[1] + dx / n * max(off, 0.15 * n))
        return m, loc
    p = pick_point(g, "whole")
    c, r = g["center"], g["radius"]
    ux, uy = (p[0] - c[0]) / r, (p[1] - c[1]) / r
    return p, (c[0] + ux * r * 1.6, c[1] + uy * r * 1.6)


def dim_command(g, kind, name, expr, size):
    """建立尺寸約束的 AutoLISP"""
    from autocad_interface import _lisp_pt
    _, _, cmd, _, need_enter = DIM_KINDS[kind]
    p, loc = pick_and_place(g, kind, size)
    enter = '"" ' if need_enter else ""
    return f'(command "{cmd}" {enter}{_lisp_pt(p)} {_lisp_pt(loc)} "{name}={expr}")'


def edit_command(name, expr):
    return f'(command "_.-PARAMETERS" "_Edit" "{name}" "{expr}")'


# 固定「點」而不是整個圖元：固定整個圓會連直徑一起鎖住（dim_test 8c 實測：改直徑被拒絕）
FIX_POINTS = {"line": [("start", "起點"), ("end", "終點")], "circle": [("center", "圓心")],
              "arc": [("center", "圓心"), ("start", "起點"), ("end", "終點")], "point": [("whole", "這個點")]}


def fix_points_for(g):
    return FIX_POINTS.get(g["type"], [])


def fix_command(g, pos):
    """固定圖元上的一個點（GCFIX 選點）。圓心不在圓周上，用物件鎖點 _cen；端點點在靠近端點的線上"""
    from autocad_interface import _sel
    strategy = "osnap" if pos == "center" else "near"
    return f'(command "_.GCFIX" {_sel(g["handle"], g, pos, strategy)})'


def scale_command(handles, base, factor):
    """把指定的圖元以 base 為基準點等比例縮放"""
    from autocad_interface import _lisp_pt
    hs = " ".join(f'"{h}"' for h in handles)
    return (f'(setq aiSS (ssadd)) (foreach h (list {hs}) (if (handent h) (ssadd (handent h) aiSS))) '
            f'(command "_.SCALE" aiSS "" {_lisp_pt(base)} "{fmt(factor)}")')


# ---------- 尺寸清單 ----------
def registry_path(doc_fullname, doc_name):
    if doc_fullname:
        root, _ = os.path.splitext(doc_fullname)
        return root + ".ai_dims.json"
    os.makedirs(C.RUNS_DIR, exist_ok=True)
    safe = re.sub(r"[^\w.-]", "_", doc_name or "未命名")
    return os.path.join(C.RUNS_DIR, f"dims_{safe}.json")


def load_registry(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"dims": [], "fixed": []}


def save_registry(path, reg):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)


def next_name(reg, kind):
    prefix = DIM_KINDS[kind][3]
    used = {d["name"].lower() for d in reg["dims"]}
    k = 1
    while f"{prefix}{k}".lower() in used:
        k += 1
    return f"{prefix}{k}"
