"""
AutoCAD 介面（D 部分）：在 AutoCAD 選圖元 → 模型給 5 個約束建議 → 選一個套用到圖面上

需要：完整版 AutoCAD（不是 LT）、已開啟一個圖檔；Python 套件 pywin32（pip install pywin32）

用法（在專案資料夾、sketchdnn 環境下，AutoCAD 要先開著）：
  python autocad_interface.py                 開啟操作視窗（主要用法）
  python autocad_interface.py --diag          連線診斷：印出 AutoCAD 版本、目前圖檔、物件數
  python autocad_interface.py --read-test     步驟 1 測試：在 AutoCAD 選圖元，印出讀到的內容
  python autocad_interface.py --test-apply    步驟 3 測試：在「新的空白圖檔」自動畫測試圖形，逐一試套用 8 種約束，
                                              產生 runs_constraint/autocad_test_report.json
可選參數：--ckpt 模型檔（預設 runs_constraint/main/best.pt，找不到就用 smoke）、--gpu 用顯示卡推論

運作方式：
  * 用 Windows 的 COM 自動化連到「正在執行」的 AutoCAD
  * 套用約束時，送出 AutoCAD 的幾何約束指令（GCCOINCIDENT、GCPARALLEL…），用 AutoLISP 指定物件與點選位置
  * 套用後重新讀取座標，確認約束真的成立（「幾何驗證」）；三種點選方式會依序嘗試（見 _sel 說明）
"""
import os
import sys
import json
import math
import time
import queue
import threading
import argparse

import constraint_config as C

# ===================== 純幾何與指令產生（不需要 AutoCAD，可單獨測試） =====================
POS_KEYS = ("whole", "start", "end", "center")


def is_pointlike(kind, pos):
    """這個位置是不是一個「點」"""
    return {"point": pos == "whole", "line": pos in ("start", "end"), "circle": pos == "center",
            "arc": pos in ("start", "end", "center")}[kind]


def _arc_sweep(g):
    return (g["end_angle"] - g["start_angle"]) % (2 * math.pi) or 2 * math.pi


def keypoint(g, pos):
    """圖元 g 在某個位置的座標（用來驗證）"""
    k = g["type"]
    if k == "point":
        return tuple(g["position"])
    if k == "line":
        s, e = g["start"], g["end"]
        if pos == "start":
            return tuple(s)
        if pos == "end":
            return tuple(e)
        return ((s[0] + e[0]) / 2, (s[1] + e[1]) / 2)
    c, r = g["center"], g["radius"]
    if k == "circle" or pos == "center":
        return tuple(c)
    a = g["start_angle"] if pos == "start" else (g["end_angle"] if pos == "end" else g["start_angle"] + _arc_sweep(g) / 2)
    return (c[0] + r * math.cos(a), c[1] + r * math.sin(a))


def pick_point(g, pos):
    """要「點」在圖元的哪裡，AutoCAD 才會選到我們要的位置
    線：起點附近（10%）、終點附近（90%）、整條或中點（50%）；圓：圓周上 45 度；弧：同線的比例；點：該點"""
    k = g["type"]
    if k == "point":
        return tuple(g["position"])
    if k == "line":
        s, e = g["start"], g["end"]
        t = {"start": 0.1, "end": 0.9}.get(pos, 0.5)
        return (s[0] + t * (e[0] - s[0]), s[1] + t * (e[1] - s[1]))
    c, r = g["center"], g["radius"]
    if k == "circle":
        return (c[0] + r * math.cos(math.pi / 4), c[1] + r * math.sin(math.pi / 4))
    t = {"start": 0.1, "end": 0.9}.get(pos, 0.5)
    a = g["start_angle"] + t * _arc_sweep(g)
    return (c[0] + r * math.cos(a), c[1] + r * math.sin(a))


def label_point(g, off):
    """編號標籤的位置：圖元中間稍微往外偏一點，不壓在線上"""
    p = pick_point(g, "whole")
    if g["type"] == "line":
        dx, dy = g["end"][0] - g["start"][0], g["end"][1] - g["start"][1]
        n = math.hypot(dx, dy) or 1.0
        return (p[0] - dy / n * off, p[1] + dx / n * off)
    if g["type"] in ("circle", "arc"):
        c = g["center"]
        dx, dy = p[0] - c[0], p[1] - c[1]
        n = math.hypot(dx, dy) or 1.0
        return (p[0] + dx / n * off, p[1] + dy / n * off)
    return (p[0] + off, p[1] + off)


def _dist_point_line(p, s, e):
    dx, dy = e[0] - s[0], e[1] - s[1]
    n = math.hypot(dx, dy) or 1e-12
    return abs((p[0] - s[0]) * dy - (p[1] - s[1]) * dx) / n


def _dist_point_curve(p, g):
    if g["type"] == "line":
        return _dist_point_line(p, g["start"], g["end"])
    if g["type"] == "point":
        return math.dist(p, g["position"])
    return abs(math.dist(p, g["center"]) - g["radius"])


def _unit(g):
    s, e = g["start"], g["end"]
    dx, dy = e[0] - s[0], e[1] - s[1]
    n = math.hypot(dx, dy) or 1e-12
    return dx / n, dy / n


def residual(ga, pa, gb, pb, ctype):
    """約束成立時應為 0；無法計算時回傳 None"""
    ka, kb = ga["type"], gb["type"]
    pla, plb = is_pointlike(ka, pa), is_pointlike(kb, pb)
    if ctype == "coincident":
        if ka == kb == "line" and pa == pb == "whole":      # 兩條線「重合」= 共線
            u, v = _unit(ga), _unit(gb)
            return max(abs(u[0] * v[1] - u[1] * v[0]), _dist_point_line(gb["start"], ga["start"], ga["end"]))
        if ka in ("circle", "arc") and kb in ("circle", "arc") and pa == pb == "whole":   # 同一個圓
            return max(math.dist(ga["center"], gb["center"]), abs(ga["radius"] - gb["radius"]))
        if pla and plb:
            return math.dist(keypoint(ga, pa), keypoint(gb, pb))
        if pla and pb == "whole":
            return _dist_point_curve(keypoint(ga, pa), gb)
        if plb and pa == "whole":
            return _dist_point_curve(keypoint(gb, pb), ga)
    elif ctype == "concentric" and ka in ("circle", "arc") and kb in ("circle", "arc"):
        return math.dist(ga["center"], gb["center"])
    elif ctype in ("horizontal", "vertical") and pla and plb:
        p, q = keypoint(ga, pa), keypoint(gb, pb)
        return abs(p[1] - q[1]) if ctype == "horizontal" else abs(p[0] - q[0])
    elif ctype in ("parallel", "perpendicular") and ka == kb == "line":
        u, v = _unit(ga), _unit(gb)
        return abs(u[0] * v[1] - u[1] * v[0]) if ctype == "parallel" else abs(u[0] * v[0] + u[1] * v[1])
    elif ctype == "tangent":
        if ka == "line" and kb in ("circle", "arc"):
            return abs(_dist_point_line(gb["center"], ga["start"], ga["end"]) - gb["radius"])
        if kb == "line" and ka in ("circle", "arc"):
            return abs(_dist_point_line(ga["center"], gb["start"], gb["end"]) - ga["radius"])
        if ka in ("circle", "arc") and kb in ("circle", "arc"):
            d = math.dist(ga["center"], gb["center"])
            return min(abs(d - (ga["radius"] + gb["radius"])), abs(d - abs(ga["radius"] - gb["radius"])))
    elif ctype == "equal":
        if ka == kb == "line":
            return abs(math.dist(ga["start"], ga["end"]) - math.dist(gb["start"], gb["end"]))
        if ka in ("circle", "arc") and kb in ("circle", "arc"):
            return abs(ga["radius"] - gb["radius"])
    elif ctype == "midpoint":
        if pla and kb in ("line", "arc"):
            return math.dist(keypoint(ga, pa), keypoint(gb, "mid"))
        if plb and ka in ("line", "arc"):
            return math.dist(keypoint(gb, pb), keypoint(ga, "mid"))
    return None


def _lisp_pt(p):
    return f"'({p[0]:.10g} {p[1]:.10g} 0.0)"


STRATEGIES = ("near", "osnap", "exact", "entsel")
STRATEGY_ZH = {"near": "點在位置附近", "osnap": "物件鎖點", "exact": "點在位置正上方", "entsel": "物件+點"}
_OSNAP = {"start": "_end", "end": "_end", "center": "_cen", "mid": "_mid"}


def _sel(handle, g, pos, strategy):
    """怎麼告訴 AutoCAD「點這裡」：
    near：只給座標，點在目標位置附近的圖元上（例如線的 10% 處代表起點），由 AutoCAD 找最近的約束點
    osnap：先下物件鎖點（_end / _cen / _mid）再點在圖元上；用在弧的圓心這類「不在圖元上」的位置
    exact：只給座標，直接點在目標位置上（端點、圓心、中點）；物件提示時同 near
    entsel：(物件 座標) 清單；第一次實測 GCCOINCIDENT 不接受這種格式，所以排最後"""
    if strategy == "exact" and (pos != "whole" or g["type"] == "point"):
        return _lisp_pt(keypoint(g, pos))
    if strategy == "osnap" and pos in _OSNAP and g["type"] != "point":
        # 先下「物件鎖點」（端點 / 圓心 / 中點），再點在圖元上 → AutoCAD 會鎖到該位置
        # 弧的圓心不在弧上，「點在附近」會被當成弧的中點（第一次實測的結果），所以需要這種方式
        on = pick_point(g, "whole" if pos == "center" else pos)
        return f'"{_OSNAP[pos]}" {_lisp_pt(on)}'

    p = pick_point(g, pos)
    return f'(list (handent "{handle}") {_lisp_pt(p)})' if strategy == "entsel" else _lisp_pt(p)


def build_command(ha, ga, pa, hb, gb, pb, ctype, strategy):
    """產生 AutoLISP 指令字串；這個約束 AutoCAD 沒有對應指令時回傳 None"""
    ka, kb = ga["type"], gb["type"]
    pla, plb = is_pointlike(ka, pa), is_pointlike(kb, pb)
    A = lambda pos: _sel(ha, ga, pos, strategy)
    B = lambda pos: _sel(hb, gb, pos, strategy)
    if ctype == "coincident":
        if ka == kb == "line" and pa == pb == "whole":      # 兩條線「重合」= 共線（Onshape 的定義）
            args = ['"_.GCCOLLINEAR"', A("whole"), B("whole")]
        elif pla and plb:
            args = ['"_.GCCOINCIDENT"', A(pa), B(pb)]
        elif pla and pb == "whole" and kb != "point":
            args = ['"_.GCCOINCIDENT"', '"_O"', B("whole"), A(pa)]
        elif plb and pa == "whole" and ka != "point":
            args = ['"_.GCCOINCIDENT"', '"_O"', A("whole"), B(pb)]
        else:
            return None
    elif ctype == "midpoint":       # 點選線/弧的正中間，AutoCAD 會抓到它的中點
        if pla and kb in ("line", "arc") and pb == "whole":
            args = ['"_.GCCOINCIDENT"', A(pa), B("mid")]
        elif plb and ka in ("line", "arc") and pa == "whole":
            args = ['"_.GCCOINCIDENT"', B(pb), A("mid")]
        else:
            return None
    elif ctype in ("horizontal", "vertical"):
        if not (pla and plb):
            return None
        args = [f'"_.GC{ctype.upper()}"', '"_2P"', A(pa), B(pb)]
    elif ctype == "concentric":
        if not (ka in ("circle", "arc") and kb in ("circle", "arc")):
            return None
        args = ['"_.GCCONCENTRIC"', A("whole"), B("whole")]
    elif ctype in ("parallel", "perpendicular", "tangent", "equal"):
        if pa != "whole" or pb != "whole":
            return None
        args = [f'"_.GC{ctype.upper()}"', A("whole"), B("whole")]
    else:
        return None
    return "(command " + " ".join(args) + ")"


def expand_steps(c, ga, gb):
    """有些約束在 AutoCAD 要拆成好幾步：兩個圓/弧「重合」（同一個圓）= 同心 + 等徑"""
    if c["type"] == "coincident" and c["first"]["position"] == c["second"]["position"] == "whole" \
            and ga["type"] in ("circle", "arc") and gb["type"] in ("circle", "arc"):
        return ["concentric", "equal"]
    return [c["type"]]


def manual_hint(c, names):
    """自動套用失敗時，告訴使用者怎麼手動做"""
    cmd = {"coincident": "GCCOINCIDENT（兩條線整條重合請用 GCCOLLINEAR）", "midpoint": "GCCOINCIDENT（點選線的中點）", "horizontal": "GCHORIZONTAL（選 2點）",
           "vertical": "GCVERTICAL（選 2點）", "parallel": "GCPARALLEL", "perpendicular": "GCPERPENDICULAR",
           "tangent": "GCTANGENT", "equal": "GCEQUAL"}[c["type"]]
    i, j = c["pair"]
    return f"請在 AutoCAD 輸入 {cmd}，依序點 {names[i]} 的{c['first']['position_zh']}、{names[j]} 的{c['second']['position_zh']}"


# ===================== AutoCAD 連線（需要 Windows + pywin32） =====================
class Acad:
    BUSY = (-2147418111, -2147417846)    # AutoCAD 忙碌中，稍後重試

    def __init__(self):
        try:
            import win32com.client
            import pythoncom
            import pywintypes
        except ImportError:
            raise SystemExit("找不到 pywin32，請先執行：pip install pywin32")
        self.win32com, self.pythoncom, self.pywintypes = win32com, pythoncom, pywintypes
        try:
            self.app = win32com.client.GetActiveObject("AutoCAD.Application")
        except Exception:
            raise SystemExit("連不到 AutoCAD：請先開啟 AutoCAD 和一個圖檔（必須是完整版，LT 不支援）")
        self.labels = []
        self.refresh_doc()

    def refresh_doc(self):
        """取得目前作用中的圖檔；AutoCAD 停在「開始」頁籤、或正在忙時給清楚的提示"""
        try:
            n = self.r(lambda: self.app.Documents.Count)
        except Exception as e:
            raise SystemExit(f"AutoCAD 沒有回應（{e}）。\n最常見的原因：AutoCAD 裡還有指令在進行（例如還在畫線、或在等你點選）。\n"
                             "請切到 AutoCAD 按 2～3 次 Esc，確認下方指令列是空白的「鍵入指令」，再重新執行。")
        if n == 0:
            raise SystemExit("AutoCAD 目前沒有開啟任何圖檔（可能停在「開始」頁籤）。請先開一個圖檔（例如輸入 QNEW）再執行")
        self.doc = self.r(lambda: self.app.ActiveDocument)
        try:
            self.r(lambda: self.doc.Name)
            self.r(lambda: self.doc.ModelSpace)
        except Exception as e:
            raise SystemExit(f"連上 AutoCAD 了，但讀不到目前圖檔的內容（{e}）。\n"
                             "可能原因：AutoCAD 裡還有指令在進行或開著對話框（請按 2～3 次 Esc），或停在「開始」頁籤（請切到圖檔頁籤）。")

    def r(self, fn, tries=200):
        """呼叫 AutoCAD；它忙碌時自動重試。
        注意：AutoCAD 忙碌時，pywin32 有時會把錯誤包裝成 AttributeError（例如「<unknown>.ModelSpace」），
        所以 AttributeError 也重試幾秒，還是失敗才真的報錯"""
        attr_tries = 0
        warned = False
        for k in range(tries):
            try:
                return fn()
            except self.pywintypes.com_error as e:
                if e.args and e.args[0] in self.BUSY:
                    time.sleep(0.1)
                    continue
                raise
            except AttributeError:
                attr_tries += 1
                if attr_tries > 100:          # 最多等約 10 秒
                    raise
                if attr_tries == 10 and not warned:
                    print("AutoCAD 忙碌中，等待回應…（如果 AutoCAD 裡還有指令在進行，請切過去按 Esc）")
                    warned = True
                time.sleep(0.1)
        return fn()

    def V(self, p):
        return self.win32com.client.VARIANT(self.pythoncom.VT_ARRAY | self.pythoncom.VT_R8,
                                             (float(p[0]), float(p[1]), 0.0))

    def front(self):
        try:
            import win32gui
            self.r(lambda: setattr(self.app, "Visible", True))
            win32gui.SetForegroundWindow(self.r(lambda: self.app.HWND))
        except Exception:
            pass

    def prompt(self, text):
        try:
            self.r(lambda: self.doc.Utility.Prompt(text + "\n"))
        except Exception:
            pass

    _token = 100

    def send(self, text, timeout=8.0):
        """送出 AutoLISP，等它真的執行完。
        SendCommand 是非同步的（AutoCAD 有空才執行），所以在指令後面加一個「記號」：
        執行完會把使用者變數 USERI5 設成這次的編號，程式等到看見這個編號、且沒有指令在進行，才算完成。
        逾時（通常是指令卡在等輸入）就按兩次 Esc 取消，回傳 False"""
        Acad._token = Acad._token % 30000 + 1
        tok = Acad._token
        # 重要：如果 AutoCAD 不接受我們給的點，指令會停在那裡等使用者點選，而 SendCommand 會一直不回來（程式卡住）。
        # 所以在同一段 AutoLISP 裡，指令後面接「還有指令在等輸入就取消」：(command) 不帶參數 = 按 Esc
        guard = '(repeat 4 (if (> (getvar "CMDACTIVE") 0) (command)))'
        self.r(lambda: self.doc.SendCommand(f'(progn {text} {guard} (setvar "USERI5" {tok}) (princ))\n'))
        t0 = time.time()
        while time.time() - t0 < timeout:
            time.sleep(0.1)
            try:
                if self.r(lambda: self.doc.GetVariable("USERI5")) == tok and \
                        self.r(lambda: self.doc.GetVariable("CMDACTIVE")) == 0:
                    return True
            except Exception:
                pass
        try:
            self.r(lambda: self.doc.SendCommand("\x03\x03"))
        except Exception:
            # AutoCAD 停在某個指令的提示（例如「選取弧或圓」）時，連「取消」都送不進去，只能請使用者手動按 Esc
            print("AutoCAD 停在指令提示等待輸入，程式無法自動取消：請切到 AutoCAD 按 2～3 次 Esc")
        time.sleep(0.5)
        return False

    # ---------- 在 AutoCAD 直接點選圖元 ----------
    # entsel：點空白處（ERRNO=7）就再點一次；按 Esc 會變成錯誤，用 vl-catch-all-apply 接住，避免整段 LISP 中斷
    _PICK_LISP = ('(vl-load-com)'
                  '(defun aiPick (/ r) (setvar "ERRNO" 0) (setq r (vl-catch-all-apply (quote entsel) nil))'
                  ' (while (and (null r) (= (getvar "ERRNO") 7)) (setvar "ERRNO" 0)'
                  ' (setq r (vl-catch-all-apply (quote entsel) nil)))'
                  ' (if (or (null r) (vl-catch-all-error-p r)) nil (cdr (assoc 5 (entget (car r))))))'
                  '(setvar "USERS1" "") (setvar "USERS2" "")'
                  '(if (setq aiH (aiPick)) (progn (setvar "USERS1" aiH)'
                  ' (if (setq aiH (aiPick)) (setvar "USERS2" aiH))))')

    def pick_one(self, timeout=300.0):
        """讓使用者在 AutoCAD 點一個圖元，回傳代碼（取消時 None）"""
        self.refresh_doc()
        self.front()
        self.prompt("\n[尺寸] 請點選一個圖元（點在線條上；按 Esc 取消）")
        lisp = self._PICK_LISP.split('(setvar "USERS1" "")')[0] + \
            '(setvar "USERS1" "") (if (setq aiH (aiPick)) (setvar "USERS1" aiH))'
        self.send(lisp, timeout=timeout)
        return self.r(lambda: self.doc.GetVariable("USERS1")) or None

    def pick_two(self, timeout=300.0):
        """讓使用者在 AutoCAD 依序點兩個圖元。回傳 (代碼1, 代碼2)，沒點到或按 Esc 的為 None"""
        self.refresh_doc()
        self.front()
        self.prompt("\n[約束建議] 請依序點選第 1 個、第 2 個圖元（點在線條上；按 Esc 取消）")
        self.send(self._PICK_LISP, timeout=timeout)
        h1 = self.r(lambda: self.doc.GetVariable("USERS1")) or None
        h2 = self.r(lambda: self.doc.GetVariable("USERS2")) or None
        return h1, h2

    # ---------- 套用記錄（存在圖檔裡，下次產生建議時排除已經加過的約束）----------
    # 存在圖檔的「具名字典」AI_CONSTRAINT_LOG 裡：每個約束一筆空的 XRecord，約束內容寫在名稱上
    # 例：2E3_start_30A_whole_coincident（圖元代碼_位置_圖元代碼_位置_類型）。存檔後下次開圖還在。
    LOG_DICT = "AI_CONSTRAINT_LOG"

    def _log_dict(self, create=False):
        try:
            return self.r(lambda: self.doc.Dictionaries.Item(self.LOG_DICT))
        except Exception:
            if not create:
                return None
            return self.r(lambda: self.doc.Dictionaries.Add(self.LOG_DICT))

    def load_records(self):
        """回傳已記錄的約束 set{(代碼, 位置, 代碼, 位置, 類型)}"""
        d = self._log_dict()
        out = set()
        if d is None:
            return out
        for k in range(self.r(lambda: d.Count)):
            obj = self.r(lambda: d.Item(k))
            parts = str(self.r(lambda: d.GetName(obj))).split("_")
            if len(parts) == 5:
                out.add(tuple(parts))
        return out

    @staticmethod
    def record_name(key):
        return "_".join(key)

    def add_record(self, key):
        d = self._log_dict(create=True)
        try:
            self.r(lambda: d.AddXRecord(self.record_name(key)))
        except Exception:
            pass        # 已經有同名的記錄

    def clear_records(self, handles=None):
        """刪除記錄；handles 給定時只刪和這些圖元有關的。回傳刪了幾筆"""
        d = self._log_dict()
        if d is None:
            return 0
        names = []
        for k in range(self.r(lambda: d.Count)):
            obj = self.r(lambda: d.Item(k))
            names.append(str(self.r(lambda: d.GetName(obj))))
        n = 0
        for name in names:
            parts = name.split("_")
            if handles is None or (len(parts) == 5 and (parts[0] in handles or parts[2] in handles)):
                try:
                    self.r(lambda: d.Remove(name)); n += 1
                except Exception:
                    pass
        return n

    # ---------- 讀取 ----------
    def read(self, ent):
        """回傳 (圖元 dict, 不支援時的原因)"""
        name = self.r(lambda: ent.ObjectName)
        h = self.r(lambda: ent.Handle)
        P = lambda v: (float(v[0]), float(v[1]))
        if name == "AcDbLine":
            g = {"type": "line", "start": P(self.r(lambda: ent.StartPoint)), "end": P(self.r(lambda: ent.EndPoint))}
        elif name == "AcDbCircle":
            g = {"type": "circle", "center": P(self.r(lambda: ent.Center)), "radius": float(self.r(lambda: ent.Radius))}
        elif name == "AcDbArc":
            g = {"type": "arc", "center": P(self.r(lambda: ent.Center)), "radius": float(self.r(lambda: ent.Radius)),
                 "start_angle": float(self.r(lambda: ent.StartAngle)), "end_angle": float(self.r(lambda: ent.EndAngle))}
        elif name == "AcDbPoint":
            g = {"type": "point", "position": P(self.r(lambda: ent.Coordinates))}
        else:
            hint = "（聚合線請先用 EXPLODE 分解）" if "Polyline" in name else ""
            return None, f"{name} 不支援{hint}"
        # 長度或半徑為 0 的圖元（重複點兩下產生的殘留線等）沒有方向/大小，模型無法使用，訓練資料也會丟掉它們
        if name == "AcDbLine" and math.dist(g["start"], g["end"]) < 1e-9:
            return None, f"長度為 0 的線（代碼 {h}，可用 OVERKILL 指令清除）"
        if name in ("AcDbCircle", "AcDbArc") and g["radius"] < 1e-9:
            return None, f"半徑為 0 的{'圓' if name == 'AcDbCircle' else '弧'}（代碼 {h}）"
        g["handle"] = h
        return g, None

    def reread(self, g):
        ent = self.r(lambda: self.doc.HandleToObject(g["handle"]))
        ng, _ = self.read(ent)
        return ng

    def select(self):
        """讓使用者在 AutoCAD 框選圖元，回傳 (支援的圖元清單, 略過的說明清單)"""
        self.refresh_doc()          # 使用者可能切換過圖檔
        self.clear_labels()         # 舊的編號文字不要被一起框選進來
        name = "AI_CONSTRAINT_SEL"
        try:
            self.r(lambda: self.doc.SelectionSets.Item(name)).Delete()
        except Exception:
            pass
        ss = self.r(lambda: self.doc.SelectionSets.Add(name))
        self.front()
        self.prompt("\n[約束建議] 請選取要分析的圖元（線、圓、弧、點；可以選很多個），選完按 Enter")
        self.r(lambda: ss.SelectOnScreen())
        prims, skipped = [], []
        for i in range(self.r(lambda: ss.Count)):
            g, why = self.read(self.r(lambda: ss.Item(i)))
            (prims.append(g) if g else skipped.append(why))
        try:
            ss.Delete()
        except Exception:
            pass
        return prims, skipped

    # ---------- 編號標籤 ----------
    def add_labels(self, prims):
        self.clear_labels()
        xs, ys = [], []
        for g in prims:
            for pos in ("start", "end", "center"):
                try:
                    p = keypoint(g, pos); xs.append(p[0]); ys.append(p[1])
                except KeyError:
                    pass
        size = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
        try:
            self.r(lambda: self.doc.Layers.Add("AI_LABELS")).Color = 1
        except Exception:
            pass
        for i, g in enumerate(prims):
            p = label_point(g, size * 0.03)
            t = self.r(lambda: self.doc.ModelSpace.AddText(str(i), self.V(p), size * 0.04))
            try:
                t.Layer = "AI_LABELS"; t.Color = 1
            except Exception:
                pass
            self.labels.append(t)
        self.r(lambda: self.doc.Regen(1))

    def clear_labels(self):
        for t in self.labels:
            try:
                self.r(lambda: t.Delete())
            except Exception:
                pass
        self.labels = []

    def zoom_to(self, gs, margin=0.3):
        xs, ys = [], []
        for g in gs:
            for pos in ("start", "end", "center", "whole"):
                try:
                    p = keypoint(g, pos); xs.append(p[0]); ys.append(p[1])
                except KeyError:
                    pass
            if g["type"] in ("circle", "arc"):
                c, r = g["center"], g["radius"]
                xs += [c[0] - r, c[0] + r]; ys += [c[1] - r, c[1] + r]
        w = max(max(xs) - min(xs), max(ys) - min(ys), 1e-3)
        lo = (min(xs) - margin * w, min(ys) - margin * w)
        hi = (max(xs) + margin * w, max(ys) + margin * w)
        self.send(f'(command "_.ZOOM" "_W" {_lisp_pt(lo)} {_lisp_pt(hi)})')

    # ---------- 套用 ----------
    def ensure_undo(self, log=print):
        """套用時用 UNDO 的「標記/返回」來還原失敗的嘗試，這需要 AutoCAD 的復原功能是「全部」模式。
        如果圖檔的復原被關掉（UNDOCTL 第 1 位元為 0）或只能復原一步（第 2 位元），
        UNDO 的提示會變成「[全部(A)/無(N)/一個(O)/組合(C)/圖層(L)]」，送 "_M" 會卡在那裡等輸入。
        所以先檢查，必要時切回「全部」模式。回傳 True = 可以使用"""
        v = int(self.r(lambda: self.doc.GetVariable("UNDOCTL")))
        if (v & 1) and not (v & 2):
            return True
        log(f"  注意：這個圖檔的復原功能被設定成「{'關閉' if not (v & 1) else '只能復原一步'}」（UNDOCTL={v}），"
            "套用需要它，已自動改回「全部」（等同在 AutoCAD 執行 UNDO → 全部）")
        if not (v & 1):
            self.send('(command "_.UNDO" "_All")')
        else:
            self.send('(command "_.UNDO" "_Control" "_All")')
        v = int(self.r(lambda: self.doc.GetVariable("UNDOCTL")))
        if (v & 1) and not (v & 2):
            return True
        log(f"  無法開啟復原功能（UNDOCTL={v}）。請在 AutoCAD 輸入 UNDO，選「全部(A)」，再重新套用")
        return False

    def apply_single(self, c, prims, log=print):
        """套用「單一條線」的水平／垂直（GCHORIZONTAL / GCVERTICAL 選物件）。回傳 (是否成功, 說明)"""
        if not self.ensure_undo(log):
            return False, f"{c['text_zh']} → 未套用（復原功能無法使用）"
        self.clear_labels()
        i = c["index"]
        horiz = c["type"] == "horizontal_line"
        name = "_.GCHORIZONTAL" if horiz else "_.GCVERTICAL"
        self.zoom_to([self.reread(prims[i])])
        for strategy in ("near", "entsel"):
            g = self.reread(prims[i])
            self.send('(command "_.UNDO" "_M")')
            fin = self.send(f'(command "{name}" {_sel(g["handle"], g, "whole", strategy)})')
            g = self.reread(prims[i]); prims[i].update(g)
            u = _unit(g)
            res = abs(u[1]) if horiz else abs(u[0])
            if fin and res <= 1e-6:
                return True, f"{c['text_zh']} → 成功（方式：{STRATEGY_ZH[strategy]}）"
            log(f"  方式「{STRATEGY_ZH[strategy]}」未成功（殘差 {res:.2g}），還原後換下一種")
            self.send('(command "_.UNDO" "_B")')
        return False, f"{c['text_zh']} → 所有點選方式都失敗"

    def apply(self, constraints, prims, log=print):
        """constraints：suggest() 回傳的某個建議的 constraints 清單。回傳 (是否全部成功, 說明)"""
        results = []
        # 重要：編號標籤（文字）放在圖元上，而程式點選的位置剛好在同一處，AutoCAD 會選到文字而不是圖元。
        # 所以套用前先移除標籤，套用完再由呼叫端重新標上
        if not self.ensure_undo(log):
            return False, [(False, f"{c['text_zh']} → 未套用（復原功能無法使用）") for c in constraints]
        self.clear_labels()
        self.zoom_to([prims[i] for c in constraints for i in c["pair"]])
        for c in constraints:
            i, j = c["pair"]
            pa, pb = c["first"]["position"], c["second"]["position"]
            msgs, all_ok = [], True
            # 套用前先檢查：如果幾何上「本來就成立」，套用後的驗證沒辦法分辨約束有沒有真的建立
            # （AutoCAD 若已有相同約束，可能會拒絕重複的約束），所以另外標註提醒
            g0a, g0b = self.reread(prims[i]), self.reread(prims[j])
            pre = [residual(g0a, pa, g0b, pb, st) for st in expand_steps(c, g0a, g0b)]
            already = all(r is not None and r <= 1e-6 * max(_size(g0a), _size(g0b), 1e-9) for r in pre)
            if already:
                log("  注意：這兩個圖元在套用前幾何上就已經符合這個約束（可能之前已經加過），"
                    "驗證只能確認幾何成立，無法確認約束是否新建立；請看 AutoCAD 圖面上的約束圖示")
            for step in expand_steps(c, prims[i], prims[j]):
                ok, msg = False, ""
                for strategy in STRATEGIES:
                    ga, gb = self.reread(prims[i]), self.reread(prims[j])     # 每一步前都重讀（上一步可能移動了圖元）
                    cmd = build_command(ga["handle"], ga, pa, gb["handle"], gb, pb, step, strategy)
                    if cmd is None:
                        msg = "AutoCAD 沒有對應的指令"
                        break
                    self.send('(command "_.UNDO" "_M")')
                    log(f"  送出（{step}，方式：{STRATEGY_ZH[strategy]}）：{cmd}")
                    finished = self.send(cmd)
                    ga, gb = self.reread(prims[i]), self.reread(prims[j])
                    prims[i].update(ga); prims[j].update(gb)
                    res = residual(ga, pa, gb, pb, step)
                    size = max(_size(ga), _size(gb), 1e-9)
                    if finished and res is not None and res <= 1e-6 * size:
                        ok, msg = True, f"成功（方式：{STRATEGY_ZH[strategy]}，殘差 {res:.1e}）"
                        break
                    log(f"  方式「{STRATEGY_ZH[strategy]}」未成功（{'指令已結束' if finished else '逾時'}，殘差 {res}），還原後換下一種")
                    self.send('(command "_.UNDO" "_B")')
                msgs.append((f"[{step}] " if len(expand_steps(c, prims[i], prims[j])) > 1 else "") + (msg or "所有點選方式都失敗"))
                all_ok &= ok
                if not ok:
                    break
            results.append((all_ok, f"{c['text_zh']} → " + "；".join(msgs) + ("（套用前已成立）" if already else "")))
        return all(r[0] for r in results), results


def _size(g):
    if g["type"] == "line":
        return math.dist(g["start"], g["end"])
    if g["type"] in ("circle", "arc"):
        return 2 * g["radius"]
    return 1.0


def describe(i, g):
    f = lambda p: f"({p[0]:.4g}, {p[1]:.4g})"
    k = g["type"]
    if k == "line":
        return f"{i}：線 {f(g['start'])} → {f(g['end'])}"
    if k == "circle":
        return f"{i}：圓 圓心{f(g['center'])} 半徑 {g['radius']:.4g}"
    if k == "arc":
        return (f"{i}：弧 圓心{f(g['center'])} 半徑 {g['radius']:.4g} "
                f"角度 {math.degrees(g['start_angle']):.1f}°→{math.degrees(g['end_angle']):.1f}°")
    return f"{i}：點 {f(g['position'])}"


def handle_key(c, prims):
    """suggest() 回傳的一個約束 → 用圖元代碼表示的記錄識別碼"""
    from constraint_suggest import constraint_key
    i, j = c["pair"]
    return constraint_key(prims[i]["handle"], c["first"]["position"],
                          prims[j]["handle"], c["second"]["position"], c["type"])


def index_records(records, prims):
    """記錄（圖元代碼）→ 這次選取範圍裡的編號；不在範圍裡的忽略"""
    from constraint_suggest import constraint_key
    idx = {g["handle"]: i for i, g in enumerate(prims)}
    out = set()
    for h1, p1, h2, p2, t in records:
        if h1 in idx and h2 in idx:
            out.add(constraint_key(idx[h1], p1, idx[h2], p2, t))
    return out


def to_model_input(prims):
    return [{k: v for k, v in g.items() if k != "handle"} for g in prims]


def default_ckpt():
    for p in ("runs_constraint/noise5/best.pt", "runs_constraint/main/best.pt", "runs_constraint/smoke/best.pt"):
        if os.path.exists(p):
            return p
    return "runs_constraint/main/best.pt"


# ===================== 測試模式 =====================
def diag():
    acad = Acad()
    a, d = acad.app, acad.doc
    print("AutoCAD 名稱：", acad.r(lambda: a.Name), "版本：", acad.r(lambda: a.Version))
    print("開啟的圖檔數：", acad.r(lambda: a.Documents.Count), "目前圖檔：", acad.r(lambda: d.Name))
    print("模型空間物件數：", acad.r(lambda: d.ModelSpace.Count))
    ok = acad.send('(princ)')
    print("送出 AutoLISP 測試：", "成功" if ok else "失敗（逾時）")


def record_test():
    """檢查「套用記錄」能不能寫進圖檔、讀回來、刪掉（不會動到任何圖元）"""
    acad = Acad()
    key = ("FFFFF1", "start", "FFFFF2", "whole", "coincident")
    print("1. 寫入一筆測試記錄 ...")
    acad.add_record(key)
    got = acad.load_records()
    print("   讀回：", "成功" if key in got else f"失敗（讀到 {got}）")
    print("2. 刪除測試記錄 ...")
    n = acad.clear_records({"FFFFF1"})
    print("   刪除：", "成功" if n >= 1 and key not in acad.load_records() else "失敗")
    print("記錄功能測試完成" if key not in acad.load_records() else "注意：測試記錄沒有刪乾淨")


def read_test():
    acad = Acad()
    print("請切到 AutoCAD 選取圖元，選完按 Enter ...")
    prims, skipped = acad.select()
    print(f"\n讀到 {len(prims)} 個支援的圖元，略過 {len(skipped)} 個")
    for s in skipped:
        print("  略過：", s)
    for i, g in enumerate(prims):
        print("  " + describe(i, g))
    if prims:
        acad.add_labels(prims)
        print("\n已在 AutoCAD 圖面上標出編號（紅字），請確認編號和上面列的對得上。按 Enter 清除編號並結束 ...")
        input()
        acad.clear_labels()
    print(json.dumps({"primitives": to_model_input(prims)}, ensure_ascii=False, indent=1)[:2000])


TEST_CASES = [
    # (說明, [圖元], 約束類型, A 位置, B 位置)
    ("線終點 重合 線起點", [("line", (0, 0), (10, 0)), ("line", (12, 2), (20, 6))], "coincident", "end", "start"),
    ("線起點 重合 圓心", [("line", (0, 0), (10, 3)), ("circle", (15, 8), 3)], "coincident", "start", "center"),
    ("線終點 在圓上", [("line", (0, 0), (10, 1)), ("circle", (16, 4), 3)], "coincident", "end", "whole"),
    ("點 在線上", [("point", (5, 4)), ("line", (0, 0), (12, 1))], "coincident", "whole", "whole"),
    ("兩點水平", [("line", (0, 0), (8, 3)), ("line", (12, 5), (20, 9))], "horizontal", "start", "end"),
    ("兩點垂直", [("line", (0, 0), (8, 3)), ("line", (12, 5), (20, 9))], "vertical", "end", "start"),
    ("平行", [("line", (0, 0), (10, 1)), ("line", (0, 5), (10, 9))], "parallel", "whole", "whole"),
    ("兩線共線", [("line", (0, 0), (10, 1)), ("line", (12, 4), (20, 6))], "coincident", "whole", "whole"),
    ("垂直相交", [("line", (0, 0), (10, 1)), ("line", (4, 3), (12, 9))], "perpendicular", "whole", "whole"),
    ("線與圓相切", [("line", (0, 0), (20, 1)), ("circle", (10, 6), 3)], "tangent", "whole", "whole"),
    ("等長（線）", [("line", (0, 0), (10, 0)), ("line", (0, 5), (6, 8))], "equal", "whole", "whole"),
    ("等徑（圓）", [("circle", (0, 0), 3), ("circle", (12, 0), 5)], "equal", "whole", "whole"),
    ("點在線中點", [("point", (3, 4)), ("line", (0, 0), (12, 2))], "midpoint", "whole", "whole"),
    ("線端點在弧中點", [("line", (-9, 9), (-3, 2)), ("arc", (0, 0), 5, 0.2, 2.0)], "midpoint", "end", "whole"),
    ("兩弧同圓", [("arc", (0, 0), 5, 0.2, 2.0), ("arc", (3, 2), 4, 3.0, 5.0)], "coincident", "whole", "whole"),
    ("弧圓心 重合 點", [("arc", (0, 0), 5, 0.2, 2.0), ("point", (9, 7))], "coincident", "center", "whole"),
    ("弧起點 重合 線終點", [("arc", (0, 0), 5, 0.3, 2.5), ("line", (8, 8), (14, 12))], "coincident", "start", "end"),
    ("弧圓心 重合 線起點", [("arc", (0, 0), 5, 0.4, 2.2), ("line", (9, 6), (15, 10))], "coincident", "center", "start"),
    ("弧圓心 與 點 水平", [("arc", (0, 0), 5, 0.4, 2.2), ("point", (10, 6))], "horizontal", "center", "whole"),
]


def test_apply(keep=False, only=None):
    acad = Acad()
    print(f"已連上 AutoCAD，目前圖檔：{acad.r(lambda: acad.doc.Name)}")
    n_ent = acad.r(lambda: acad.doc.ModelSpace.Count)
    if n_ent > 0:
        print(f"注意：目前圖檔裡已經有 {n_ent} 個物件。建議在「新的空白圖檔」執行測試（QNEW 開新圖），避免影響你的圖。")
        if input("仍要繼續嗎？輸入 y 繼續：").strip().lower() != "y":
            return
    try:
        acad.r(lambda: acad.doc.Layers.Add("AI_TEST"))
        acad.r(lambda: setattr(acad.doc, "ActiveLayer", acad.doc.Layers.Item("AI_TEST")))
    except Exception:
        pass
    ms = acad.r(lambda: acad.doc.ModelSpace)
    report = {"時間": time.strftime("%Y-%m-%d %H:%M:%S"), "AutoCAD": str(acad.r(lambda: acad.app.Version)), "結果": []}
    created = []
    for k, (title, shapes, ctype, pa, pb) in enumerate(TEST_CASES):
        if only and only not in title:
            continue
        ox = 100.0 * k
        ents = []
        for s in shapes:
            if s[0] == "line":
                e = acad.r(lambda: ms.AddLine(acad.V((s[1][0] + ox, s[1][1])), acad.V((s[2][0] + ox, s[2][1]))))
            elif s[0] == "circle":
                e = acad.r(lambda: ms.AddCircle(acad.V((s[1][0] + ox, s[1][1])), s[2]))
            elif s[0] == "arc":
                e = acad.r(lambda: ms.AddArc(acad.V((s[1][0] + ox, s[1][1])), s[2], s[3], s[4]))
            else:
                e = acad.r(lambda: ms.AddPoint(acad.V((s[1][0] + ox, s[1][1]))))
            ents.append(e); created.append(e)
        prims = [acad.read(e)[0] for e in ents]
        before = residual(prims[0], pa, prims[1], pb, ctype)
        c = {"pair": [0, 1], "type": ctype, "first": {"position": pa, "position_zh": pa},
             "second": {"position": pb, "position_zh": pb}, "text_zh": title}
        logs = []
        ok, res = acad.apply([c], prims, log=logs.append)
        line = {"測試": title, "類型": ctype, "套用前殘差": before, "成功": ok, "說明": res[0][1], "紀錄": logs}
        report["結果"].append(line)
        print(f"{'✓' if ok else '✗'} {title}：{res[0][1]}")
    n_ok = sum(r["成功"] for r in report["結果"])
    report["總結"] = f"{n_ok}/{len(report['結果'])} 成功"
    os.makedirs(C.RUNS_DIR, exist_ok=True)
    path = os.path.join(C.RUNS_DIR, "autocad_test_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n{report['總結']}，報告已存到 {path}")
    if not keep:
        input("按 Enter 刪除測試圖形（想留著看就先去 AutoCAD 看，再回來按 Enter）...")
        for e in created:
            try:
                acad.r(lambda: e.Delete())
            except Exception:
                pass


# ===================== 操作視窗 =====================
def gui(ckpt, use_gpu):
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
    from constraint_suggest import Suggester

    acad = Acad()
    root = tk.Tk()
    root.title("約束建議（AutoCAD）")
    root.attributes("-topmost", True)
    F = ("Microsoft JhengHei", 11)
    state = {"prims": [], "sg": None, "result": None}

    def log(msg):
        out.insert("end", msg + "\n"); out.see("end"); root.update()

    def load_model():
        try:
            state["sg"] = Suggester(ckpt_var.get(), None if use_gpu else "cpu")
            log(f"模型已載入：{ckpt_var.get()}（訓練了 {state['sg'].epoch} 個 epoch）")
        except Exception as e:
            log(f"模型載入失敗：{e}")

    def do_select():
        if busy():
            return
        log("請切到 AutoCAD 選取圖元，選完按 Enter ...")
        try:
            prims, skipped = acad.select()
        except Exception as e:
            log(f"選取失敗：{e}"); return
        for s in skipped:
            log("  略過：" + s)
        if len(prims) > C.SUGGEST_HARD_MAX:
            log(f"選了 {len(prims)} 個，超過 {C.SUGGEST_HARD_MAX} 個上限，請少選一些"); return
        if len(prims) > C.SUGGEST_DIRECT_MAX:
            log(f"圖元超過 {C.SUGGEST_DIRECT_MAX} 個：產生建議時只會用選定圖元附近的 {C.SUGGEST_CROP_K} 個"
                f"（大圖測試中，這樣的準確率和整張圖幾乎相同）")
        if len(prims) < 2:
            log("至少要選 2 個支援的圖元"); return
        state["prims"] = prims
        lst.delete(0, "end")
        for i, g in enumerate(prims):
            lst.insert("end", describe(i, g))
        nums = [str(i) for i in range(len(prims))]
        for cb in (a1, b1, a2, b2):
            cb["values"] = nums
        a1.set("0"); b1.set("1"); a2.set("1"); b2.set(str(min(2, len(prims) - 1)))
        acad.add_labels(prims)
        log(f"讀到 {len(prims)} 個圖元，已在 AutoCAD 標上紅色編號")
        root.lift()

    def busy():
        if state.get("picking"):
            log(state.get("busy_msg") or "正在等你在 AutoCAD 點選圖元：請先點完兩個圖元，或在 AutoCAD 按 Esc 取消")
            return True
        return False

    def do_pick(which):
        """在 AutoCAD 點兩個圖元，自動填入第 which 對的編號。
        等待點選的工作放在背景執行緒（自己連一條 AutoCAD 連線），介面視窗在等待時仍可操作、可關閉"""
        if busy():
            return
        if not state["prims"]:
            log("請先按 ① 選取圖元"); return
        log(f"請切到 AutoCAD，依序點第 {which} 對的兩個圖元（在 AutoCAD 按 Esc 取消）...")
        q = queue.Queue()

        def worker():
            import pythoncom
            pythoncom.CoInitialize()
            try:
                q.put(("ok", Acad().pick_two()))
            except BaseException as e:          # 連線失敗時 Acad() 會丟 SystemExit
                q.put(("err", e))
            finally:
                pythoncom.CoUninitialize()

        state["picking"] = True
        threading.Thread(target=worker, daemon=True).start()

        def poll():
            try:
                kind, val = q.get_nowait()
            except queue.Empty:
                root.after(150, poll); return
            state["picking"] = False
            root.lift()
            if kind == "err":
                log(f"點選失敗：{val}"); return
            finish_pick(which, *val)
        poll()

    def finish_pick(which, h1, h2):
        idx = {g["handle"]: i for i, g in enumerate(state["prims"])}
        if h1 is None or h2 is None:
            log("已取消（或只點了一個）"); return
        if any(h not in idx for h in (h1, h2)):
            log("點到的物件不在 ① 選取的範圍裡（可能點到紅色編號文字或其他物件），請重點一次"); return
        i, j = idx[h1], idx[h2]
        if i == j:
            log("兩次點到同一個圖元，請重點一次"); return
        (a1 if which == 1 else a2).set(str(i))
        (b1 if which == 1 else b2).set(str(j))
        if which == 2:
            two.set(True)
        lst.selection_clear(0, "end")
        for k in (i, j):
            lst.selection_set(k)
        lst.see(i)
        log(f"第 {which} 對：{describe(i, state['prims'][i])}  與  {describe(j, state['prims'][j])}")

    def do_suggest():
        if busy():
            return
        if not state["prims"]:
            log("請先選取圖元"); return
        if state["sg"] is None:
            load_model()
            if state["sg"] is None:
                return
        pairs = [[int(a1.get()), int(b1.get())]]
        if two.get():
            pairs.append([int(a2.get()), int(b2.get())])
        try:
            for g in state["prims"]:
                g.update(acad.reread(g))
            try:
                excl = index_records(acad.load_records(), state["prims"])
            except Exception as e:
                excl = set()
                log(f"（讀不到套用記錄，這次不排除已加過的約束：{e}）")
            res = state["sg"].suggest(to_model_input(state["prims"]), pairs, exclude=excl)
        except Exception as e:
            log(f"產生建議失敗：{e}"); return
        state["result"] = res
        for w in sug_frame.winfo_children():
            w.destroy()
        choice.set(0)
        for s in res["suggestions"]:
            p = "補充" if s["probability"] is None else f"{s['probability']:.1%}"
            tag = "（補充）" if s["filler"] else ("（放寬）" if s["relaxed"] else "")
            text = f"第{s['rank']}名  {p}  " + "；".join(c["text_zh"] for c in s["constraints"]) + tag
            tk.Radiobutton(sug_frame, text=text, variable=choice, value=s["rank"], font=F, anchor="w",
                           justify="left", wraplength=620).pack(fill="x", anchor="w")
        if res.get("excluded"):
            log(f"（已排除 {res['excluded']} 個先前透過本介面加過的約束）")
        log(f"已產生 {len(res['suggestions'])} 個建議（圖元對 {pairs}；模型看的是{res['context']['mode_zh']}）")

    def do_apply():
        if busy():
            return
        res = state["result"]
        if not res or choice.get() == 0:
            log("請先產生建議並選一個"); return
        s = res["suggestions"][choice.get() - 1]
        if s["filler"] and not messagebox.askyesno("確認", "這是「補充」選項（資料中幾乎沒出現過），確定要套用嗎？"):
            return
        log(f"套用第 {s['rank']} 名 ...")
        try:
            recs = acad.load_records()
        except Exception:
            recs = set()
        todo = []
        for c in s["constraints"]:
            if handle_key(c, state["prims"]) in recs:
                log(f"  已經加過，略過：{c['text_zh']}")
            else:
                todo.append(c)
        if not todo:
            log("這個建議的約束都已經加過了"); return
        try:
            ok, details = acad.apply(todo, state["prims"], log=log)
        except Exception as e:
            log(f"套用時發生錯誤：{e}"); return
        names = {i: describe(i, g).split("：")[1].split(" ")[0] + str(i) for i, g in enumerate(state["prims"])}
        for (good, msg), c in zip(details, todo):
            log(("  ✓ " if good else "  ✗ ") + msg)
            if good:
                try:
                    acad.add_record(handle_key(c, state["prims"]))
                except Exception as e:
                    log(f"    （記錄失敗，下次可能會再建議這個約束：{e}）")
            else:
                log("    " + manual_hint(c, names))
        log("全部成功" if ok else "部分失敗（失敗的已還原）；可以在 AutoCAD 按 Ctrl+Z 復原成功的部分")
        acad.add_labels(state["prims"])

    def do_batch():
        """④ 掃過目前選取的所有圖元，列出值得加的約束，讓使用者勾選後一次套用"""
        if busy():
            return
        if not state["prims"]:
            log("請先按 ① 選取圖元（批次功能會掃過選取的全部圖元）"); return
        if len(state["prims"]) > C.BATCH_MAX_PRIMS:
            log(f"批次功能一次最多 {C.BATCH_MAX_PRIMS} 個圖元，目前 {len(state['prims'])} 個，請分區選取"); return
        if state["sg"] is None:
            load_model()
            if state["sg"] is None:
                return
        import constraint_batch as CB
        log("批次分析中 ...")
        root.update()
        try:
            for g in state["prims"]:
                g.update(acad.reread(g))
            prims = state["prims"]
            kinds, raw, _ = CB.raw_primitives(to_model_input(prims))
            pairs = CB.neighbor_pairs(kinds, raw)
            canon, probs, legal = state["sg"].score_pairs(to_model_input(prims), pairs)
            try:
                excl = index_records(acad.load_records(), prims)
            except Exception:
                excl = set()
            items, stats = CB.select_constraints(prims, canon, probs, legal, exclude=excl, rules=rule_var.get())
        except Exception as e:
            log(f"批次分析失敗：{e}"); return
        log("批次分析結果：" + "，".join(f"{k} {v}" for k, v in stats.items()))
        if not items:
            log("沒有找到需要加的約束"); return
        show_batch_window(items)

    def show_batch_window(items):
        win = tk.Toplevel(root)
        win.title(f"批次約束：共 {len(items)} 個（高把握 {sum(x['auto'] for x in items)} 個已預先勾選）")
        win.attributes("-topmost", True)
        win.geometry("900x620")
        tk.Label(win, font=F, justify="left", anchor="w", text=(
            "【高把握】= 模型機率 ≥ {:.0%} 且目前圖上幾乎已經成立（套用後圖形幾乎不會動）\n"
            "【請確認】= 機率夠高但圖上差比較多，套用後圖形可能會移動，請確認是你要的\n"
            "【規則】= 規則補強（不是模型）：接近水平/垂直的單一條線、已經相切的圓角接點".format(C.BATCH_AUTO_PROB))
                 ).pack(fill="x", padx=8, pady=4)
        box = tk.Frame(win); box.pack(fill="both", expand=True, padx=8)
        cv = tk.Canvas(box, highlightthickness=0)
        sb = tk.Scrollbar(box, orient="vertical", command=cv.yview)
        inner = tk.Frame(cv)
        inner.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.create_window((0, 0), window=inner, anchor="nw")
        cv.configure(yscrollcommand=sb.set)
        cv.pack(side="left", fill="both", expand=True); sb.pack(side="right", fill="y")
        cv.bind_all("<MouseWheel>", lambda e: cv.yview_scroll(int(-e.delta / 120), "units"))
        import constraint_batch as CB
        vars_ = []
        for x in items:
            v = tk.BooleanVar(value=x["auto"])
            color = "#1f4e9c" if x.get("rule") else ("#1a6b1a" if x["auto"] else "#8a5a00")
            tk.Checkbutton(inner, text=CB.describe_item(x), variable=v, font=F, anchor="w",
                           fg=color).pack(fill="x", anchor="w")
            vars_.append(v)
        bar = tk.Frame(win); bar.pack(fill="x", padx=8, pady=6)
        stop = {"flag": False}

        def set_all(mode):
            for v, x in zip(vars_, items):
                if mode == "rule":
                    if x.get("rule"):
                        v.set(not v.get())          # 規則項整批切換勾選
                    continue
                v.set(mode == "all" or (mode == "auto" and x["auto"]))

        def run():
            todo = [x for v, x in zip(vars_, items) if v.get()]
            if not todo:
                log("沒有勾選任何約束"); return
            if not messagebox.askyesno("確認", f"要依序套用勾選的 {len(todo)} 個約束嗎？\n"
                                       "（套用中可以按「停止」；每個約束都會驗證，失敗的會自動還原）", parent=win):
                return
            stop["flag"] = False
            state["picking"] = True          # 借用忙碌旗標，避免套用中又按其他按鈕
            state["busy_msg"] = "批次套用進行中，請等它完成，或在批次視窗按「停止」"
            n_ok = n_skip = n_fail = 0
            try:
                recs = acad.load_records()
            except Exception:
                recs = set()
            try:
                for k, x in enumerate(todo):
                    root.update()
                    if stop["flag"]:
                        log("已停止"); break
                    c = x["constraint"]
                    key = handle_key(c, state["prims"])
                    if key in recs:
                        n_skip += 1; continue
                    log(f"[{k + 1}/{len(todo)}] {c['text_zh']}")
                    try:
                        if c.get("single"):
                            ok1, msg1 = acad.apply_single(c, state["prims"], log=lambda m: None)
                            ok, det = ok1, [(ok1, msg1)]
                        else:
                            ok, det = acad.apply([c], state["prims"], log=lambda m: None)
                    except Exception as e:
                        ok, det = False, [(False, str(e))]
                    if ok:
                        n_ok += 1
                        recs.add(key)
                        try:
                            acad.add_record(key)
                        except Exception:
                            pass
                    else:
                        n_fail += 1
                        log("    ✗ " + det[0][1])
            finally:
                state["picking"] = False
                state["busy_msg"] = None
            log(f"批次套用完成：成功 {n_ok}、失敗 {n_fail}（已還原）、已加過而略過 {n_skip}")
            acad.add_labels(state["prims"])
            win.destroy()

        tk.Button(bar, text="只勾高把握", font=F, command=lambda: set_all("auto")).pack(side="left")
        tk.Button(bar, text="全選", font=F, command=lambda: set_all("all")).pack(side="left", padx=4)
        tk.Button(bar, text="全不選", font=F, command=lambda: set_all("none")).pack(side="left")
        if any(x.get("rule") for x in items):
            tk.Button(bar, text="規則項 勾選/取消", font=F, command=lambda: set_all("rule")).pack(side="left", padx=4)
        tk.Button(bar, text="停止", font=F, command=lambda: stop.update(flag=True)).pack(side="right")
        tk.Button(bar, text="套用勾選的約束", font=F, command=run).pack(side="right", padx=4)

    def show_dim_window():
        """⑤ 尺寸與縮放：加尺寸約束、改數值（可用運算式）、固定基準、等比例縮放"""
        if busy():
            return
        if not state["prims"]:
            log("請先按 ① 選取圖元"); return
        import dims as D
        doc_full = acad.r(lambda: acad.doc.FullName)
        doc_name = acad.r(lambda: acad.doc.Name)
        reg_path = D.registry_path(doc_full, doc_name)
        reg = D.load_registry(reg_path)
        w = tk.Toplevel(root)
        w.title("⑤ 尺寸與縮放")
        w.attributes("-topmost", True)
        w.geometry("860x640")

        def prims_extent():
            xs, ys = [], []
            for g in state["prims"]:
                for pos in ("start", "end", "center"):
                    try:
                        p = keypoint(g, pos); xs.append(p[0]); ys.append(p[1])
                    except KeyError:
                        pass
                if g["type"] in ("circle", "arc"):
                    c, r = g["center"], g["radius"]
                    xs += [c[0] - r, c[0] + r]; ys += [c[1] - r, c[1] + r]
            return (min(xs), min(ys)), (max(xs), max(ys))

        def ent_size(g):
            return math.dist(g["start"], g["end"]) if g["type"] == "line" else 2 * g.get("radius", 1.0)

        # ---- 選圖元 ----
        f1 = tk.LabelFrame(w, text="1. 選一個圖元", font=F); f1.pack(fill="x", padx=8, pady=4)
        ent_cb = ttk.Combobox(f1, width=6, state="readonly", values=[str(i) for i in range(len(state["prims"]))])
        ent_cb.grid(row=0, column=0, padx=4, pady=4)
        ent_lbl = tk.Label(f1, text="", font=F, anchor="w"); ent_lbl.grid(row=0, column=2, sticky="w")

        # ---- 加尺寸 ----
        f2 = tk.LabelFrame(w, text="2. 加尺寸約束（數值預設為目前大小，圖形不會動；改成別的數值圖形會跟著變）", font=F)
        f2.pack(fill="x", padx=8, pady=4)
        kind_cb = ttk.Combobox(f2, width=16, state="readonly"); kind_cb.grid(row=0, column=0, padx=4, pady=4)
        val_var = tk.StringVar()
        tk.Entry(f2, textvariable=val_var, width=24, font=F).grid(row=0, column=1, padx=4)
        kinds_now = []

        def cur_index():
            try:
                return int(ent_cb.get())
            except ValueError:
                return None

        def on_entity(*_):
            i = cur_index()
            if i is None:
                return
            g = state["prims"][i]
            ent_lbl.config(text=describe(i, g))
            fixpos_now[:] = D.fix_points_for(g)
            fixpos_cb["values"] = [zh for _, zh in fixpos_now]
            if fixpos_now:
                fixpos_cb.current(0)
            kinds_now[:] = D.kinds_for(g)
            kind_cb["values"] = [zh for _, zh in kinds_now]
            if kinds_now:
                kind_cb.current(0); on_kind()
            else:
                kind_cb.set(""); val_var.set("")
                log("這種圖元（點）不能加尺寸約束")

        def on_kind(*_):
            i = cur_index()
            if i is None or not kinds_now or kind_cb.current() < 0:
                return
            k = kinds_now[kind_cb.current()][0]
            val_var.set(D.fmt(D.measure(acad.reread(state["prims"][i]), k)))

        ent_cb.bind("<<ComboboxSelected>>", on_entity)
        kind_cb.bind("<<ComboboxSelected>>", on_kind)

        def pick_entity():
            if busy():
                return
            log("請切到 AutoCAD 點選一個圖元（Esc 取消）...")
            q = queue.Queue()

            def worker():                       # 背景等待點選，視窗不會卡住
                import pythoncom
                pythoncom.CoInitialize()
                try:
                    q.put(("ok", Acad().pick_one()))
                except BaseException as e:
                    q.put(("err", e))
                finally:
                    pythoncom.CoUninitialize()

            state["picking"] = True
            state["busy_msg"] = "正在等你在 AutoCAD 點選圖元：請先點選，或在 AutoCAD 按 Esc 取消"
            threading.Thread(target=worker, daemon=True).start()

            def poll():
                try:
                    kind, h = q.get_nowait()
                except queue.Empty:
                    w.after(150, poll); return
                state["picking"] = False
                state["busy_msg"] = None
                w.lift()
                if kind == "err":
                    log(f"點選失敗：{h}"); return
                idx = {g["handle"]: i for i, g in enumerate(state["prims"])}
                if h is None:
                    log("已取消"); return
                if h not in idx:
                    log("點到的物件不在 ① 選取的範圍裡（可能點到紅色編號文字），請重點"); return
                ent_cb.set(str(idx[h])); on_entity()
            poll()

        tk.Button(f1, text="在 AutoCAD 點選", font=F, command=pick_entity).grid(row=0, column=1, padx=4)

        def add_dim():
            if busy():
                return
            i = cur_index()
            if i is None or not kinds_now or kind_cb.current() < 0:
                log("請先選圖元和尺寸類型"); return
            k = kinds_now[kind_cb.current()][0]
            expr = val_var.get().strip()
            err = D.check_expr(expr)
            if err:
                messagebox.showwarning("數值有誤", err, parent=w); return
            if not acad.ensure_undo(log):
                return
            g = acad.reread(state["prims"][i]); state["prims"][i].update(g)
            name = D.next_name(reg, k)
            acad.clear_labels()
            acad.zoom_to([g], margin=1.5)
            cmd = D.dim_command(g, k, name, expr, ent_size(g))
            fin = acad.send(cmd)
            g2 = acad.reread(state["prims"][i]); state["prims"][i].update(g2)
            now = D.measure(g2, k)
            try:
                want = float(expr)
                ok = fin and abs(now - want) <= 1e-6 * max(1.0, want)
                note = f"目前 {D.fmt(now)}（預期 {D.fmt(want)}）"
            except ValueError:
                ok, note = fin, f"目前 {D.fmt(now)}（運算式，無法自動比對）"
            if ok:
                reg["dims"].append({"name": name, "kind": k, "handle": g2["handle"], "expr": expr})
                D.save_registry(reg_path, reg)
                log(f"已加尺寸 {name} = {expr}：{note}")
                if abs(D.measure(g, k) - now) < 1e-9 and fin:
                    log("  （數值等於原本大小，圖形不會動；之後改數值時就能看出尺寸有沒有生效）")
            else:
                log(f"加尺寸可能失敗（{'指令已結束' if fin else '逾時被取消'}；{note}）。送出的指令：{cmd}")
            refresh_all()

        tk.Button(f2, text="加尺寸", font=F, command=add_dim).grid(row=0, column=2, padx=4)

        fixpos_cb = ttk.Combobox(f2, width=8, state="readonly"); fixpos_cb.grid(row=1, column=0, padx=4, pady=4, sticky="e")
        fixpos_now = []

        def fix_point():
            if busy():
                return
            i = cur_index()
            if i is None or fixpos_cb.current() < 0:
                log("請先選圖元和要固定的點"); return
            if not acad.ensure_undo(log):
                return
            pos, pos_zh = fixpos_now[fixpos_cb.current()]
            g = acad.reread(state["prims"][i])
            before = keypoint(g, pos)
            acad.clear_labels(); acad.zoom_to([g], margin=1.5)
            fin = acad.send(D.fix_command(g, pos))
            if fin:
                reg.setdefault("fixed", []).append({"handle": g["handle"], "position": pos})
                D.save_registry(reg_path, reg)
                log(f"已固定圖元 {i} 的{pos_zh} {tuple(round(v, 4) for v in before)}：之後改尺寸時這個點不會移動"
                    "（圖元的大小、方向仍可改變）；請看 AutoCAD 圖面上的鎖頭圖示確認")
            else:
                log("固定指令沒有正常結束，請到 AutoCAD 按 Esc 後重試")
            refresh_all()

        tk.Button(f2, text="固定這個點（當基準）", font=F, command=fix_point).grid(row=1, column=1, padx=4, sticky="w")
        tk.Label(f2, text="← 只固定一個點：位置不動，但尺寸仍可修改", font=("Microsoft JhengHei", 9), fg="#666").grid(
            row=1, column=2, columnspan=2, sticky="w")

        # ---- 尺寸清單與改值 ----
        f3 = tk.LabelFrame(w, text="3. 已加的尺寸：選一列，輸入新數值或運算式（例如 dimD1*2）後按「修改」", font=F)
        f3.pack(fill="both", expand=True, padx=8, pady=4)
        cols = ("名字", "類型", "圖元", "設定值", "目前量測值")
        tv = ttk.Treeview(f3, columns=cols, show="headings", height=8)
        for c_, wd in zip(cols, (90, 150, 60, 200, 120)):
            tv.heading(c_, text=c_); tv.column(c_, width=wd, anchor="w")
        tv.pack(fill="both", expand=True, padx=4, pady=4)
        fr = tk.Frame(f3); fr.pack(fill="x")
        new_var = tk.StringVar()
        tk.Entry(fr, textvariable=new_var, width=30, font=F).pack(side="left", padx=4)

        def refresh_all():
            idx = {g["handle"]: i for i, g in enumerate(state["prims"])}
            tv.delete(*tv.get_children())
            for d in reg["dims"]:
                try:
                    g = acad.read(acad.r(lambda: acad.doc.HandleToObject(d["handle"])))[0]
                    now = D.fmt(D.measure(g, d["kind"]))
                except Exception:
                    now = "（圖元已刪除）"
                tv.insert("", "end", iid=d["name"], values=(d["name"], D.DIM_KINDS[d["kind"]][0],
                                                            idx.get(d["handle"], "—"), d["expr"], now))
            acad.add_labels(state["prims"])

        def on_row(*_):
            sel = tv.selection()
            if sel:
                d = next(x for x in reg["dims"] if x["name"] == sel[0])
                new_var.set(d["expr"])

        tv.bind("<<TreeviewSelect>>", on_row)

        def edit_dim():
            if busy():
                return
            sel = tv.selection()
            if not sel:
                log("請先在清單選一個尺寸"); return
            d = next(x for x in reg["dims"] if x["name"] == sel[0])
            expr = new_var.get().strip()
            err = D.check_expr(expr)
            if err:
                messagebox.showwarning("數值有誤", err, parent=w); return
            if not acad.ensure_undo(log):
                return
            acad.clear_labels()
            fin = acad.send(D.edit_command(d["name"], expr))
            try:
                g = acad.read(acad.r(lambda: acad.doc.HandleToObject(d["handle"])))[0]
                now = D.measure(g, d["kind"])
            except Exception:
                now = None
            for g_ in state["prims"]:              # 改尺寸可能讓很多圖元移動，全部重讀
                try:
                    g_.update(acad.reread(g_))
                except Exception:
                    pass
            try:
                want = float(expr)
                ok = fin and now is not None and abs(now - want) <= 1e-6 * max(1.0, want)
            except ValueError:
                want, ok = None, fin
            if ok:
                d["expr"] = expr
                D.save_registry(reg_path, reg)
                log(f"{d['name']} 改成 {expr}：目前量測值 {D.fmt(now) if now is not None else '—'}")
            else:
                log(f"{d['name']} 修改可能失敗（{'指令已結束' if fin else '逾時'}；目前量測值 "
                    f"{D.fmt(now) if now is not None else '—'}"
                    + (f"，預期 {D.fmt(want)}）" if want is not None else "）")
                    + "。若約束互相衝突（過度約束），AutoCAD 會拒絕修改")
            refresh_all()

        tk.Button(fr, text="修改", font=F, command=edit_dim).pack(side="left", padx=4)
        tk.Button(fr, text="重新整理", font=F, command=refresh_all).pack(side="left", padx=4)

        # ---- 等比例縮放 ----
        f4 = tk.LabelFrame(w, text="4. 等比例縮放（第一種做法）：① 選取的全部圖元，以外框左下角為基準", font=F)
        f4.pack(fill="x", padx=8, pady=4)
        tk.Label(f4, text="倍率：", font=F).pack(side="left", padx=4)
        fac_var = tk.StringVar(value="1.5")
        tk.Entry(f4, textvariable=fac_var, width=10, font=F).pack(side="left")

        def do_scale():
            if busy():
                return
            try:
                fac = float(fac_var.get())
                assert fac > 0
            except Exception:
                messagebox.showwarning("倍率有誤", "倍率要是大於 0 的數字", parent=w); return
            if reg["dims"] and not messagebox.askyesno(
                    "注意", "這張圖已經加了尺寸約束。縮放時 AutoCAD 會維持這些尺寸的數值，\n"
                            "所以形狀可能不會等比例變大／變小。\n要等比例縮放，建議改用「修改尺寸數值」，\n"
                            "或先刪除尺寸約束。仍要繼續嗎？", parent=w):
                return
            if not acad.ensure_undo(log):
                return
            for g_ in state["prims"]:
                g_.update(acad.reread(g_))
            (x0, y0), (x1, y1) = prims_extent()
            acad.clear_labels()
            acad.zoom_to(state["prims"], margin=max(0.3, fac - 1 + 0.3))
            fin = acad.send(D.scale_command([g["handle"] for g in state["prims"]], (x0, y0), fac))
            for g_ in state["prims"]:
                g_.update(acad.reread(g_))
            (a0, b0), (a1, b1) = prims_extent()
            ratio = max(a1 - a0, b1 - b0) / max(x1 - x0, y1 - y0)
            ok = fin and abs(ratio - fac) <= 1e-6 * fac
            log(f"等比例縮放 {D.fmt(fac)} 倍：外框實際變為 {ratio:.6g} 倍" + ("（成功）" if ok else "（與預期不同，請檢查）"))
            refresh_all()

        tk.Button(f4, text="等比例縮放", font=F, command=do_scale).pack(side="left", padx=8)
        tk.Label(w, text=f"尺寸清單存在：{reg_path}", font=("Microsoft JhengHei", 9), fg="#666").pack(anchor="w", padx=8)
        refresh_all()

    def do_clear():
        if busy():
            return
        if not state["prims"]:
            log("請先按 ① 選取圖元，再清除這些圖元的套用記錄"); return
        if not messagebox.askyesno("清除套用記錄",
                                   "清除後，之前透過本介面加過的約束會重新出現在建議裡。\n"
                                   "（只清除記錄，不會刪掉 AutoCAD 裡的約束）\n\n"
                                   "「是」= 只清除目前選取的圖元的記錄\n「否」= 不清除", parent=root):
            return
        try:
            n = acad.clear_records({g["handle"] for g in state["prims"]})
        except Exception as e:
            log(f"清除失敗：{e}"); return
        log(f"已清除 {n} 筆套用記錄")

    def on_close():
        if state.get("picking"):
            # AutoCAD 還在等點選，這時無法刪除編號文字；直接結束程式
            print("介面已關閉。AutoCAD 若還在等你點選，請按 Esc；紅色編號文字在 AI_LABELS 圖層，可以直接刪掉。")
            root.destroy()
            os._exit(0)
        acad.clear_labels()
        root.destroy()

    top = tk.Frame(root); top.pack(fill="x", padx=8, pady=4)
    tk.Label(top, text="模型：", font=F).pack(side="left")
    ckpt_var = tk.StringVar(value=ckpt)
    tk.Entry(top, textvariable=ckpt_var, width=45, font=F).pack(side="left")
    tk.Button(top, text="載入", font=F, command=load_model).pack(side="left", padx=4)
    tk.Button(top, text="清除套用記錄", font=F, command=do_clear).pack(side="left", padx=4)

    tk.Button(root, text="① 從 AutoCAD 選取圖元", font=F, command=do_select).pack(fill="x", padx=8, pady=4)
    lst = tk.Listbox(root, height=8, font=F, selectmode="multiple", exportselection=False); lst.pack(fill="x", padx=8)

    pf = tk.Frame(root); pf.pack(fill="x", padx=8, pady=4)
    tk.Label(pf, text="第 1 對：", font=F).grid(row=0, column=0)
    a1 = ttk.Combobox(pf, width=4, state="readonly"); a1.grid(row=0, column=1)
    tk.Label(pf, text="與", font=F).grid(row=0, column=2)
    b1 = ttk.Combobox(pf, width=4, state="readonly"); b1.grid(row=0, column=3)
    two = tk.BooleanVar(value=False)
    tk.Checkbutton(pf, text="加第 2 對：", variable=two, font=F).grid(row=0, column=4, padx=(16, 0))
    a2 = ttk.Combobox(pf, width=4, state="readonly"); a2.grid(row=0, column=5)
    tk.Label(pf, text="與", font=F).grid(row=0, column=6)
    b2 = ttk.Combobox(pf, width=4, state="readonly"); b2.grid(row=0, column=7)
    tk.Button(pf, text="在 AutoCAD 點選第 1 對", font=F, command=lambda: do_pick(1)).grid(
        row=1, column=0, columnspan=4, sticky="we", pady=(4, 0))
    tk.Button(pf, text="在 AutoCAD 點選第 2 對", font=F, command=lambda: do_pick(2)).grid(
        row=1, column=4, columnspan=4, sticky="we", pady=(4, 0), padx=(16, 0))

    tk.Button(root, text="② 產生 5 個建議", font=F, command=do_suggest).pack(fill="x", padx=8, pady=4)
    choice = tk.IntVar(value=0)
    sug_frame = tk.Frame(root); sug_frame.pack(fill="x", padx=8)
    tk.Button(root, text="③ 套用選中的建議到 AutoCAD", font=F, command=do_apply).pack(fill="x", padx=8, pady=4)
    tk.Button(root, text="④ 批次：自動找出選取範圍內值得加的約束", font=F, command=do_batch).pack(fill="x", padx=8, pady=(0, 4))
    tk.Button(root, text="⑤ 尺寸與縮放（加尺寸、改數值、固定基準、等比例縮放）", font=F,
              command=show_dim_window).pack(fill="x", padx=8, pady=(0, 4))
    rule_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="批次時加入規則補強（接近水平/垂直的單一條線、已相切的圓角接點）", variable=rule_var,
                   font=F, anchor="w").pack(fill="x", padx=8)
    out = scrolledtext.ScrolledText(root, height=12, font=("Microsoft JhengHei", 10)); out.pack(fill="both", expand=True, padx=8, pady=4)

    root.protocol("WM_DELETE_WINDOW", on_close)
    log(f"已連上 AutoCAD：{acad.r(lambda: acad.doc.Name)}")
    load_model()
    root.mainloop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AutoCAD 約束建議介面")
    ap.add_argument("--ckpt", default=default_ckpt())
    ap.add_argument("--gpu", action="store_true", help="用顯示卡推論（預設用 CPU，避免和訓練搶顯示卡）")
    ap.add_argument("--diag", action="store_true", help="連線診斷")
    ap.add_argument("--read-test", action="store_true")
    ap.add_argument("--record-test", action="store_true", help="測試套用記錄能否寫入圖檔")
    ap.add_argument("--test-apply", action="store_true")
    ap.add_argument("--keep", action="store_true", help="測試後保留測試圖形")
    ap.add_argument("--only", default=None, help="只跑名稱包含這段文字的測試，例如 --only 弧")
    args = ap.parse_args()
    if args.diag:
        diag()
    elif args.record_test:
        record_test()
    elif args.read_test:
        read_test()
    elif args.test_apply:
        test_apply(args.keep, args.only)
    else:
        gui(args.ckpt, args.gpu)
