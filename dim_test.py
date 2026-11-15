"""
尺寸約束可行性測試（第二種做法「改尺寸、約束不變」的前置測試）

要確認三件事：
  1. 程式能不能幫圖元加上「尺寸約束」（直徑、水平距離），並取我們自己定的名字
  2. 程式能不能改尺寸的數值，而且圖形真的跟著變
  3. 改尺寸時，已經加好的幾何約束（例如相切）會不會維持

用法（請在「新的空白圖檔」執行：AutoCAD 輸入 QNEW 開新圖）：
  python dim_test.py            測完會問要不要刪掉測試圖形
  python dim_test.py --keep     保留測試圖形
結果：畫面會印出每一步是否成功，並存成 runs_constraint/dim_test_report.json（請把這個檔案給 Claude）

說明：
  * 尺寸約束用 AutoCAD 的 DCDIAMETER、DCHORIZONTAL 指令建立；建立時在數值處輸入「名字=數值」（例如 AIDIA1=20），
    這樣就知道它叫什麼名字；改數值用 -PARAMETERS 指令的「編輯」。
  * 判斷成功與否完全看「幾何有沒有變成預期的樣子」（例如直徑設成 20，圓的半徑就該變成 10），不依賴 AutoCAD 的回報。
"""
import os
import json
import math
import time
import argparse

import constraint_config as C
from autocad_interface import Acad, residual, _lisp_pt

TOL = 1e-6


def main(keep):
    acad = Acad()
    print(f"已連上 AutoCAD，目前圖檔：{acad.r(lambda: acad.doc.Name)}")
    n_ent = acad.r(lambda: acad.doc.ModelSpace.Count)
    if n_ent > 0:
        print(f"注意：目前圖檔裡已經有 {n_ent} 個物件。請在新的空白圖檔測試（AutoCAD 輸入 QNEW）。")
        if input("仍要繼續嗎？輸入 y 繼續：").strip().lower() != "y":
            return
    ms = acad.r(lambda: acad.doc.ModelSpace)
    report = {"時間": time.strftime("%Y-%m-%d %H:%M:%S"), "AutoCAD": str(acad.r(lambda: acad.app.Version)),
              "CCONSTRAINTFORM": acad.r(lambda: acad.doc.GetVariable("CCONSTRAINTFORM")), "步驟": []}
    created = []
    logs = []

    def step(name, ok, detail):
        report["步驟"].append({"步驟": name, "成功": bool(ok), "說明": detail})
        print(f"{'✓' if ok else '✗'} {name}：{detail}")
        return ok

    def send(lisp):
        logs.append(lisp)
        return acad.send(lisp)

    def circle_r(e):
        return float(acad.r(lambda: e.Radius))

    def read(e):
        return acad.read(e)[0]

    acad.ensure_undo(log=print)

    # ---------- 測試 1：直徑尺寸約束 ----------
    print("\n[測試 1] 對一個圓加直徑尺寸約束，再改數值")
    cx, cy, r0 = 1000.0, 0.0, 8.0
    c1 = acad.r(lambda: ms.AddCircle(acad.V((cx, cy)), r0)); created.append(c1)
    # 重要：用「點座標」選取圖元時，圖元必須在目前畫面上看得到，所以每個測試前先把畫面移到測試圖形
    acad.zoom_to([read(c1)], margin=2.0)
    on_c = (cx + r0, cy)                          # 圓上的點（右邊）
    loc = (cx + r0 + 6, cy + r0 + 6)              # 尺寸線放的位置
    fin = send(f'(command "_.DCDIAMETER" {_lisp_pt(on_c)} {_lisp_pt(loc)} "AIDIA1=20")')
    r1 = circle_r(c1)
    ok1 = step("1a 建立直徑約束 AIDIA1=20（原本直徑 16）", abs(r1 - 10) < TOL,
               f"指令{'已結束' if fin else '逾時被取消'}；半徑 {r0} → {r1:.6g}（預期 10）")
    if not ok1:
        # 備案：先用預設值建立（不改數值），再用改名＋編輯
        fin = send(f'(command "_.DCDIAMETER" {_lisp_pt((cx + r1, cy))} {_lisp_pt(loc)} "")')
        fin2 = send('(command "_.-PARAMETERS" "_Rename" "dia1" "AIDIA1")')
        step("1a' 備案：用預設值建立後改名", fin and fin2, f"建立指令{'已結束' if fin else '逾時'}，改名指令{'已結束' if fin2 else '逾時'}"
             "（這一步無法從幾何判斷，要看下一步改數值有沒有效）")
    fin = send('(command "_.-PARAMETERS" "_Edit" "AIDIA1" "30")')
    r2 = circle_r(c1)
    step("1b 用 -PARAMETERS 把 AIDIA1 改成 30", abs(r2 - 15) < TOL,
         f"指令{'已結束' if fin else '逾時被取消'}；半徑 → {r2:.6g}（預期 15）")

    # ---------- 測試 2：改直徑時，相切約束要維持 ----------
    print("\n[測試 2] 圓與一條線相切，改圓的直徑後是否仍相切")
    cx2 = 1100.0
    c2 = acad.r(lambda: ms.AddCircle(acad.V((cx2, cy)), r0)); created.append(c2)
    ln = acad.r(lambda: ms.AddLine(acad.V((cx2 - 20, cy + r0)), acad.V((cx2 + 20, cy + r0)))); created.append(ln)
    prims = [read(c2), read(ln)]
    acad.zoom_to(prims, margin=1.0)
    c = {"pair": [0, 1], "type": "tangent", "first": {"position": "whole", "position_zh": "整個"},
         "second": {"position": "whole", "position_zh": "整個"}, "text_zh": "相切：圓 與 線"}
    ok, res = acad.apply([c], prims, log=logs.append)
    step("2a 加相切約束", ok, res[0][1])
    acad.zoom_to([read(c2), read(ln)], margin=1.0)   # 套用相切後重新對準畫面
    on_c2 = (cx2 - r0, cy)                        # 圓的左邊（避開上方的線）
    fin = send(f'(command "_.DCDIAMETER" {_lisp_pt(on_c2)} {_lisp_pt((cx2 - r0 - 6, cy - r0 - 6))} "AIDIA2=16")')
    step("2b 加直徑約束 AIDIA2=16（等於目前值，圖形不應變）", fin and abs(circle_r(c2) - 8) < TOL,
         f"指令{'已結束' if fin else '逾時被取消'}；半徑 {circle_r(c2):.6g}")
    fin = send('(command "_.-PARAMETERS" "_Edit" "AIDIA2" "26")')
    gc, gl = read(c2), read(ln)
    rt = residual(gc, "whole", gl, "whole", "tangent")
    size = 2 * gc["radius"]
    step("2c 把 AIDIA2 改成 26：圓變大", abs(gc["radius"] - 13) < TOL, f"半徑 → {gc['radius']:.6g}（預期 13）")
    step("2d 改完後仍然相切", rt is not None and rt <= TOL * max(size, 1),
         f"相切殘差 {rt}；線現在在 y={gl['start'][1]:.6g}（圓心 y={gc['center'][1]:.6g}）")

    # ---------- 測試 3：線長（水平距離）尺寸約束 ----------
    print("\n[測試 3] 對一條水平線加水平距離約束，再改數值")
    x3 = 1200.0
    l3 = acad.r(lambda: ms.AddLine(acad.V((x3, cy)), acad.V((x3 + 30, cy)))); created.append(l3)
    acad.zoom_to([read(l3)], margin=1.0)
    on_l = (x3 + 15, cy)
    fin = send(f'(command "_.DCHORIZONTAL" "" {_lisp_pt(on_l)} {_lisp_pt((x3 + 15, cy - 8))} "AIL1=40")')
    g = read(l3)
    L = math.dist(g["start"], g["end"])
    step("3a 建立水平距離約束 AIL1=40（原本 30）", abs(L - 40) < TOL,
         f"指令{'已結束' if fin else '逾時被取消'}；長度 30 → {L:.6g}（預期 40）")
    fin = send('(command "_.-PARAMETERS" "_Edit" "AIL1" "55")')
    g = read(l3)
    L = math.dist(g["start"], g["end"])
    step("3b 把 AIL1 改成 55", abs(L - 55) < TOL, f"指令{'已結束' if fin else '逾時被取消'}；長度 → {L:.6g}（預期 55）")

    # ---------- 測試 4：用運算式連動（之後齒輪這類圖會用到）----------
    print("\n[測試 4] 用運算式讓兩個尺寸連動：AIL1 = AIDIA1 * 2")
    fin = send('(command "_.-PARAMETERS" "_Edit" "AIL1" "AIDIA1*2")')
    g = read(l3)
    L = math.dist(g["start"], g["end"])
    d = 2 * circle_r(c1)
    step("4a 設定 AIL1 = AIDIA1*2", abs(L - 2 * d) < TOL, f"直徑 {d:.6g}，長度 → {L:.6g}（預期 {2 * d:.6g}）")
    fin = send('(command "_.-PARAMETERS" "_Edit" "AIDIA1" "12")')
    g = read(l3)
    L = math.dist(g["start"], g["end"])
    step("4b 只改 AIDIA1 = 12，長度應自動變 24", abs(L - 24) < TOL,
         f"直徑 {2 * circle_r(c1):.6g}，長度 → {L:.6g}（預期 24）")

    # ===== 以下測試 ⑤ 尺寸視窗用到、但前面沒測過的指令（送出的指令和視窗完全相同，來自 dims.py）=====
    import dims as D
    print("\n[測試 5] 弧的半徑約束（DCRADIUS）")
    x5 = 1300.0
    a5 = acad.r(lambda: ms.AddArc(acad.V((x5, cy)), 10.0, 0.0, math.pi / 2)); created.append(a5)
    g = read(a5); acad.zoom_to([g], margin=1.5)
    fin = send(D.dim_command(g, "radius", "dimR901", "14", 2 * g["radius"]))
    step("5a 弧半徑 dimR901=14（原本 10）", abs(read(a5)["radius"] - 14) < TOL,
         f"指令{'已結束' if fin else '逾時被取消'}；半徑 → {read(a5)['radius']:.6g}（預期 14）")

    print("\n[測試 6] 垂直距離約束（DCVERTICAL）")
    x6 = 1400.0
    l6 = acad.r(lambda: ms.AddLine(acad.V((x6, cy)), acad.V((x6, cy + 25)))); created.append(l6)
    g = read(l6); acad.zoom_to([g], margin=1.0)
    fin = send(D.dim_command(g, "vertical", "dimV901", "35", 25))
    g = read(l6)
    step("6a 垂直距離 dimV901=35（原本 25）", abs(abs(g["end"][1] - g["start"][1]) - 35) < TOL,
         f"指令{'已結束' if fin else '逾時被取消'}；垂直距離 → {abs(g['end'][1] - g['start'][1]):.6g}（預期 35）")

    print("\n[測試 7] 斜線長度約束（DCALIGNED）")
    x7 = 1500.0
    l7 = acad.r(lambda: ms.AddLine(acad.V((x7, cy)), acad.V((x7 + 20, cy + 15)))); created.append(l7)
    g = read(l7); acad.zoom_to([g], margin=1.0)
    fin = send(D.dim_command(g, "length", "dimL901", "30", 25))
    g = read(l7)
    step("7a 斜線長度 dimL901=30（原本 25）", abs(math.dist(g["start"], g["end"]) - 30) < TOL,
         f"指令{'已結束' if fin else '逾時被取消'}；長度 → {math.dist(g['start'], g['end']):.6g}（預期 30）")

    print("\n[測試 8] 固定圓心（GCFIX 選點），再改直徑：圓心不應移動、直徑要能改，改由線移動")
    x8 = 1600.0
    c8 = acad.r(lambda: ms.AddCircle(acad.V((x8, cy)), r0)); created.append(c8)
    l8 = acad.r(lambda: ms.AddLine(acad.V((x8 - 20, cy + r0)), acad.V((x8 + 20, cy + r0)))); created.append(l8)
    pr = [read(c8), read(l8)]
    ok, res = acad.apply([dict(c, pair=[0, 1])], pr, log=logs.append)
    step("8a 加相切約束", ok, res[0][1])
    acad.zoom_to([read(c8), read(l8)], margin=1.0)
    fin = send(D.fix_command(read(c8), "center"))
    step("8b 固定圓心", fin, f"指令{'已結束' if fin else '逾時被取消'}（固定不會改變圖形，要看下一步）")
    acad.zoom_to([read(c8), read(l8)], margin=1.0)
    fin = send(D.dim_command(read(c8), "diameter", "dimD901", "16", 16))
    fin2 = send(D.edit_command("dimD901", "26"))
    gc, gl = read(c8), read(l8)
    rt = residual(gc, "whole", gl, "whole", "tangent")
    step("8c 直徑改成 26 後：圓心不動", math.dist(gc["center"], (x8, cy)) < TOL and abs(gc["radius"] - 13) < TOL,
         f"圓心 ({gc['center'][0]:.6g}, {gc['center'][1]:.6g})（應為 ({x8:g}, 0)），半徑 {gc['radius']:.6g}（預期 13）")
    step("8d 仍然相切（線移動了）", rt is not None and rt <= TOL * 26, f"相切殘差 {rt}；線在 y={gl['start'][1]:.6g}（預期 13）")

    print("\n[測試 9] 等比例縮放（SCALE）")
    x9 = 1700.0
    l9a = acad.r(lambda: ms.AddLine(acad.V((x9, cy)), acad.V((x9 + 30, cy)))); created.append(l9a)
    l9b = acad.r(lambda: ms.AddLine(acad.V((x9 + 30, cy)), acad.V((x9 + 30, cy + 20)))); created.append(l9b)
    acad.zoom_to([read(l9a), read(l9b)], margin=1.5)
    fin = send(D.scale_command([read(l9a)["handle"], read(l9b)["handle"]], (x9, cy), 2))
    ga, gb = read(l9a), read(l9b)
    step("9a 兩條線放大 2 倍（基準點不動）",
         abs(math.dist(ga["start"], ga["end"]) - 60) < TOL and abs(math.dist(gb["start"], gb["end"]) - 40) < TOL
         and math.dist(ga["start"], (x9, cy)) < TOL,
         f"指令{'已結束' if fin else '逾時被取消'}；長度 {math.dist(ga['start'], ga['end']):.6g}、"
         f"{math.dist(gb['start'], gb['end']):.6g}（預期 60、40）")

    n_ok = sum(s["成功"] for s in report["步驟"])
    report["總結"] = f"{n_ok}/{len(report['步驟'])} 成功"
    report["送出的指令"] = logs
    os.makedirs(C.RUNS_DIR, exist_ok=True)
    path = os.path.join(C.RUNS_DIR, "dim_test_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n{report['總結']}，報告已存到 {path}")
    acad.zoom_to([read(e) for e in created])
    if not keep:
        input("按 Enter 刪除測試圖形（想先看就去 AutoCAD 看，再回來按 Enter）...")
        for e in created:
            try:
                acad.r(lambda: e.Delete())
            except Exception:
                pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="尺寸約束可行性測試")
    ap.add_argument("--keep", action="store_true", help="保留測試圖形")
    main(ap.parse_args().keep)
