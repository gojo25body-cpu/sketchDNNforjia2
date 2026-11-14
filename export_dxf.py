import os, math, torch, ezdxf
import utils

IN_FILE = os.path.join("samples", "samples.pt")
OUT_DIR = "samples"
SIZE_MM = 100.0          # 草圖外框邊長（mm），可自行調整
S = SIZE_MM / 2          # 模型座標範圍是 [-1, 1]

nodes_all = torch.load(IN_FILE)

for i, nodes in enumerate(nodes_all):
    nodes = utils.ToNaive(nodes.clone())   # 弧轉成 圓心 / 半徑 / 起訖角

    doc = ezdxf.new("R2018", setup=True)
    doc.layers.add("GEOMETRY", color=7)
    doc.layers.add("CONSTRUCTION", color=8, linetype="DASHED")
    msp = doc.modelspace()
    count = {"line": 0, "circle": 0, "arc": 0, "point": 0}

    for p in nodes:
        kind = int(torch.argmax(p[2:7]))           # 0 線, 1 圓, 2 弧, 3 點, 4 無
        layer = "CONSTRUCTION" if p[0] < 0.5 else "GEOMETRY"
        attr = {"layer": layer}

        if kind == 0:
            x1, y1, x2, y2 = (p[7:11] * S).tolist()
            if math.hypot(x2 - x1, y2 - y1) < 1e-6: continue
            msp.add_line((x1, y1), (x2, y2), dxfattribs=attr); count["line"] += 1
        elif kind == 1:
            cx, cy, r = (p[11:14] * S).tolist()
            if r <= 1e-6: continue
            msp.add_circle((cx, cy), r, dxfattribs=attr); count["circle"] += 1
        elif kind == 2:
            cx, cy, r = (p[14:17] * S).tolist()
            a0, a1 = (p[17:19] * 360).tolist()     # 角度存的是 0~1 的比例
            if r <= 1e-6: continue
            msp.add_arc((cx, cy), r, a0, a1, dxfattribs=attr); count["arc"] += 1
        elif kind == 3:
            x, y = (p[19:21] * S).tolist()
            msp.add_point((x, y), dxfattribs=attr); count["point"] += 1

    path = os.path.join(OUT_DIR, f"sample_{i}.dxf")
    doc.saveas(path)
    print(f"{path}: {count}")