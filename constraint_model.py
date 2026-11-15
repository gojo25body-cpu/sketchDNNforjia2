"""
階段 2-2：約束建議模型

架構：
  1. 每個圖元 → 一個向量（類型、輔助線、幾何、方向、長度）
  2. Transformer 讓每個圖元「看到」整張圖的其他圖元（沒有位置編碼 → 圖元的順序不影響結果）
  3. 針對指定的圖元對 (A, B)：把 A、B 的向量 + 兩者之間的幾何關係（距離、角度、半徑差…）
     + 「第 1 對已選的約束」（沒有就用空條件）一起送進輸出層
  4. 輸出 128 個選項各自「成立的機率」（sigmoid），不合法的選項機率設為 0
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import constraint_config as C

PRIM_FEAT_DIM = 17
PAIR_FEAT_DIM = 46

# 每種圖元的「起點 / 終點 / 中心」是否有意義（點、線、圓、弧）
_KP_VALID = [[1, 1, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]]


def prim_features(kinds, geom, construct):
    """kinds (B,P) long，-1 = 空位；geom (B,P,7)；construct (B,P) bool → (B,P,17)"""
    valid = (kinds >= 0)
    k = kinds.clamp(min=0)
    onehot = F.one_hot(k, 4).to(geom.dtype)
    seg = ((kinds == C.PRIM_LINE) | (kinds == C.PRIM_ARC)).to(geom.dtype).unsqueeze(-1)
    d = geom[..., 2:4] - geom[..., 0:2]
    length = d.norm(dim=-1, keepdim=True)
    u = d / length.clamp(min=1e-6) * seg
    ang2 = torch.stack([u[..., 0] ** 2 - u[..., 1] ** 2, 2 * u[..., 0] * u[..., 1]], dim=-1)
    feats = torch.cat([onehot, construct.to(geom.dtype).unsqueeze(-1), geom, u, ang2, length * seg], dim=-1)
    return feats * valid.to(geom.dtype).unsqueeze(-1)


def pair_features(ga, gb, ka, kb, kp_valid):
    """ga, gb (N,7)；ka, kb (N,) long（0~3）→ (N,46) 兩個圖元之間的幾何關係"""
    dt = ga.dtype
    va, vb = kp_valid[ka].to(dt), kp_valid[kb].to(dt)                       # (N,3)
    pa = torch.stack([ga[:, 0:2], ga[:, 2:4], ga[:, 4:6]], dim=1)           # (N,3,2)
    pb = torch.stack([gb[:, 0:2], gb[:, 2:4], gb[:, 4:6]], dim=1)
    diff = pa.unsqueeze(2) - pb.unsqueeze(1)                                # (N,3,3,2)
    m = (va.unsqueeze(2) * vb.unsqueeze(1)).unsqueeze(-1)
    kp = torch.stack([diff.norm(dim=-1), diff[..., 0].abs(), diff[..., 1].abs()], dim=-1) * m
    kp = kp.flatten(1)                                                      # 27

    def unit(g):
        d = g[:, 2:4] - g[:, 0:2]
        n = d.norm(dim=-1, keepdim=True)
        return d / n.clamp(min=1e-6), n.squeeze(-1)

    ua, la = unit(ga)
    ub, lb = unit(gb)
    seg_a = ((ka == C.PRIM_LINE) | (ka == C.PRIM_ARC)).to(dt)
    seg_b = ((kb == C.PRIM_LINE) | (kb == C.PRIM_ARC)).to(dt)
    line_a, line_b = (ka == C.PRIM_LINE).to(dt), (kb == C.PRIM_LINE).to(dt)
    cur_a = ((ka == C.PRIM_CIRCLE) | (ka == C.PRIM_ARC)).to(dt)
    cur_b = ((kb == C.PRIM_CIRCLE) | (kb == C.PRIM_ARC)).to(dt)

    dot = (ua * ub).sum(-1).abs()
    cross = (ua[:, 0] * ub[:, 1] - ua[:, 1] * ub[:, 0]).abs()
    direction = torch.stack([dot, cross], -1) * (seg_a * seg_b).unsqueeze(-1)           # 2
    ra, rb = ga[:, 6], gb[:, 6]
    radius = torch.stack([(ra - rb).abs(), torch.minimum(ra, rb) / torch.maximum(ra, rb).clamp(min=1e-6)], -1) \
        * (cur_a * cur_b).unsqueeze(-1)                                                  # 2
    length = torch.stack([(la - lb).abs(), torch.minimum(la, lb) / torch.maximum(la, lb).clamp(min=1e-6)], -1) \
        * (line_a * line_b).unsqueeze(-1)                                                # 2

    def to_carrier(p, g, k, u):
        """p (N,3,2) 到圖元 g 的距離：線→到無限長直線；圓/弧→到圓周；點→到該點"""
        rel = p - g[:, None, 0:2]
        d_line = (rel[..., 0] * u[:, None, 1] - rel[..., 1] * u[:, None, 0]).abs()
        d_circ = ((p - g[:, None, 4:6]).norm(dim=-1) - g[:, None, 6]).abs()
        d_pt = rel.norm(dim=-1)
        kk = k.unsqueeze(-1)
        return torch.where(kk == C.PRIM_LINE, d_line, torch.where(kk == C.PRIM_POINT, d_pt, d_circ))

    a2b = to_carrier(pa, gb, kb, ub) * va                                   # 3
    b2a = to_carrier(pb, ga, ka, ua) * vb                                   # 3

    # 相切程度
    c_dist = (ga[:, 4:6] - gb[:, 4:6]).norm(dim=-1)
    tan_cc = torch.minimum((c_dist - (ra + rb)).abs(), (c_dist - (ra - rb).abs()).abs())
    rel_a = gb[:, 4:6] - ga[:, 0:2]     # B 的圓心到 A 線
    d_lc = (rel_a[:, 0] * ua[:, 1] - rel_a[:, 1] * ua[:, 0]).abs()
    rel_b = ga[:, 4:6] - gb[:, 0:2]
    d_cl = (rel_b[:, 0] * ub[:, 1] - rel_b[:, 1] * ub[:, 0]).abs()
    tan = torch.where((cur_a * cur_b) > 0, tan_cc,
                      torch.where((line_a * cur_b) > 0, (d_lc - rb).abs(),
                                  torch.where((cur_a * line_b) > 0, (d_cl - ra).abs(), torch.zeros_like(ra))))
    return torch.cat([kp, direction, radius, length, a2b, b2a, tan.unsqueeze(-1), va, vb], dim=-1)


def _mlp(i, h, o, dropout=0.0):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, o))


class ConstraintModel(nn.Module):
    def __init__(self, d=C.MODEL_DIM, layers=C.MODEL_LAYERS, heads=C.MODEL_HEADS, dropout=C.MODEL_DROPOUT):
        super().__init__()
        self.d = d
        self.register_buffer("kp_valid", torch.tensor(_KP_VALID, dtype=torch.float32), persistent=False)
        self.register_buffer("legal", torch.zeros(4, 4, C.NUM_OPTIONS, dtype=torch.bool))   # 會跟著存檔
        self.prim_in = _mlp(PRIM_FEAT_DIM, d, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.enc_norm = nn.LayerNorm(d)
        self.pair_in = _mlp(PAIR_FEAT_DIM, d, d)
        self.opt_emb = nn.Embedding(C.NUM_OPTIONS, d)
        self.cond_mlp = _mlp(3 * d, d, d)
        self.null_cond = nn.Parameter(torch.zeros(d))
        self.head = _mlp(5 * d, 2 * d, C.NUM_OPTIONS, dropout)

    def encode(self, kinds, geom, construct):
        x = self.prim_in(prim_features(kinds, geom, construct))
        x = self.encoder(x, src_key_padding_mask=(kinds < 0))
        return self.enc_norm(x)

    @staticmethod
    def _gather(t, idx):
        """t (B,P,...)，idx (B,K) → (B,K,...)"""
        shape = idx.shape + t.shape[2:]
        flat = idx.reshape(idx.shape[0], -1)
        ex = flat.view(*flat.shape, *([1] * (t.dim() - 2))).expand(*flat.shape, *t.shape[2:])
        return t.gather(1, ex).view(shape)

    def forward(self, kinds, geom, construct, qa, qb, ca=None, cb=None, copt=None, has_cond=None, h=None):
        """qa, qb (B,K)：要預測的圖元對；ca, cb, copt, has_cond (B,K)：第 1 對與它選的選項（可省略）
        回傳 logits (B,K,128) 與合法遮罩 (B,K,128)"""
        if h is None:
            h = self.encode(kinds, geom, construct)
        B, K = qa.shape
        ha, hb = self._gather(h, qa), self._gather(h, qb)
        ka, kb = self._gather(kinds, qa), self._gather(kinds, qb)
        ga, gb = self._gather(geom, qa), self._gather(geom, qb)
        pf = pair_features(ga.reshape(-1, 7), gb.reshape(-1, 7), ka.reshape(-1), kb.reshape(-1),
                           self.kp_valid.to(geom.dtype))
        pf = self.pair_in(pf).view(B, K, -1)
        dt = pf.dtype     # 混合精度時各層輸出型別可能不同，統一成同一型別再串接
        cond = self.null_cond.view(1, 1, -1).expand(B, K, -1).to(dt)
        if has_cond is not None and bool(has_cond.any()):
            hc = torch.cat([self._gather(h, ca).to(dt), self._gather(h, cb).to(dt), self.opt_emb(copt).to(dt)], dim=-1)
            c = self.cond_mlp(hc).to(dt)
            cond = torch.where(has_cond.unsqueeze(-1), c, cond)
        ha, hb = ha.to(dt), hb.to(dt)
        x = torch.cat([ha, hb, ha * hb, pf, cond], dim=-1)
        logits = self.head(x)
        legal = self.legal[ka.clamp(min=0), kb.clamp(min=0)]
        return logits, legal

    @torch.no_grad()
    def probs(self, *args, **kw):
        logits, legal = self.forward(*args, **kw)
        return torch.sigmoid(logits.float()) * legal
