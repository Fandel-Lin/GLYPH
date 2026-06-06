import os
import json
import glob
import math
import argparse
import numpy as np
import rasterio
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from rasterio import features
from shapely.geometry import shape as shp_shape, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from rasterio.enums import Resampling
from tqdm import tqdm


# ============================================================
# Filesystem
# ============================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ============================================================
# JSON parsing (LabelMe-style)
# ============================================================
def read_json_shapes(json_path: str):
    with open(json_path, "r") as f:
        j = json.load(f)
    shapes = j.get("shapes", [])
    labels = [s.get("label", "") for s in shapes]
    return shapes, labels


def bbox_from_points(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0 = int(math.floor(min(xs)))
    y0 = int(math.floor(min(ys)))
    x1 = int(math.ceil(max(xs)))
    y1 = int(math.ceil(max(ys)))
    return x0, y0, x1, y1


def clip_bbox(x0, y0, x1, y1, W, H, pad=0):
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W, x1 + pad)
    y1 = min(H, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


# ============================================================
# Raster reads (low-res)
# ============================================================
def read_rgb_lowres_u8(src_img: rasterio.DatasetReader, scale: float, read_bands=3) -> Tuple[np.ndarray, Tuple[int, int]]:
    H, W = src_img.height, src_img.width
    h2 = max(1, int(round(H * scale)))
    w2 = max(1, int(round(W * scale)))

    img = src_img.read(
        indexes=list(range(1, read_bands + 1)),
        out_shape=(read_bands, h2, w2),
        resampling=Resampling.bilinear
    )

    if img.dtype == np.uint8:
        rgb = np.transpose(img[:3], (1, 2, 0)).copy()
    else:
        if img.dtype == np.uint16:
            denom = 65535.0
        else:
            denom = float(np.max(img)) if img.size else 1.0
            denom = max(denom, 1.0)
        rgb = np.transpose(
            (img[:3].astype(np.float32) / denom * 255.0).clip(0, 255).astype(np.uint8),
            (1, 2, 0)
        )
    return rgb, (h2, w2)


def read_mask_lowres(mask_path: str, out_hw: Tuple[int, int], threshold=0) -> np.ndarray:
    h2, w2 = out_hw
    with rasterio.open(mask_path) as src_m:
        m = src_m.read(
            1,
            out_shape=(h2, w2),
            resampling=Resampling.nearest
        )
    return (m > threshold).astype(np.uint8)


def build_expert_label_map_lowres(base: str, labels: List[str], sol_dir: str, out_hw: Tuple[int, int]) -> np.ndarray:
    """
    y in {0..L} at low-res by reading per-label hard masks.
    Tie-break: first label wins (stable).
    """
    h2, w2 = out_hw
    y = np.zeros((h2, w2), dtype=np.int16)
    for li, lab in enumerate(labels, start=1):
        p = os.path.join(sol_dir, f"{base}_{lab}.tif")
        if not os.path.exists(p):
            continue
        mk = read_mask_lowres(p, out_hw, threshold=0)
        fill = (mk == 1) & (y == 0)
        if np.any(fill):
            y[fill] = li
    return y


# ============================================================
# Boundary + regions
# ============================================================
def boundary_from_label_map(y: np.ndarray) -> np.ndarray:
    H, W = y.shape
    b = np.zeros((H, W), dtype=np.uint8)

    d = (y[1:, :] != y[:-1, :])
    b[1:, :] |= d.astype(np.uint8)
    b[:-1, :] |= d.astype(np.uint8)

    d = (y[:, 1:] != y[:, :-1])
    b[:, 1:] |= d.astype(np.uint8)
    b[:, :-1] |= d.astype(np.uint8)

    return b


def connected_components_regions(non_boundary: np.ndarray) -> Tuple[np.ndarray, int]:
    nb = (non_boundary.astype(np.uint8) * 255)
    K, cc = cv2.connectedComponents(nb, connectivity=4)
    region_id = cc.astype(np.int32) - 1
    K_regions = K - 1
    if K_regions <= 0:
        region_id[:, :] = 0
        K_regions = 1
    return region_id, K_regions


def propagate_region_ids_into_boundaries(region_id: np.ndarray, max_iters=128) -> np.ndarray:
    rid = region_id.copy()
    lab = rid + 1
    lab[lab < 0] = 0
    max_lab = int(lab.max()) if lab.size else 0

    if max_lab <= 65535:
        lab_m = lab.astype(np.uint16, copy=False)
        zero = np.uint16(0)
    else:
        lab_m = lab.astype(np.float32, copy=False)
        zero = np.float32(0.0)

    kernel = np.ones((3, 3), np.uint8)

    unk = (lab_m == zero)
    it = 0
    while np.any(unk) and it < max_iters:
        dil = cv2.dilate(lab_m, kernel, iterations=1)
        fill = (lab_m == zero) & (dil != zero)
        if not np.any(fill):
            break
        lab_m[fill] = dil[fill]
        unk = (lab_m == zero)
        it += 1

    lab_m[lab_m == zero] = 1
    rid_full = lab_m.astype(np.int32) - 1
    rid_full[rid_full < 0] = 0
    return rid_full.astype(np.int32)


def merge_tiny_regions(region_id: np.ndarray, min_area: int = 8) -> np.ndarray:
    H, W = region_id.shape
    rid = region_id.copy()
    K = int(rid.max()) + 1
    if K <= 1:
        return rid

    area = np.bincount(rid.reshape(-1), minlength=K).astype(np.int32)
    tiny = np.where(area < min_area)[0]
    if tiny.size == 0:
        return rid

    a = rid[:, :-1].reshape(-1)
    b = rid[:, 1:].reshape(-1)
    m = (a != b)
    a1 = a[m]; b1 = b[m]

    a = rid[:-1, :].reshape(-1)
    b = rid[1:, :].reshape(-1)
    m = (a != b)
    a2 = a[m]; b2 = b[m]

    src = np.concatenate([a1, b1, a2, b2], axis=0)
    dst = np.concatenate([b1, a1, b2, a2], axis=0)

    key = src.astype(np.int64) * K + dst.astype(np.int64)
    bc = np.bincount(key, minlength=K * K).reshape(K, K).astype(np.int32)

    for t in tiny.tolist():
        neigh = bc[t]
        neigh[t] = 0
        j = int(np.argmax(neigh))
        if neigh[j] <= 0:
            continue
        rid[rid == t] = j

    uniq = np.unique(rid)
    remap = np.zeros((int(uniq.max()) + 1,), dtype=np.int32)
    remap[uniq] = np.arange(uniq.size, dtype=np.int32)
    rid = remap[rid]
    return rid


# ============================================================
# RGB->LAB (on GPU)
# ============================================================
@torch.no_grad()
def rgb_u8_to_lab_torch(rgb_u8: torch.Tensor) -> torch.Tensor:
    """
    Input:  rgb_u8 (...,3) uint8
    Output: lab    (...,3) float32  (L in [0,100], a/b roughly [-128,127])

    Works for:
      - [H,W,3]
      - [N,3]
      - [N,1,1,3]
      - any shape ending in 3.
    """
    if rgb_u8.dtype != torch.uint8:
        raise TypeError(f"rgb_u8_to_lab_torch expects uint8, got {rgb_u8.dtype}")

    device = rgb_u8.device
    orig_shape = rgb_u8.shape
    if orig_shape[-1] != 3:
        raise ValueError(f"rgb_u8_to_lab_torch expects last dim=3, got shape={orig_shape}")

    x = rgb_u8.float() / 255.0
    x = x.reshape(-1, 3)  # [N,3]

    # sRGB -> linear
    a = 0.055
    lin = torch.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)  # [N,3]

    # RGB->XYZ (D65) with matmul (no einsum shape pitfalls)
    M = torch.tensor(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]],
        device=device, dtype=torch.float32
    )  # [3,3]
    xyz = lin @ M.t()  # [N,3]

    white = torch.tensor([0.95047, 1.00000, 1.08883], device=device, dtype=torch.float32)
    xyz = xyz / white

    eps = 216 / 24389
    kappa = 24389 / 27
    f = torch.where(xyz > eps, xyz ** (1/3), (kappa * xyz + 16) / 116)

    L = 116 * f[:, 1] - 16
    a_ = 500 * (f[:, 0] - f[:, 1])
    b_ = 200 * (f[:, 1] - f[:, 2])

    lab = torch.stack([L, a_, b_], dim=1)  # [N,3]
    lab = lab.reshape(*orig_shape[:-1], 3)
    return lab


@torch.no_grad()
def compute_region_stats_gpu(lab: torch.Tensor, region_id: torch.Tensor, K: int) -> Dict[str, torch.Tensor]:
    """
    lab: [H,W,3] float32
    region_id: [H,W] int64 in [0..K-1]
    """
    rid = region_id.view(-1)
    x = lab.view(-1, 3)

    cnt = torch.bincount(rid, minlength=K).float().clamp(min=1.0)

    mu = torch.zeros((K, 3), device=lab.device, dtype=torch.float32)
    mu.scatter_add_(0, rid[:, None].expand(-1, 3), x)
    mu = mu / cnt[:, None]

    m2 = torch.zeros((K, 3), device=lab.device, dtype=torch.float32)
    m2.scatter_add_(0, rid[:, None].expand(-1, 3), x * x)
    m2 = m2 / cnt[:, None]

    var = (m2 - mu * mu).clamp(min=0.0)
    return {"mu": mu, "var": var, "cnt": cnt}


@torch.no_grad()
def compute_global_style_features(lab: torch.Tensor) -> torch.Tensor:
    mu = lab.view(-1, 3).mean(dim=0)
    std = lab.view(-1, 3).std(dim=0)

    Lc = lab[..., 0].unsqueeze(0).unsqueeze(0)
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], device=lab.device, dtype=torch.float32).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], device=lab.device, dtype=torch.float32).view(1, 1, 3, 3)
    gx = F.conv2d(Lc, kx, padding=1)
    gy = F.conv2d(Lc, ky, padding=1)
    g = torch.sqrt(gx * gx + gy * gy + 1e-6).mean().view(1)

    return torch.cat([mu, std, g], dim=0)  # [7]


# ============================================================
# Expert votes
# ============================================================
def expert_region_vote_counts(region_id: np.ndarray, y_expert: np.ndarray, K: int, L: int) -> np.ndarray:
    rid = region_id.reshape(-1).astype(np.int64)
    lab = y_expert.reshape(-1).astype(np.int64)
    base = L + 1
    key = rid * base + lab
    bc = np.bincount(key, minlength=K * base).reshape(K, base).astype(np.int32)
    return bc


def agreement_mask_from_majority(countsA, countsB, countsC) -> Tuple[np.ndarray, np.ndarray]:
    majA = np.argmax(countsA, axis=1).astype(np.int32)
    majB = np.argmax(countsB, axis=1).astype(np.int32)
    majC = np.argmax(countsC, axis=1).astype(np.int32)

    K = majA.shape[0]
    agree2 = np.zeros((K,), dtype=bool)
    for k in range(K):
        v0, v1, v2 = int(majA[k]), int(majB[k]), int(majC[k])
        if (v0 == v1) or (v0 == v2) or (v1 == v2):
            agree2[k] = True
    maj = np.stack([majA, majB, majC], axis=1)
    return maj, agree2


# ============================================================
# Swatch sampling (low-res)
# ============================================================
def extract_swatch_pixels_lowres(
    rgb_lr_u8: np.ndarray,
    shapes,
    labels,
    full_hw: Tuple[int, int],
    low_hw: Tuple[int, int],
    swatch_pad: int = 1,
    max_pixels_per_swatch: int = 4096,
    seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    H, W = full_hw
    h2, w2 = low_hw

    sx = w2 / max(1, W)
    sy = h2 / max(1, H)

    all_rgb = []
    all_y = []

    for li, (s, lab) in enumerate(zip(shapes, labels), start=1):
        pts = s.get("points", None)
        if not pts or len(pts) < 2:
            continue
        x0, y0, x1, y1 = bbox_from_points(pts)
        bb = clip_bbox(x0, y0, x1, y1, W=W, H=H, pad=swatch_pad)
        if bb is None:
            continue
        x0, y0, x1, y1 = bb

        lx0 = int(math.floor(x0 * sx)); lx1 = int(math.ceil(x1 * sx))
        ly0 = int(math.floor(y0 * sy)); ly1 = int(math.ceil(y1 * sy))
        lx0 = max(0, min(w2 - 1, lx0)); lx1 = max(1, min(w2, lx1))
        ly0 = max(0, min(h2 - 1, ly0)); ly1 = max(1, min(h2, ly1))
        if lx1 <= lx0 or ly1 <= ly0:
            continue

        crop = rgb_lr_u8[ly0:ly1, lx0:lx1]
        if crop.size == 0:
            continue

        pix = crop.reshape(-1, 3)
        if pix.shape[0] > max_pixels_per_swatch:
            idx = rng.choice(pix.shape[0], size=max_pixels_per_swatch, replace=False)
            pix = pix[idx]

        all_rgb.append(pix)
        all_y.append(np.full((pix.shape[0],), li, dtype=np.int32))

    if len(all_rgb) == 0:
        return np.zeros((0, 3), dtype=np.uint8), np.zeros((0,), dtype=np.int32)

    X = np.concatenate(all_rgb, axis=0).astype(np.uint8)
    y = np.concatenate(all_y, axis=0).astype(np.int32)
    return X, y


# ============================================================
# Morphological postprocessing
# ============================================================
def morph_open_mask_u8(mask_u8_0_255: np.ndarray, k: int = 3, iters: int = 1) -> np.ndarray:
    """
    Morphological opening to remove small FP blobs.
    Input/Output: uint8 mask in {0,255}.
    """
    if mask_u8_0_255.dtype != np.uint8:
        mask_u8_0_255 = mask_u8_0_255.astype(np.uint8, copy=False)

    k = int(k)
    if k <= 1:
        return mask_u8_0_255
    if (k % 2) == 0:
        k += 1  # force odd

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = mask_u8_0_255

    # Open = erode then dilate
    m = cv2.erode(m, kernel, iterations=int(iters))
    m = cv2.dilate(m, kernel, iterations=int(iters))
    return m


# ============================================================
# Geometric postprocessing
# ============================================================
def _iter_polys(geom):
    """Yield Polygon(s) from any shapely geometry."""
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            yield g
    elif isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            yield from _iter_polys(g)

def _fill_small_holes(poly: Polygon, max_hole_area: float) -> Polygon:
    """Remove interior rings whose area <= max_hole_area."""
    if poly.is_empty:
        return poly
    if not poly.interiors:
        return poly
    keep_interiors = []
    for ring in poly.interiors:
        hole_poly = Polygon(ring)
        if hole_poly.area > max_hole_area:
            keep_interiors.append(ring)
    return Polygon(poly.exterior, keep_interiors)

def vector_simplify_and_fill_holes_u8(
    mask_u8_0_255: np.ndarray,
    transform,
    simplify_tol_px: float = 1.0,
    min_poly_area_px: float = 20.0,
    max_hole_area_px: float = 30.0,
) -> np.ndarray:
    if mask_u8_0_255.dtype != np.uint8:
        mask_u8_0_255 = mask_u8_0_255.astype(np.uint8, copy=False)

    # boolean mask
    m = (mask_u8_0_255 > 0)

    if not np.any(m):
        return mask_u8_0_255  # nothing to do

    H, W = m.shape

    # 1) raster -> vector polygons
    # rasterio.features.shapes returns geojson geometries in the raster coordinate system.
    geoms = []
    for geom, val in features.shapes(m.astype(np.uint8), mask=m, transform=transform):
        if val != 1:
            continue
        g = shp_shape(geom)
        if not g.is_empty:
            geoms.append(g)

    if not geoms:
        return mask_u8_0_255

    # 2) merge overlapping pieces (optional but helps simplify)
    merged = unary_union(geoms)

    # 3) simplify + drop tiny polys + fill tiny holes
    cleaned = []
    for poly in _iter_polys(merged):
        if poly.is_empty:
            continue

        # simplify geometry (preserve topology prevents self-crossings but can keep artifacts;
        # we still buffer(0) after to clean)
        p = poly.simplify(float(simplify_tol_px), preserve_topology=True)

        # fix minor invalidities introduced by simplify
        if not p.is_valid:
            p = p.buffer(0)

        # p may become MultiPolygon after buffer(0)
        for q in _iter_polys(p):
            if q.is_empty:
                continue

            # fill small FN holes (remove small interior rings)
            q2 = _fill_small_holes(q, float(max_hole_area_px))

            # drop tiny polygons
            if q2.area >= float(min_poly_area_px):
                cleaned.append(q2)

    if not cleaned:
        return np.zeros((H, W), dtype=np.uint8)

    final_geom = unary_union(cleaned)

    # 4) rasterize back to mask
    out = features.rasterize(
        [(final_geom, 1)],
        out_shape=(H, W),
        transform=transform,
        fill=0,
        all_touched=False,  # keep boundaries stable
        dtype=np.uint8
    )
    return (out * 255).astype(np.uint8)


# ============================================================
# Lightweight learning modules
# ============================================================
class ColorEmbed(nn.Module):
    def __init__(self, emb_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, emb_dim),
        )

    def forward(self, lab3):
        z = self.net(lab3)
        return F.normalize(z, dim=-1)


class ExpertGater(nn.Module):
    def __init__(self, in_dim=7 + 6, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)


def supervised_contrastive_loss(z: torch.Tensor, y: torch.Tensor, T: float = 0.2) -> torch.Tensor:
    sim = (z @ z.t()) / T
    sim = sim - sim.max(dim=1, keepdim=True).values
    exp = torch.exp(sim)

    y = y.view(-1, 1)
    pos = (y == y.t()).float()
    pos.fill_diagonal_(0.0)

    denom = exp.sum(dim=1) - torch.exp(sim.diagonal())
    denom = denom.clamp(min=1e-6)

    num = (exp * pos).sum(dim=1).clamp(min=1e-6)
    loss = -torch.log(num / denom)
    has_pos = (pos.sum(dim=1) > 0)
    if has_pos.any():
        return loss[has_pos].mean()
    return loss.mean() * 0.0


# ============================================================
# Config
# ============================================================
@dataclass
class Config:
    region_scale: float = 0.25
    boundary_vote_thr: int = 1
    max_propagate_iters: int = 128
    min_region_area: int = 8

    roi_dir: Optional[str] = None

    swatch_pad: int = 1
    max_swatch_pixels: int = 4096
    max_conf_region_samples: int = 8192

    emb_dim: int = 32
    steps: int = 256
    lr: float = 5e-3
    contrast_T: float = 0.2
    lambda_region: float = 0.5
    lambda_gate: float = 0.1
    lambda_sim: float = 1.0
    tau_base: float = 0.4
    tau_scale: float = 0.8

    bg_vote_bias: float = 0.25
    sim_bg_reject: float = 0.20

    enable_cache: bool = True
    seed: int = 39

    write_prelim: bool = True
    read_bands: int = 3

    preview_max_side: int = 2000

    morph_open: bool = True
    morph_kernel: int = 11
    morph_iters: int = 1

    vectorize_simplify: bool = True
    simplify_tol_px: float = 4.0
    min_poly_area_px: float = 32.0
    max_hole_area_px: float = 32.0


# ============================================================
# Refiner
# ============================================================
class LogicPlusContrastRefiner:
    def __init__(self, device: str, cfg: Config):
        self.device = device
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.seed)

        self.color_embed = ColorEmbed(emb_dim=cfg.emb_dim).to(device)
        self.gater = ExpertGater(in_dim=7 + 6).to(device)

        self.cache: Dict[str, Dict] = {}

    def _load_roi_lowres(self, base: str, out_hw: Tuple[int, int]) -> Optional[np.ndarray]:
        if not self.cfg.roi_dir:
            return None
        p = os.path.join(self.cfg.roi_dir, f"{base}.tif")
        if not os.path.exists(p):
            return None
        return read_mask_lowres(p, out_hw, threshold=0)

    def _prepare_one(self, img_path: str, json_path: str, sol_dirs: List[str]):
        base = os.path.basename(img_path).replace(".tif", "")
        if self.cfg.enable_cache and base in self.cache:
            return self.cache[base]

        shapes, labels = read_json_shapes(json_path)
        L = len(labels)
        if L == 0:
            return None

        with rasterio.open(img_path) as src_img:
            H, W = src_img.height, src_img.width
            rgb_lr_u8, (h2, w2) = read_rgb_lowres_u8(src_img, self.cfg.region_scale, read_bands=self.cfg.read_bands)

        roi_lr = self._load_roi_lowres(base, (h2, w2))

        yA = build_expert_label_map_lowres(base, labels, sol_dirs[0], (h2, w2))
        yB = build_expert_label_map_lowres(base, labels, sol_dirs[1], (h2, w2))
        yC = build_expert_label_map_lowres(base, labels, sol_dirs[2], (h2, w2))

        bA = boundary_from_label_map(yA)
        bB = boundary_from_label_map(yB)
        bC = boundary_from_label_map(yC)
        bsum = (bA.astype(np.uint8) + bB.astype(np.uint8) + bC.astype(np.uint8))
        union_boundary = (bsum >= int(self.cfg.boundary_vote_thr)).astype(np.uint8)

        if roi_lr is not None:
            union_boundary = np.maximum(union_boundary, (1 - roi_lr).astype(np.uint8))

        non_boundary = (1 - union_boundary).astype(np.uint8)
        region_id, _ = connected_components_regions(non_boundary)
        region_id = propagate_region_ids_into_boundaries(region_id, max_iters=self.cfg.max_propagate_iters)
        region_id = merge_tiny_regions(region_id, min_area=self.cfg.min_region_area)
        K = int(region_id.max()) + 1

        sw_rgb, sw_y = extract_swatch_pixels_lowres(
            rgb_lr_u8,
            shapes, labels,
            full_hw=(H, W),
            low_hw=(h2, w2),
            swatch_pad=self.cfg.swatch_pad,
            max_pixels_per_swatch=self.cfg.max_swatch_pixels,
            seed=self.cfg.seed
        )

        pack = {
            "base": base,
            "labels": labels,
            "L": L,
            "H": H, "W": W,
            "h2": h2, "w2": w2,
            "rgb_lr_u8": rgb_lr_u8,
            "roi_lr": roi_lr,
            "yA": yA, "yB": yB, "yC": yC,
            "union_boundary": union_boundary,
            "region_id": region_id,
            "K": K,
            "sw_rgb": sw_rgb,
            "sw_y": sw_y,
            "img_path": img_path,
        }
        if self.cfg.enable_cache:
            self.cache[base] = pack
        return pack

    def _write_prelim_experts(self, pack: Dict, preliminary_dir: str):
        if not (self.cfg.write_prelim and preliminary_dir):
            return
        ensure_dir(preliminary_dir)
        base = pack["base"]
        L = pack["L"]

        # rgb_lr
        cv2.imwrite(os.path.join(preliminary_dir, f"{base}_rgb_lr.png"), pack["rgb_lr_u8"][..., ::-1])

        # roi_lr
        if pack["roi_lr"] is not None:
            cv2.imwrite(os.path.join(preliminary_dir, f"{base}_roi_lr.png"), (pack["roi_lr"] * 255).astype(np.uint8))

        # expert label maps as random colors
        cmap = np.random.RandomState(123).randint(0, 255, size=(L + 1, 3), dtype=np.uint8)
        cmap[0] = np.array([0, 0, 0], dtype=np.uint8)

        for tag, y in [("A", pack["yA"]), ("B", pack["yB"]), ("C", pack["yC"])]:
            vis = cmap[y.astype(np.int32)]
            cv2.imwrite(os.path.join(preliminary_dir, f"{base}_expert{tag}_lr.png"), vis[..., ::-1])

        # boundary/regions
        rid_np = pack["region_id"]
        Kc = max(1, int(rid_np.max()) + 1)
        rid_vis = (rid_np.astype(np.float32) / max(1, (Kc - 1)) * 255.0).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(preliminary_dir, f"{base}_regions_id_lr.png"), rid_vis)
        cv2.imwrite(os.path.join(preliminary_dir, f"{base}_union_boundary_lr.png"), (pack["union_boundary"] * 255).astype(np.uint8))

    def _train_per_image(self, pack: Dict):
        device = self.device
        L = pack["L"]
        K = pack["K"]

        if pack["sw_rgb"].shape[0] < 64:
            return

        rgb_lr = torch.from_numpy(pack["rgb_lr_u8"]).to(device, non_blocking=True)
        lab_lr = rgb_u8_to_lab_torch(rgb_lr)
        rid = torch.from_numpy(pack["region_id"].astype(np.int64)).to(device, non_blocking=True)

        stats = compute_region_stats_gpu(lab_lr, rid, K)
        mu = stats["mu"]

        countsA = expert_region_vote_counts(pack["region_id"], pack["yA"], K, L)
        countsB = expert_region_vote_counts(pack["region_id"], pack["yB"], K, L)
        countsC = expert_region_vote_counts(pack["region_id"], pack["yC"], K, L)
        maj, agree2 = agreement_mask_from_majority(countsA, countsB, countsC)

        majA, majB, majC = maj[:, 0], maj[:, 1], maj[:, 2]
        disagree = ((majA != majB).astype(np.float32) + (majA != majC).astype(np.float32) + (majB != majC).astype(np.float32)) / 3.0
        disagree_rate = float(disagree.mean())

        def maj_frac(counts):
            m = np.max(counts, axis=1).astype(np.float32)
            s = np.sum(counts, axis=1).astype(np.float32)
            s = np.maximum(s, 1.0)
            return m / s

        fracA = maj_frac(countsA)
        fracB = maj_frac(countsB)
        fracC = maj_frac(countsC)

        conf_vec = np.array([
            float(fracA.mean()), float(fracB.mean()), float(fracC.mean()),
            float(np.median(fracA)), float(np.median(fracB)), float(np.median(fracC))
        ], dtype=np.float32)

        style = compute_global_style_features(lab_lr).detach().cpu().numpy().astype(np.float32)
        gate_in = np.concatenate([style, conf_vec], axis=0)
        gate_in_t = torch.from_numpy(gate_in).to(device)

        self.color_embed.train()
        self.gater.train()
        opt = torch.optim.AdamW(list(self.color_embed.parameters()) + list(self.gater.parameters()), lr=self.cfg.lr)

        # swatch samples to LAB on GPU 
        sw_rgb = torch.from_numpy(pack["sw_rgb"].astype(np.uint8)).to(device)   # [M,3] uint8
        sw_lab = rgb_u8_to_lab_torch(sw_rgb)                                    # [M,3] float
        sw_y = torch.from_numpy(pack["sw_y"].astype(np.int64)).to(device)       # [M] labels 1..L

        agree_idx = np.where(agree2)[0]
        if agree_idx.size > 0:
            if agree_idx.size > self.cfg.max_conf_region_samples:
                agree_idx = self.rng.choice(agree_idx, size=self.cfg.max_conf_region_samples, replace=False)

            pseudo = []
            for k in agree_idx.tolist():
                v0, v1, v2 = int(majA[k]), int(majB[k]), int(majC[k])
                if v0 == v1 or v0 == v2:
                    pseudo.append(v0)
                else:
                    pseudo.append(v1)
            pseudo_t = torch.tensor(pseudo, dtype=torch.long, device=device)

            agree_idx_t = torch.from_numpy(agree_idx.astype(np.int64)).to(device)
            fg = (pseudo_t > 0)
            reg_lab = mu[agree_idx_t[fg]]
            reg_y = pseudo_t[fg]
        else:
            reg_lab = None
            reg_y = None

        for _ in range(int(self.cfg.steps)):
            w = self.gater(gate_in_t)

            z_sw = self.color_embed(sw_lab)
            loss_sw = supervised_contrastive_loss(z_sw, sw_y, T=self.cfg.contrast_T)

            if reg_lab is not None and reg_lab.shape[0] >= 32:
                z_reg = self.color_embed(reg_lab)
                loss_reg = supervised_contrastive_loss(z_reg, reg_y, T=self.cfg.contrast_T)
            else:
                loss_reg = torch.zeros((), device=device)

            ent = -(w * (w + 1e-8).log()).sum()
            ent_target = self.cfg.tau_base + self.cfg.tau_scale * disagree_rate
            loss_gate = (ent - ent_target).abs()

            loss = loss_sw + self.cfg.lambda_region * loss_reg + self.cfg.lambda_gate * loss_gate 

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        self.color_embed.eval()
        self.gater.eval()

    @torch.no_grad()
    def _infer_per_image(self, pack: Dict, preliminary_dir: str) -> np.ndarray:
        device = self.device
        L = pack["L"]
        K = pack["K"]

        rgb_lr = torch.from_numpy(pack["rgb_lr_u8"]).to(device, non_blocking=True)
        lab_lr = rgb_u8_to_lab_torch(rgb_lr)
        rid = torch.from_numpy(pack["region_id"].astype(np.int64)).to(device, non_blocking=True)

        stats = compute_region_stats_gpu(lab_lr, rid, K)
        mu = stats["mu"]
        z_reg = self.color_embed(mu)  # [K,D]

        # Swatch prototypes
        sw_rgb = pack["sw_rgb"]
        sw_y = pack["sw_y"]
        if sw_rgb.shape[0] < 16:
            sw_proto = torch.zeros((L, self.cfg.emb_dim), device=device)
        else:
            sw_rgb_t = torch.from_numpy(sw_rgb.astype(np.uint8)).to(device)  # [M,3]
            sw_lab = rgb_u8_to_lab_torch(sw_rgb_t)                           # [M,3]
            z_sw = self.color_embed(sw_lab)                                  # [M,D]
            y_sw = torch.from_numpy(sw_y.astype(np.int64)).to(device)        # [M] 1..L

            sw_proto = torch.zeros((L + 1, z_sw.shape[1]), device=device)
            cnt = torch.zeros((L + 1,), device=device)
            sw_proto.scatter_add_(0, y_sw[:, None].expand(-1, z_sw.shape[1]), z_sw)
            cnt.scatter_add_(0, y_sw, torch.ones_like(y_sw, dtype=torch.float32))
            sw_proto = sw_proto / cnt.clamp(min=1.0)[:, None]
            sw_proto = F.normalize(sw_proto, dim=1)[1:]  # [L,D]

        sim = (z_reg @ sw_proto.t()).clamp(-1, 1)  # [K,L]

        # votes
        countsA = expert_region_vote_counts(pack["region_id"], pack["yA"], K, L)
        countsB = expert_region_vote_counts(pack["region_id"], pack["yB"], K, L)
        countsC = expert_region_vote_counts(pack["region_id"], pack["yC"], K, L)

        vA = torch.from_numpy(countsA.astype(np.float32)).to(device)
        vB = torch.from_numpy(countsB.astype(np.float32)).to(device)
        vC = torch.from_numpy(countsC.astype(np.float32)).to(device)

        def norm(v):
            return v / v.sum(dim=1, keepdim=True).clamp(min=1.0)

        pA = norm(vA); pB = norm(vB); pC = norm(vC)

        style = compute_global_style_features(lab_lr)
        fracA = (vA.max(dim=1).values / vA.sum(dim=1).clamp(min=1.0))
        fracB = (vB.max(dim=1).values / vB.sum(dim=1).clamp(min=1.0))
        fracC = (vC.max(dim=1).values / vC.sum(dim=1).clamp(min=1.0))
        conf_vec = torch.stack([fracA.mean(), fracB.mean(), fracC.mean(),
                                fracA.median(), fracB.median(), fracC.median()], dim=0)
        gate_in = torch.cat([style, conf_vec], dim=0)
        w = self.gater(gate_in)  # [3]

        p = w[0] * pA + w[1] * pB + w[2] * pC  # [K,L+1]

        # foreground scores
        score_fg = p[:, 1:] + self.cfg.lambda_sim * sim  # [K,L]
        best_fg_score, best_fg = score_fg.max(dim=1)
        best_fg_label = best_fg + 1

        bg_score = p[:, 0] + self.cfg.bg_vote_bias
        best_sim = sim.max(dim=1).values if L > 0 else torch.zeros((K,), device=device)

        choose_bg = (bg_score > best_fg_score) | ((best_sim < self.cfg.sim_bg_reject) & (p[:, 0] > 0.20))
        region_label = torch.where(choose_bg, torch.zeros_like(best_fg_label), best_fg_label)

        # ROI enforcement
        if pack["roi_lr"] is not None:
            roi = torch.from_numpy(pack["roi_lr"].astype(np.uint8)).to(device)
            roi_f = roi.float()
            rid_flat = rid.view(-1)
            roi_flat = roi_f.view(-1)
            cnt = torch.bincount(rid_flat, minlength=K).float().clamp(min=1.0)
            s = torch.zeros((K,), device=device)
            s.scatter_add_(0, rid_flat, roi_flat)
            frac_in = s / cnt
            region_label = torch.where(frac_in >= 0.5, region_label, torch.zeros_like(region_label))

        label_map_lr = region_label[rid]  # [h2,w2]

        # --- extra prelim outputs (vote-only, sim-only, final) ---
        if self.cfg.write_prelim and preliminary_dir:
            ensure_dir(preliminary_dir)
            base = pack["base"]

            # vote-only argmax
            vote_only = torch.argmax(p, dim=1)  # [K] in 0..L
            vote_map = vote_only[rid].detach().cpu().numpy().astype(np.int32)

            # sim-only argmax (foreground only), with bg fallback if sim too low
            sim_best, sim_idx = sim.max(dim=1)  # [K], [K] in 0..L-1
            sim_lab = sim_idx + 1
            sim_bg = (sim_best < self.cfg.sim_bg_reject)
            sim_lab = torch.where(sim_bg, torch.zeros_like(sim_lab), sim_lab)
            sim_map = sim_lab[rid].detach().cpu().numpy().astype(np.int32)

            # final
            final_map = label_map_lr.detach().cpu().numpy().astype(np.int32)

            cmap = np.random.RandomState(123).randint(0, 255, size=(L + 1, 3), dtype=np.uint8)
            cmap[0] = np.array([0, 0, 0], dtype=np.uint8)

            cv2.imwrite(os.path.join(preliminary_dir, f"{base}_vote_argmax_lr.png"), cmap[vote_map][..., ::-1])
            cv2.imwrite(os.path.join(preliminary_dir, f"{base}_sim_argmax_lr.png"), cmap[sim_map][..., ::-1])
            cv2.imwrite(os.path.join(preliminary_dir, f"{base}_final_label_lr.png"), cmap[final_map][..., ::-1])

        return label_map_lr.detach().cpu().numpy().astype(np.int32)

    def run(
        self,
        img_dir: str,
        json_dir: str,
        sol_a: str,
        sol_b: str,
        sol_c: str,
        out_dir: str,
        preliminary_dir: str,
    ):
        ensure_dir(out_dir)
        ensure_dir(preliminary_dir)

        sol_dirs = [sol_a, sol_b, sol_c]
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))

        for img_path in tqdm(img_paths, desc="Logic+Contrast refine"):
            base = os.path.basename(img_path).replace(".tif", "")
            json_path = os.path.join(json_dir, f"{base}.json")
            if not os.path.exists(json_path):
                continue

            pack = self._prepare_one(img_path, json_path, sol_dirs)
            if pack is None:
                continue

            # prelim: write inputs/experts/regions/boundary
            self._write_prelim_experts(pack, preliminary_dir)

            # per-image tiny learning
            self._train_per_image(pack)

            # inference lowres
            label_map_lr = self._infer_per_image(pack, preliminary_dir=preliminary_dir)

            # upsample to full-res and export per-label masks
            H, W = pack["H"], pack["W"]
            label_map_full = cv2.resize(label_map_lr.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)

            with rasterio.open(img_path) as src_img:
                profile = src_img.profile
            out_profile = profile.copy()
            out_profile.update(count=1, dtype=rasterio.uint8)

            labels = pack["labels"]
            for li, lab in enumerate(labels, start=1):
                out_p = os.path.join(out_dir, f"{base}_{lab}.tif")
                m = (label_map_full == li).astype(np.uint8) * 255

                # morphological postprocessing (remove small FP noise)
                if self.cfg.morph_open:
                    m = morph_open_mask_u8(m, k=self.cfg.morph_kernel, iters=self.cfg.morph_iters)

                # geometrical postprocessing (vectorize -> simplify -> fill small holes (FN) -> rasterize back)
                if self.cfg.vectorize_simplify:
                    # IMPORTANT: use the SAME transform you will write with.
                    # If you already have out_profile with transform, use out_profile["transform"].
                    m = vector_simplify_and_fill_holes_u8(
                        m,
                        transform=out_profile["transform"],
                        simplify_tol_px=self.cfg.simplify_tol_px,
                        min_poly_area_px=self.cfg.min_poly_area_px,
                        max_hole_area_px=self.cfg.max_hole_area_px,
                    )
                
                with rasterio.open(out_p, "w", **out_profile) as dst:
                    dst.write(m, 1)

            # full-res preview image (downsample)
            if self.cfg.write_prelim and preliminary_dir:
                L = pack["L"]
                cmap = np.random.RandomState(123).randint(0, 255, size=(L + 1, 3), dtype=np.uint8)
                cmap[0] = np.array([0, 0, 0], dtype=np.uint8)
                vis_full = cmap[label_map_full.astype(np.int32)][..., ::-1]  # BGR

                # downsample for preview
                max_side = int(self.cfg.preview_max_side)
                h, w = vis_full.shape[:2]
                s = max(h, w)
                if s > max_side:
                    scale = max_side / s
                    nh = max(1, int(round(h * scale)))
                    nw = max(1, int(round(w * scale)))
                    vis_full = cv2.resize(vis_full, (nw, nh), interpolation=cv2.INTER_NEAREST)

                cv2.imwrite(os.path.join(preliminary_dir, f"{base}_final_label_full_preview.png"), vis_full)


# ============================================================
# CLI
# ============================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", type=str, required=True)
    ap.add_argument("--json_dir", type=str, required=True)
    ap.add_argument("--sol_a", type=str, required=True)
    ap.add_argument("--sol_b", type=str, required=True)
    ap.add_argument("--sol_c", type=str, required=True)

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--preliminary_dir", type=str, required=True)

    ap.add_argument("--roi_dir", type=str, default=None)

    ap.add_argument("--region_scale", type=float, default=0.25)
    ap.add_argument("--boundary_vote_thr", type=int, default=1)
    ap.add_argument("--max_propagate_iters", type=int, default=256) 
    ap.add_argument("--min_region_area", type=int, default=32) 

    ap.add_argument("--emb_dim", type=int, default=32)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--contrast_T", type=float, default=0.2)
    ap.add_argument("--lambda_region", type=float, default=0.5)
    ap.add_argument("--lambda_gate", type=float, default=0.1)
    ap.add_argument("--lambda_sim", type=float, default=1.0)
    ap.add_argument("--tau_base", type=float, default=0.4)
    ap.add_argument("--tau_scale", type=float, default=0.8)

    ap.add_argument("--bg_vote_bias", type=float, default=0.25)
    ap.add_argument("--sim_bg_reject", type=float, default=0.8)

    ap.add_argument("--max_swatch_pixels", type=int, default=4096)
    ap.add_argument("--max_conf_region_samples", type=int, default=8192)

    ap.add_argument("--read_bands", type=int, default=3)

    ap.add_argument("--disable_cache", action="store_true")
    ap.add_argument("--no_prelim", action="store_true")
    ap.add_argument("--seed", type=int, default=39)

    ap.add_argument("--preview_max_side", type=int, default=2000)

    ap.add_argument("--morph_open", action="store_true")
    ap.add_argument("--morph_kernel", type=int, default=11) 
    ap.add_argument("--morph_iters", type=int, default=1)

    ap.add_argument("--vectorize_simplify", action="store_true")
    ap.add_argument("--simplify_tol_px", type=float, default=8.0)
    ap.add_argument("--min_poly_area_px", type=float, default=32.0)
    ap.add_argument("--max_hole_area_px", type=float, default=32.0)


    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Device]", device)

    cfg = Config(
        region_scale=args.region_scale,
        boundary_vote_thr=args.boundary_vote_thr,
        max_propagate_iters=args.max_propagate_iters,
        min_region_area=args.min_region_area,

        roi_dir=args.roi_dir,

        max_swatch_pixels=args.max_swatch_pixels,
        max_conf_region_samples=args.max_conf_region_samples,

        emb_dim=args.emb_dim,
        steps=args.steps,
        lr=args.lr,
        contrast_T=args.contrast_T,
        lambda_region=args.lambda_region,
        lambda_gate=args.lambda_gate,
        lambda_sim=args.lambda_sim,
        tau_base=args.tau_base,
        tau_scale=args.tau_scale,

        bg_vote_bias=args.bg_vote_bias,
        sim_bg_reject=args.sim_bg_reject,

        enable_cache=(not args.disable_cache),
        write_prelim=(not args.no_prelim),
        seed=args.seed,
        read_bands=args.read_bands,

        preview_max_side=args.preview_max_side,

        morph_open=args.morph_open,
        morph_kernel=args.morph_kernel,
        morph_iters=args.morph_iters,

        vectorize_simplify=args.vectorize_simplify,
        simplify_tol_px=args.simplify_tol_px,
        min_poly_area_px=args.min_poly_area_px,
        max_hole_area_px=args.max_hole_area_px,
    )

    refiner = LogicPlusContrastRefiner(device=device, cfg=cfg)
    refiner.run(
        img_dir=args.img_dir,
        json_dir=args.json_dir,
        sol_a=args.sol_a,
        sol_b=args.sol_b,
        sol_c=args.sol_c,
        out_dir=args.out_dir,
        preliminary_dir=args.preliminary_dir
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
