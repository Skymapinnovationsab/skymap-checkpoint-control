#!/usr/bin/env python3
"""
Change detection between two orthophotos with polygon export.

- Aligns the two rasters to a common grid (intersection area)
- Computes color-space change (HSV) robust to lighting
- Thresholds via Otsu (or user-provided threshold)
- Cleans small speckles and vectorizes to polygons
- Exports polygons as ESRI Shapefile with area attribute

Dependencies (see requirements.txt):
- numpy, rasterio, shapely, fiona, pyproj

Example:
  python change_detect.py \
    --before before.tif --after after.tif \
    --out changes.shp --min-area-m2 100 --simplify 0.5 --smooth-buffer 0.5
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import rasterio
from rasterio import windows
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.features import shapes, sieve, geometry_mask
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union, transform as shp_transform
from shapely.geometry.base import BaseGeometry
import fiona
from pyproj import CRS, Transformer

# Safe headless plotting for preview images
import matplotlib
matplotlib.use("Agg")  # ensure no display needed
import matplotlib.pyplot as plt


@dataclass
class AlignedData:
    before: np.ndarray  # HxWxC float32 [0..1]
    after: np.ndarray   # HxWxC float32 [0..1]
    valid_mask: np.ndarray  # HxW bool
    transform: rasterio.Affine
    crs: CRS
    pixel_area: float  # in square map units (e.g., m^2 for projected)


def _select_rgb(arr: np.ndarray) -> np.ndarray:
    """Ensure array is HxWx3; pick first 3 bands if >3; expand if single band."""
    if arr.ndim != 3:
        raise ValueError("Expected 3D array (H, W, C)")
    h, w, c = arr.shape
    if c == 3:
        return arr
    if c == 4:
        return arr[:, :, :3]
    if c == 1:
        return np.repeat(arr, 3, axis=2)
    # more than 4 bands; use first 3
    return arr[:, :, :3]


def _to_float01(arr: np.ndarray) -> np.ndarray:
    if np.issubdtype(arr.dtype, np.floating):
        out = np.clip(arr, 0, 1).astype(np.float32, copy=False)
        return out
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return (arr.astype(np.float32) / float(info.max))
    # fallback
    arr = arr.astype(np.float32)
    arr -= arr.min()
    mx = arr.max()
    if mx > 0:
        arr /= mx
    return arr


def _rgb_to_hsv_np(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized RGB->HSV for arrays in [0,1]. Returns H,S,V in [0,1]."""
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    v = maxc
    delt = maxc - minc

    s = np.zeros_like(maxc)
    nz = maxc != 0
    s[nz] = delt[nz] / maxc[nz]

    h = np.zeros_like(maxc)
    mask = delt != 0
    # Avoid division by zero
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / delt[mask]
    gc[mask] = (maxc - g)[mask] / delt[mask]
    bc[mask] = (maxc - b)[mask] / delt[mask]

    cond = (r == maxc) & mask
    h[cond] = (bc - gc)[cond]
    cond = (g == maxc) & mask
    h[cond] = 2.0 + (rc - bc)[cond]
    cond = (b == maxc) & mask
    h[cond] = 4.0 + (gc - rc)[cond]
    h = (h / 6.0) % 1.0

    return h, s, v


def _otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Compute Otsu threshold on 1D array of values in [0,1]."""
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return 1.0
    vals = np.clip(vals, 0, 1)
    hist, bin_edges = np.histogram(vals, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    prob = hist / hist.sum()
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * (bin_edges[:-1] + bin_edges[1:]) * 0.5)
    mu_t = mu[-1]
    sigma_b_sq = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    idx = np.nanargmax(sigma_b_sq)
    # threshold at bin center
    t = (bin_edges[idx] + bin_edges[idx + 1]) * 0.5
    return float(t)


def read_and_align(before_path: str, after_path: str) -> AlignedData:
    with rasterio.open(before_path) as bsrc, rasterio.open(after_path) as asrc:
        target_crs = CRS.from_wkt(bsrc.crs.to_wkt()) if bsrc.crs else CRS.from_epsg(3857)

        # Compute intersection bounds in target CRS
        b_bounds = bsrc.bounds
        a_bounds_in_b = transform_bounds(asrc.crs, target_crs, *asrc.bounds, densify_pts=21)

        left = max(b_bounds.left, a_bounds_in_b[0])
        bottom = max(b_bounds.bottom, a_bounds_in_b[1])
        right = min(b_bounds.right, a_bounds_in_b[2])
        top = min(b_bounds.top, a_bounds_in_b[3])
        if not (left < right and bottom < top):
            raise ValueError("Rasters do not overlap in space; no intersection area found.")

        inter_window = windows.from_bounds(left, bottom, right, top, transform=bsrc.transform)
        inter_window = inter_window.round_offsets().round_lengths()
        # Destination grid
        out_transform = windows.transform(inter_window, bsrc.transform)
        out_h = int(inter_window.height)
        out_w = int(inter_window.width)

        # Read before in native grid window
        b_count = min(bsrc.count, 4)  # cap at 4 bands
        before = bsrc.read(indexes=list(range(1, b_count + 1)), window=inter_window)
        before = np.transpose(before, (1, 2, 0))  # HWC
        # Build valid mask from masks (non-zero across bands)
        bmask = np.ones((out_h, out_w), dtype=bool)
        for i in range(1, b_count + 1):
            bmask &= (bsrc.read_masks(i, window=inter_window) > 0)

        # Reproject after into destination grid
        a_count = min(asrc.count, 4)
        after = np.zeros((out_h, out_w, a_count), dtype=np.float32)
        for i in range(1, a_count + 1):
            dest = np.zeros((out_h, out_w), dtype=np.float32)
            reproject(
                source=rasterio.band(asrc, i),
                destination=dest,
                src_transform=asrc.transform,
                src_crs=asrc.crs,
                dst_transform=out_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
            )
            after[..., i - 1] = dest
        # Build after valid mask (nearest for mask)
        amask = np.zeros((out_h, out_w), dtype=bool)
        tmp_mask = np.zeros((out_h, out_w), dtype=np.uint8)
        reproject(
            source=asrc.read_masks(1),
            destination=tmp_mask,
            src_transform=asrc.transform,
            src_crs=asrc.crs,
            dst_transform=out_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest,
        )
        amask = tmp_mask > 0

        # Normalize and get RGB
        before = _to_float01(before)
        after = _to_float01(after)
        before = _select_rgb(before)
        after = _select_rgb(after)

        valid_mask = bmask & amask
        # Compute pixel area from transform (assumes square pixels, projected CRS -> m^2)
        # For non-square, use area of parallelogram
        a = out_transform.a
        e = out_transform.e
        pixel_area = abs(a * e)

        return AlignedData(
            before=before.astype(np.float32),
            after=after.astype(np.float32),
            valid_mask=valid_mask,
            transform=out_transform,
            crs=target_crs,
            pixel_area=pixel_area,
        )


def hsv_change_metric(b_rgb: np.ndarray, a_rgb: np.ndarray, weights=(2.0, 1.0, 0.5)) -> np.ndarray:
    """Compute per-pixel change magnitude in HSV: combine hue, saturation, value differences.
    Returns array in [0, ~sqrt(sum(w^2))]. Will be normalized to [0,1] downstream.
    """
    bh, bs, bv = _rgb_to_hsv_np(b_rgb)
    ah, as_, av = _rgb_to_hsv_np(a_rgb)
    # Circular hue distance in [0, 0.5]
    dh = np.abs(bh - ah)
    dh = np.minimum(dh, 1.0 - dh)
    ds = np.abs(bs - as_)
    dv = np.abs(bv - av)
    w_h, w_s, w_v = weights
    d = np.sqrt((w_h * dh) ** 2 + (w_s * ds) ** 2 + (w_v * dv) ** 2)
    # normalize by max possible length for given weights (hue max=0.5)
    max_len = math.sqrt((w_h * 0.5) ** 2 + (w_s * 1.0) ** 2 + (w_v * 1.0) ** 2)
    if max_len > 0:
        d = d / max_len
    return d.astype(np.float32)


def _ensure_skimage():
    try:
        import skimage  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "scikit-image is required for --ai-method fuse/gmm. Install with: pip install scikit-image scikit-learn"
        ) from e


def compute_aux_change_maps(b_rgb: np.ndarray, a_rgb: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute additional change maps: Lab deltaE, 1-SSIM, gradient diff. Returns maps in [0,1]."""
    _ensure_skimage()
    from skimage.color import rgb2lab, deltaE_cie76, rgb2gray
    from skimage.metrics import structural_similarity as ssim
    from skimage.filters import sobel

    lab_b = rgb2lab(np.clip(b_rgb, 0, 1))
    lab_a = rgb2lab(np.clip(a_rgb, 0, 1))
    de = deltaE_cie76(lab_b, lab_a).astype(np.float32)
    de_norm = np.clip(de / 100.0, 0, 1)

    try:
        ssim_map = ssim(b_rgb, a_rgb, channel_axis=2, data_range=1.0, gaussian_weights=True, win_size=11, full=True)[1]
    except TypeError:
        ssim_map = ssim(b_rgb, a_rgb, multichannel=True, data_range=1.0, gaussian_weights=True, win_size=11, full=True)[1]
    if ssim_map.ndim == 3 and ssim_map.shape[-1] == 3:
        ssim_map = ssim_map.mean(axis=2)
    ssim_map = np.clip(ssim_map.astype(np.float32), -1.0, 1.0)
    d_ssim = 1.0 - ((ssim_map + 1.0) * 0.5)

    gb = rgb2gray(np.clip(b_rgb, 0, 1)).astype(np.float32)
    ga = rgb2gray(np.clip(a_rgb, 0, 1)).astype(np.float32)
    sob_b = sobel(gb)
    sob_a = sobel(ga)
    d_grad = np.abs(sob_b - sob_a).astype(np.float32)
    q = np.quantile(d_grad, 0.999) if np.isfinite(d_grad).any() else 1.0
    d_grad = np.clip(d_grad / (q + 1e-6), 0, 1)

    return {"lab": de_norm, "ssim": d_ssim, "grad": d_grad}


def fuse_change_maps(hsv_d: np.ndarray, aux: Dict[str, np.ndarray], weights: Tuple[float, float, float, float] = (1.0, 1.0, 0.5, 1.5)) -> np.ndarray:
    wl_hsv, wl_lab, wl_grad, wl_ssim = weights
    comps = [
        (np.asarray(hsv_d, dtype=np.float32), wl_hsv),
        (np.asarray(aux["lab"], dtype=np.float32), wl_lab),
        (np.asarray(aux["grad"], dtype=np.float32), wl_grad),
        (np.asarray(aux["ssim"], dtype=np.float32), wl_ssim),
    ]
    wsum = sum(w for _, w in comps) + 1e-6
    fused = sum(arr * w for arr, w in comps) / wsum
    return np.clip(fused, 0, 1).astype(np.float32)


def gmm_change_prob(hsv_d: np.ndarray, aux: Dict[str, np.ndarray], valid_mask: np.ndarray, n_components: int = 2, subsample: int = 200000, random_state: int = 0) -> np.ndarray:
    _ensure_skimage()
    try:
        from sklearn.mixture import GaussianMixture
    except Exception as e:
        raise SystemExit("scikit-learn is required for --ai-method gmm. Install with: pip install scikit-learn scikit-image") from e

    h, w = hsv_d.shape
    f1 = hsv_d.reshape(-1, 1)
    f2 = aux["lab"].reshape(-1, 1)
    f3 = aux["grad"].reshape(-1, 1)
    f4 = aux["ssim"].reshape(-1, 1)
    X = np.concatenate([f1, f2, f3, f4], axis=1)
    vm = valid_mask.reshape(-1)
    Xv = X[vm]

    n = Xv.shape[0]
    idx = np.arange(n)
    if n > subsample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=subsample, replace=False)
    Xfit = Xv[idx]

    mu = Xfit.mean(axis=0)
    sd = Xfit.std(axis=0) + 1e-6
    Xfit_z = (Xfit - mu) / sd

    gmm = GaussianMixture(n_components=n_components, covariance_type="full", reg_covar=1e-6, random_state=random_state)
    gmm.fit(Xfit_z)

    Xv_z = (Xv - mu) / sd
    proba = gmm.predict_proba(Xv_z)
    means = gmm.means_
    score = means[:, 0] + means[:, 1] + means[:, 3]
    change_comp = int(np.argmax(score))
    p_change = proba[:, change_comp]

    P = np.zeros(h * w, dtype=np.float32)
    P[vm] = p_change.astype(np.float32)
    return P.reshape(h, w)


def mask_postprocess(binary: np.ndarray, min_pixels: int) -> np.ndarray:
    """Remove small connected components below min_pixels using sieve (8-connectivity)."""
    if min_pixels <= 1:
        return binary
    arr = binary.astype(np.uint8)
    out = sieve(arr, size=min_pixels, connectivity=8)
    return out.astype(bool)


def vectorize_polygons(mask: np.ndarray, transform, crs: CRS) -> List[BaseGeometry]:
    geoms: List[BaseGeometry] = []
    for geom, val in shapes(mask.astype(np.uint8), mask=mask, transform=transform):
        if val == 1:
            geoms.append(shape(geom))
    # Fix potential topology issues
    cleaned: List[BaseGeometry] = []
    for g in geoms:
        if isinstance(g, (Polygon, MultiPolygon)):
            g = g.buffer(0)  # fix self-intersections
        if not g.is_empty:
            cleaned.append(g)
    return cleaned


def area_in_m2(geom: BaseGeometry, crs: CRS) -> float:
    if crs.is_projected:
        # assume meters
        return float(geom.area)
    # Geographic CRS; approx by projecting to suitable UTM zone
    lonlat = CRS.from_epsg(4326)
    to_lonlat = Transformer.from_crs(crs, lonlat, always_xy=True).transform
    g_ll = shp_transform(to_lonlat, geom)
    try:
        centroid = g_ll.centroid
        lon, lat = centroid.x, centroid.y
        utm_zone = int((lon + 180) / 6) + 1
        is_northern = lat >= 0
        epsg = 32600 + utm_zone if is_northern else 32700 + utm_zone
        utm = CRS.from_epsg(epsg)
    except Exception:
        utm = CRS.from_epsg(3857)
    to_utm = Transformer.from_crs(lonlat, utm, always_xy=True).transform
    g_utm = shp_transform(to_utm, g_ll)
    return float(g_utm.area)


def dissolve_and_smooth(
    geoms: List[BaseGeometry], simplify_tolerance: float = 0.0, smooth_buffer: float = 0.0
) -> List[BaseGeometry]:
    if not geoms:
        return []
    dissolved = unary_union(geoms)
    if smooth_buffer and smooth_buffer > 0:
        dissolved = dissolved.buffer(smooth_buffer).buffer(-smooth_buffer)
    if simplify_tolerance and simplify_tolerance > 0:
        dissolved = dissolved.simplify(simplify_tolerance, preserve_topology=True)
    if isinstance(dissolved, (Polygon, MultiPolygon)):
        return [dissolved] if isinstance(dissolved, Polygon) else list(dissolved.geoms)
    return [dissolved]


def _hsv_category_fractions(hsv: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Dict[str, float]:
    h, s, v = hsv
    green = np.mean(((h >= 0.20) & (h <= 0.45) & (s >= 0.20)).astype(np.float32))
    brown = np.mean(((h >= 0.05) & (h <= 0.18) & (s >= 0.20)).astype(np.float32))
    gray = np.mean((s <= 0.20).astype(np.float32))
    blue = np.mean(((h >= 0.55) & (h <= 0.75) & (s >= 0.20)).astype(np.float32))
    return {"green": float(green), "brown": float(brown), "gray": float(gray), "blue": float(blue)}


def analyze_polygons(
    aligned: AlignedData,
    geoms: List[BaseGeometry],
    hsv_map: np.ndarray,
    change_map: np.ndarray,
    aux_maps: Optional[Dict[str, np.ndarray]] = None,
) -> List[Dict[str, Any]]:
    flat = _flatten_geoms(geoms)
    attrs: List[Dict[str, Any]] = []
    bh, bs, bv = _rgb_to_hsv_np(aligned.before)
    ah, as_, av = _rgb_to_hsv_np(aligned.after)
    for poly in flat:
        mask = geometry_mask([mapping(poly)], out_shape=hsv_map.shape, transform=aligned.transform, invert=True)
        region = mask & aligned.valid_mask
        if not np.any(region):
            attrs.append({"conf": 0.0, "comment": "No valid pixels"})
            continue
        area = area_in_m2(poly, aligned.crs)
        mean_change = float(np.mean(change_map[region]))
        mean_hsv = float(np.mean(hsv_map[region]))
        frac_b = _hsv_category_fractions((bh[region], bs[region], bv[region]))
        frac_a = _hsv_category_fractions((ah[region], as_[region], av[region]))
        de_mean = grad_mean = ssim_mean = 0.0
        if aux_maps:
            de_mean = float(np.mean(aux_maps.get("lab", np.zeros_like(hsv_map))[region]))
            grad_mean = float(np.mean(aux_maps.get("grad", np.zeros_like(hsv_map))[region]))
            ssim_mean = float(np.mean(aux_maps.get("ssim", np.zeros_like(hsv_map))[region]))

        comment_parts = []
        confidence = 0.3 + 0.7 * min(1.0, mean_change * 1.2)
        dg = frac_b["green"] - frac_a["green"]
        dbrown = frac_a["brown"] - frac_b["brown"]
        dgray = frac_a["gray"] - frac_b["gray"]
        dblue = frac_a["blue"] - frac_b["blue"]

        if frac_b["green"] > 0.3 and dg > 0.15 and (dbrown > 0.1 or dgray > 0.1):
            comment_parts.append("Vegetation clearing / soil exposure")
            confidence = max(confidence, 0.6 + 0.4 * min(1.0, (dg + max(dbrown, dgray)) / 0.6))
        if dbrown > 0.15 and de_mean > 0.1:
            comment_parts.append("Earthmoving / exposed soil")
        if dgray > 0.15 and ssim_mean > 0.2 and grad_mean > 0.1:
            comment_parts.append("New hard surface or structure")
        if dblue > 0.1:
            comment_parts.append("Increased water presence")
        if not comment_parts:
            comment_parts.append("General change detected")

        attrs.append({
            "conf": round(float(confidence), 3),
            "comment": ", ".join(comment_parts),
            "area_m2": round(float(area), 2),
            "mean_change": round(mean_change, 3),
            "mean_hsv": round(mean_hsv, 3),
            "green_before": round(frac_b["green"], 3),
            "green_after": round(frac_a["green"], 3),
            "brown_after": round(frac_a["brown"], 3),
            "gray_after": round(frac_a["gray"], 3),
            "blue_after": round(frac_a["blue"], 3),
            "delta_green": round(dg, 3),
            "delta_brown": round(dbrown, 3),
            "delta_gray": round(dgray, 3),
            "delta_blue": round(dblue, 3),
            "lab_mean": round(de_mean, 3),
            "grad_mean": round(grad_mean, 3),
            "ssim_mean": round(ssim_mean, 3),
        })
    return attrs


def _parse_output(out_path: str) -> Tuple[str, Optional[str], str]:
    """Returns (file_path, layer_name, driver). Layer is only used for GPKG.
    Supports notation: out.gpkg or out.gpkg:layername
    """
    p = out_path
    if p.lower().endswith(".shp"):
        return p, None, "ESRI Shapefile"
    if ".gpkg" in p.lower():
        if ":" in p:
            file_path, layer = p.split(":", 1)
        else:
            file_path, layer = p, "changes"
        if not file_path.lower().endswith(".gpkg"):
            raise SystemExit("When using :layer syntax, the base path must end with .gpkg")
        return file_path, layer, "GPKG"
    raise SystemExit("--out must end with .shp or .gpkg (optionally .gpkg:layer)")


def _flatten_geoms(geoms: List[BaseGeometry]) -> List[Polygon]:
    flat: List[Polygon] = []
    for g in geoms:
        if isinstance(g, Polygon):
            flat.append(g)
        elif isinstance(g, MultiPolygon):
            flat.extend(list(g.geoms))
    return flat


def save_vector(out_file: str, layer: Optional[str], driver: str, geoms: List[BaseGeometry], crs: CRS, attrs: Optional[List[Dict[str, Any]]] = None) -> None:
    schema = {"geometry": "Polygon", "properties": {"id": "int", "area_m2": "float", "conf": "float", "comment": "str:254"}}
    open_kwargs = dict(driver=driver, schema=schema, crs_wkt=crs.to_wkt())
    if driver == "GPKG" and layer:
        open_kwargs["layer"] = layer

    if not geoms:
        with fiona.open(out_file, "w", **open_kwargs):
            pass
        return

    flat = _flatten_geoms(geoms)
    with fiona.open(out_file, "w", **open_kwargs) as dst:
        for i, g in enumerate(flat, start=1):
            area = area_in_m2(g, crs)
            props = {"id": i, "area_m2": float(area), "conf": None, "comment": None}
            if attrs and i - 1 < len(attrs):
                if "conf" in attrs[i - 1]:
                    props["conf"] = float(attrs[i - 1]["conf"])
                if "comment" in attrs[i - 1]:
                    props["comment"] = str(attrs[i - 1]["comment"])[:254]
            dst.write({"geometry": mapping(g), "properties": props})


def _plot_polygons_on_axes(ax, geoms: List[BaseGeometry], transform):
    from rasterio.transform import rowcol
    for g in geoms:
        if isinstance(g, Polygon):
            rings = [g.exterior] + list(g.interiors)
        elif isinstance(g, MultiPolygon):
            rings = []
            for p in g.geoms:
                rings.append(p.exterior)
                rings.extend(list(p.interiors))
        else:
            continue
        for ring in rings:
            xs, ys = ring.coords.xy
            rows, cols = rowcol(transform, xs, ys)
            ax.plot(cols, rows, color="cyan", linewidth=1.2)


def save_preview_png(path: str, background_rgb: np.ndarray, change: np.ndarray, geoms: List[BaseGeometry], transform) -> None:
    plt.figure(figsize=(10, 10), dpi=150)
    ax = plt.gca()
    ax.imshow(np.clip(background_rgb, 0, 1))
    ax.imshow(change, cmap="inferno", alpha=0.35)
    if geoms:
        _plot_polygons_on_axes(ax, geoms, transform)
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_mask_geotiff(path: str, mask_bool: np.ndarray, transform, crs: CRS) -> None:
    import rasterio
    from rasterio.crs import CRS as RCRS
    h, w = mask_bool.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "uint8",
        "crs": RCRS.from_wkt(crs.to_wkt()),
        "transform": transform,
        "compress": "LZW",
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask_bool.astype(np.uint8), 1)
        # Set dataset mask to indicate valid pixels (optional but helpful)
        try:
            dst.write_mask((mask_bool > -1).astype(np.uint8) * 255)
        except Exception:
            pass


def save_heatmap_geotiff(path: str, change_float: np.ndarray, transform, crs: CRS) -> None:
    import rasterio
    from rasterio.crs import CRS as RCRS
    h, w = change_float.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",
        "crs": RCRS.from_wkt(crs.to_wkt()),
        "transform": transform,
        "compress": "LZW",
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(change_float.astype(np.float32), 1)
        try:
            dst.write_mask(np.ones((h, w), dtype=np.uint8) * 255)
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser(description="Detect changes between two orthophotos and export polygons (SHP/GPKG)")
    p.add_argument("--before", required=False, help="Path to 'before' orthophoto (GeoTIFF)")
    p.add_argument("--after", required=False, help="Path to 'after' orthophoto (GeoTIFF)")
    p.add_argument("--out", required=False, help="Output path: *.shp or *.gpkg[:layer]")
    p.add_argument("--weights", default="2.0,1.0,0.5", help="HSV weights w_h,w_s,w_v (default 2,1,0.5)")
    p.add_argument("--ai-method", default="hsv", choices=["hsv", "fuse", "gmm"], help="Change score method: hsv (default), fuse (HSV+Lab+SSIM+grad), gmm (unsupervised ML)")
    p.add_argument("--fuse-weights", default="1.0,1.0,0.5,1.5", help="Weights for fused maps hsv,lab,grad,ssim (used with fuse/gmm)")
    p.add_argument("--prob-threshold", type=float, default=None, help="Threshold for probability map (gmm). Default uses Otsu")
    p.add_argument("--threshold", type=float, default=None, help="Manual threshold in [0,1]. If omitted, Otsu is used")
    p.add_argument("--min-pixels", type=int, default=64, help="Remove connected components smaller than this pixel count")
    p.add_argument("--min-area-m2", type=float, default=0.0, help="Filter polygons by minimum area in m^2 (if CRS projected, else approximated)")
    p.add_argument("--simplify", type=float, default=0.0, help="Douglas-Peucker simplification tolerance in map units (e.g., meters)")
    p.add_argument("--smooth-buffer", type=float, default=0.0, help="Buffer radius in map units to smooth polygon edges (buffer then unbuffer)")
    p.add_argument("--preview", default=None, help="Optional PNG file path to save a visual overlay of changes")
    p.add_argument("--preview-background", default="after", choices=["after", "before"], help="Which image to display under the overlay (default: after)")
    p.add_argument("--mask-tif", default=None, help="Optional output GeoTIFF path for the binary change mask (1=change, 0=no change)")
    p.add_argument("--heatmap-tif", default=None, help="Optional output GeoTIFF path for the continuous change heatmap (float32 [0,1])")
    p.add_argument("--batch-csv", default=None, help="CSV with columns: before,after,out,[threshold,weights,min_pixels,min_area_m2,simplify,smooth_buffer,preview,preview_background,mask_tif,heatmap_tif]")
    p.add_argument("--batch-continue", action="store_true", help="Continue processing batch rows on error")
    p.add_argument("--report-json", default=None, help="Optional JSON path to save per-polygon analysis and comments")
    p.add_argument("--report-csv", default=None, help="Optional CSV path to save per-polygon analysis and comments")
    args = p.parse_args()

    def parse_weights(s: str) -> Tuple[float, float, float]:
        wv = tuple(float(x) for x in s.split(","))
        if len(wv) != 3:
            raise SystemExit("--weights must be three comma-separated numbers, e.g., 2,1,0.5")
        return wv  # type: ignore

    # Batch mode
    if args.batch_csv:
        import csv
        batch_path = args.batch_csv
        if not os.path.exists(batch_path):
            raise SystemExit(f"Batch CSV not found: {batch_path}")
        print(f"Running batch from {batch_path}")
        with open(batch_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row_idx = 0
            for row in reader:
                row_idx += 1
                try:
                    b = row.get("before") or row.get("BEFORE")
                    a = row.get("after") or row.get("AFTER")
                    o = row.get("out") or row.get("OUT")
                    if not b or not a or not o:
                        raise ValueError("Row missing required columns before, after, or out")
                    thr = float(row["threshold"]) if row.get("threshold") not in (None, "") else args.threshold
                    ww = parse_weights(row["weights"]) if row.get("weights") else parse_weights(args.weights)
                    mp = int(row["min_pixels"]) if row.get("min_pixels") else args.min_pixels
                    ma = float(row["min_area_m2"]) if row.get("min_area_m2") else args.min_area_m2
                    simp = float(row["simplify"]) if row.get("simplify") else args.simplify
                    sb = float(row["smooth_buffer"]) if row.get("smooth_buffer") else args.smooth_buffer
                    prev = row.get("preview") or args.preview
                    prev_bg = row.get("preview_background") or args.preview_background
                    maskp = row.get("mask_tif") or args.mask_tif
                    heatp = row.get("heatmap_tif") or None
                    print(f"[{row_idx}] Processing out={o}")

                    # Run the pipeline for this row
                    aligned = read_and_align(b, a)
                    change = hsv_change_metric(aligned.before, aligned.after, weights=ww)
                    change[~aligned.valid_mask] = 0.0
                    tthr = float(thr) if thr is not None else _otsu_threshold(change[aligned.valid_mask])
                    binary = (change >= tthr) & aligned.valid_mask
                    binary = mask_postprocess(binary, min_pixels=max(1, mp))
                    geoms = vectorize_polygons(binary, aligned.transform, aligned.crs)
                    if ma and ma > 0:
                        geoms = [g for g in geoms if area_in_m2(g, aligned.crs) >= ma]
                    geoms = dissolve_and_smooth(geoms, simplify_tolerance=simp, smooth_buffer=sb)
                    out_file, out_layer, out_driver = _parse_output(o)
                    os.makedirs(os.path.dirname(os.path.abspath(out_file)) or ".", exist_ok=True)
                    # Analysis
                    aux = {"lab": np.zeros_like(change), "grad": np.zeros_like(change), "ssim": np.zeros_like(change)}
                    try:
                        from skimage.color import rgb2lab
                        _ = rgb2lab
                        aux = compute_aux_change_maps(aligned.before, aligned.after)
                    except Exception:
                        pass
                    attrs = analyze_polygons(aligned, geoms, change, change, aux)
                    save_vector(out_file, out_layer, out_driver, geoms, aligned.crs, attrs=attrs)
                    # Optional reports per row are not produced in batch mode to keep outputs simple
                    if prev:
                        try:
                            bg = aligned.after if (prev_bg or "after") == "after" else aligned.before
                            save_preview_png(prev, bg, change, geoms, aligned.transform)
                        except Exception as e:
                            print(f"Warning: failed to save preview: {e}")
                    if maskp:
                        try:
                            save_mask_geotiff(maskp, binary, aligned.transform, aligned.crs)
                        except Exception as e:
                            print(f"Warning: failed to save mask GeoTIFF: {e}")
                    if heatp:
                        try:
                            save_heatmap_geotiff(heatp, change, aligned.transform, aligned.crs)
                        except Exception as e:
                            print(f"Warning: failed to save heatmap GeoTIFF: {e}")
                    total_area = sum(area_in_m2(g, aligned.crs) for g in geoms)
                    print(f"[{row_idx}] Done: {len(geoms)} polys | ~{total_area:.1f} m^2 | thr={tthr:.4f}")
                except Exception as e:
                    msg = f"Row {row_idx} failed: {e}"
                    if args.batch_continue:
                        print("Warning:", msg)
                        continue
                    raise SystemExit(msg)
        return

    # Single pair mode
    if not (args.before and args.after and args.out):
        raise SystemExit("--before, --after, and --out are required (or use --batch-csv)")

    w = parse_weights(args.weights)
    aligned = read_and_align(args.before, args.after)
    hsv_map = hsv_change_metric(aligned.before, aligned.after, weights=w)
    hsv_map[~aligned.valid_mask] = 0.0
    if args.ai_method == "hsv":
        change = hsv_map
        thr = float(args.threshold) if args.threshold is not None else _otsu_threshold(change[aligned.valid_mask])
        binary = (change >= thr) & aligned.valid_mask
    else:
        aux = compute_aux_change_maps(aligned.before, aligned.after)
        fw = tuple(float(x) for x in args.fuse_weights.split(","))
        if len(fw) != 4:
            raise SystemExit("--fuse-weights must have 4 comma-separated numbers for hsv,lab,grad,ssim")
        if args.ai_method == "fuse":
            change = fuse_change_maps(hsv_map, aux, weights=fw)
            thr = float(args.threshold) if args.threshold is not None else _otsu_threshold(change[aligned.valid_mask])
            binary = (change >= thr) & aligned.valid_mask
        else:  # gmm
            change = gmm_change_prob(hsv_map, aux, aligned.valid_mask)
            if args.prob_threshold is not None:
                thr = float(args.prob_threshold)
            else:
                thr = _otsu_threshold(change[aligned.valid_mask])
            binary = (change >= thr) & aligned.valid_mask
    binary = mask_postprocess(binary, min_pixels=max(1, args.min_pixels))
    geoms = vectorize_polygons(binary, aligned.transform, aligned.crs)
    if args.min_area_m2 and args.min_area_m2 > 0:
        geoms = [g for g in geoms if area_in_m2(g, aligned.crs) >= args.min_area_m2]
    geoms = dissolve_and_smooth(geoms, simplify_tolerance=args.simplify, smooth_buffer=args.smooth_buffer)

    out_file, out_layer, out_driver = _parse_output(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out_file)) or ".", exist_ok=True)
    # Analyze and save attributes with comments
    aux_for_report = None
    try:
        aux_for_report = compute_aux_change_maps(aligned.before, aligned.after)
    except Exception:
        aux_for_report = None
    attrs = analyze_polygons(aligned, geoms, hsv_map, change, aux_for_report)
    save_vector(out_file, out_layer, out_driver, geoms, aligned.crs, attrs=attrs)

    if args.preview:
        try:
            bg = aligned.after if args.preview_background == "after" else aligned.before
            save_preview_png(args.preview, bg, change, geoms, aligned.transform)
        except Exception as e:
            print(f"Warning: failed to save preview: {e}")
    if args.mask_tif:
        try:
            save_mask_geotiff(args.mask_tif, binary, aligned.transform, aligned.crs)
        except Exception as e:
            print(f"Warning: failed to save mask GeoTIFF: {e}")
    if args.heatmap_tif:
        try:
            save_heatmap_geotiff(args.heatmap_tif, change, aligned.transform, aligned.crs)
        except Exception as e:
            print(f"Warning: failed to save heatmap GeoTIFF: {e}")

    # Optional reports (single mode)
    if args.report_json or args.report_csv:
        try:
            rows = []
            for i, (g, a) in enumerate(zip(_flatten_geoms(geoms), attrs), start=1):
                rec = {"id": i, **a}
                rows.append(rec)
            if args.report_json:
                import json
                os.makedirs(os.path.dirname(os.path.abspath(args.report_json)) or ".", exist_ok=True)
                with open(args.report_json, "w", encoding="utf-8") as f:
                    json.dump({"summary": {"count": len(rows)}, "features": rows}, f, ensure_ascii=False, indent=2)
            if args.report_csv:
                import csv
                os.makedirs(os.path.dirname(os.path.abspath(args.report_csv)) or ".", exist_ok=True)
                if rows:
                    with open(args.report_csv, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                        writer.writeheader()
                        writer.writerows(rows)
        except Exception as e:
            print(f"Warning: failed to write report(s): {e}")

    total_area = sum(area_in_m2(g, aligned.crs) for g in geoms)
    print(f"Change polygons: {len(geoms)} | Total area ~ {total_area:.1f} m^2 | threshold={thr:.4f}")


if __name__ == "__main__":
    main()
