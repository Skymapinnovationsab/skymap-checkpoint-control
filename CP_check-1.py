#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CP vs point cloud check (auto NEH/ENH), saves outputs next to the point cloud.
# Includes TOTAL row in CSV and optional export of used pointcloud points per CP.
# Author: ChatGPT for Jon Bengtsson (SkyMap)

import argparse
import os
import math
import re
import numpy as np
import pandas as pd

# ---------- Optional imports ----------
HAVE_LASPY = False
try:
    import laspy
    HAVE_LASPY = True
except Exception:
    pass

HAVE_SCIPY = False
try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except Exception:
    pass

# ---------- Helpers ----------

def _normalize_colname(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9åäö]", "", s)
    return s

def _looks_like_headerless(df):
    def _numlike(s):
        s = str(s).strip().replace("'", "")
        return bool(re.fullmatch(r"[+-]?\d+([.,]\d+)?", s))
    try:
        return all(_numlike(c) for c in df.columns)
    except Exception:
        return False

# ---------- Loaders ----------

def load_point_cloud(path, csv_sep=None, csv_decimal="."):
    ext = os.path.splitext(path.lower())[1]
    if ext in ['.las', '.laz']:
        if not HAVE_LASPY:
            raise RuntimeError("laspy not installed. Run: pip install 'laspy[lazrs]'")
        with laspy.open(path) as fh:
            las = fh.read()
            x, y, z = las.x, las.y, las.z
        return np.column_stack([x, y, z])
    elif ext in ['.csv', '.tsv', '.txt']:
        if csv_sep is None:
            df = pd.read_csv(path, sep=None, engine="python", decimal=csv_decimal)
        else:
            df = pd.read_csv(path, sep=csv_sep, decimal=csv_decimal)
        if _looks_like_headerless(df):
            df = pd.read_csv(path, sep=csv_sep or ";", decimal=csv_decimal,
                             header=None, names=["x","y","z"])
        cols = [c.lower() for c in df.columns]
        def pick(names):
            for nm in names:
                if nm in cols:
                    return df.columns[cols.index(nm)]
            return None
        xcol = pick(["x","e","east","easting","ost","öst"])
        ycol = pick(["y","n","north","northing","nord"])
        zcol = pick(["z","h","height","elev","elevation","hojd","höjd"])
        if not (xcol and ycol and zcol):
            raise RuntimeError(f"Could not detect x,y,z in {list(df.columns)}")
        return df[[xcol,ycol,zcol]].to_numpy(dtype=float)
    else:
        raise RuntimeError(f"Unsupported point cloud format: {ext}")

def load_cps_variants(path, csv_sep=";", csv_decimal=","):
    """Return ids and two CP arrays:
       cps_ne: X<-N, Y<-E, Z<-H   (NEH)
       cps_en: X<-E, Y<-N, Z<-H   (ENH)
       If headerless, assumes columns: id;A;B;H where A,B are E/N.
    """
    df_probe = pd.read_csv(path, sep=csv_sep, decimal=csv_decimal)
    headerless = _looks_like_headerless(df_probe)
    if headerless:
        df = pd.read_csv(path, sep=csv_sep, decimal=csv_decimal,
                         header=None, names=["id","A","B","H"])
        ids = df["id"].astype(str).to_list()
        H = df["H"].to_numpy(float)
        cps_ne = np.column_stack([df["B"].to_numpy(float), df["A"].to_numpy(float), H])  # X<-N(B), Y<-E(A)
        cps_en = np.column_stack([df["A"].to_numpy(float), df["B"].to_numpy(float), H])  # X<-E(A), Y<-N(B)
        return ids, cps_ne, cps_en
    else:
        # assume id, E, N, H (vi testar ändå båda mappingarna)
        ids = df_probe.iloc[:,0].astype(str).to_list()
        A = df_probe.iloc[:,1].to_numpy(float)  # E
        B = df_probe.iloc[:,2].to_numpy(float)  # N
        H = df_probe.iloc[:,3].to_numpy(float)
        cps_ne = np.column_stack([B,A,H])
        cps_en = np.column_stack([A,B,H])
        return ids, cps_ne, cps_en

# ---------- Math ----------

def mad_based_filter(z, alpha=3.0):
    if len(z) == 0:
        return np.array([], dtype=bool)
    med = np.median(z)
    mad = np.median(np.abs(z - med))
    if mad == 0:
        return np.ones_like(z, dtype=bool)
    zscore = 1.4826 * np.abs(z - med) / mad
    return zscore <= alpha

def fit_plane_ls(points):
    if points.shape[0] < 3:
        return None, None
    A = np.column_stack([points[:,:2], np.ones(len(points))])
    try:
        coef, *_ = np.linalg.lstsq(A, points[:,2], rcond=None)
    except np.linalg.LinAlgError:
        return None, None
    return (coef[0], coef[1], coef[2]), points[:,2] - A @ coef

def ransac_plane(points, thresh=0.03, iters=200, min_inliers=30, random_state=42):
    if points.shape[0] < 3:
        return None, None
    rng = np.random.default_rng(random_state)
    best_inliers = None
    best_count = 0
    def plane_from_3(p0,p1,p2):
        v1,v2 = p1-p0, p2-p0
        normal = np.cross(v1,v2)
        if np.linalg.norm(normal) < 1e-12:
            return None
        a,b,c = normal; d = -np.dot(normal, p0)
        return a,b,c,d
    def dist(p,plane):
        a,b,c,d = plane
        return np.abs(a*p[:,0]+b*p[:,1]+c*p[:,2]+d)/math.sqrt(a*a+b*b+c*c)
    for _ in range(iters):
        tri = points[rng.choice(points.shape[0], 3, replace=False)]
        plane = plane_from_3(tri[0], tri[1], tri[2])
        if plane is None:
            continue
        inliers = dist(points, plane) <= thresh
        count = int(inliers.sum())
        if count > best_count and count >= min_inliers:
            best_count = count
            best_inliers = inliers
    if best_inliers is None:
        coef, _ = fit_plane_ls(points)
        if coef is None:
            return None, None
        a,b,c = coef
        return (a,b,-1.0,c), np.ones(points.shape[0], dtype=bool)
    coef, _ = fit_plane_ls(points[best_inliers])
    if coef is None:
        return None, None
    a,b,c = coef
    return (a,b,-1.0,c), best_inliers

def eval_plane_at_xy(plane4, X, Y):
    a,b,c,d = plane4
    return (-(a*X + b*Y + d))/c if abs(c + 1.0) > 1e-6 else a*X + b*Y + d

def plane_point_distance(plane4, pts):
    a,b,c,d = plane4
    return np.abs(a*pts[:,0]+b*pts[:,1]+c*pts[:,2]+d)/math.sqrt(a*a+b*b+c*c)

def build_kdtree(points_xy):
    if not HAVE_SCIPY:
        raise RuntimeError("scipy not installed. Run: pip install scipy")
    return cKDTree(points_xy)

def query_neighbors(tree, pts_xy, X, Y, radius=None, knn=None):
    if radius is not None:
        idxs = tree.query_ball_point([X, Y], r=radius)
        return np.array(idxs, dtype=int)
    dists, idxs = tree.query([X, Y], k=knn)
    if np.isscalar(idxs):
        return np.array([int(idxs)], dtype=int)
    return np.array(idxs, dtype=int)

def compute_metrics(dz, dp):
    arr = np.array(dz, float)
    arrp = np.array(dp, float)
    metrics = {
        'n': int(np.isfinite(arr).sum()),
        'bias_Z': float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan,
        'rmse_Z': float(np.sqrt(np.nanmean(arr**2))) if np.isfinite(arr).any() else np.nan,
        'P68_absZ': float(np.nanpercentile(np.abs(arr), 68)) if np.isfinite(arr).any() else np.nan,
        'P95_absZ': float(np.nanpercentile(np.abs(arr), 95)) if np.isfinite(arr).any() else np.nan,
    }
    if np.isfinite(arrp).any():
        metrics.update({
            'rmse_3D_perp': float(np.sqrt(np.nanmean(arrp**2))),
            'P68_perp': float(np.nanpercentile(arrp, 68)),
            'P95_perp': float(np.nanpercentile(arrp, 95)),
        })
    metrics['ok_count'] = metrics['n']
    return metrics

# ---------- Evaluation (one mapping pass) ----------

def evaluate_cps(pts, ids, cps, *, radius=None, knn=None, mad_alpha=6.0,
                 ransac_thresh=0.08, ransac_iters=400, min_inliers=5, min_neighbors=5,
                 export_stage=None):
    """
    export_stage: None | 'neighbors' | 'after_mad' | 'inliers' | 'all'
    Returns: (results, metrics, export_rows)
    """
    tree = build_kdtree(pts[:, :2])
    results = []
    dz = []
    dp = []
    export_rows = []  # rows: (cp_id, stage, Xp, Yp, Zp, dZ_to_plane, d_perp_to_plane, dist_xy_to_cp)

    for cid, cp in zip(ids, cps):
        X, Y, Z = map(float, cp)

        idxs = query_neighbors(tree, pts[:, :2], X, Y, radius=radius, knn=knn)
        n_neighbors = int(idxs.size)
        if n_neighbors == 0:
            results.append((cid, X, Y, Z, np.nan, np.nan, n_neighbors, 0, 0, 'no_neighbors'))
            dz.append(np.nan); dp.append(np.nan); continue

        neigh = pts[idxs].copy()

        # Optional export: neighbors
        if export_stage in ('neighbors', 'all'):
            for p in neigh:
                dist_xy = math.hypot(p[0]-X, p[1]-Y)
                export_rows.append((cid, 'neighbors', float(p[0]), float(p[1]), float(p[2]), np.nan, np.nan, dist_xy))

        mask = mad_based_filter(neigh[:, 2], alpha=mad_alpha)
        neigh_after = neigh[mask]
        n_after_mad = int(neigh_after.shape[0])

        if export_stage in ('after_mad', 'all'):
            for p in neigh_after:
                dist_xy = math.hypot(p[0]-X, p[1]-Y)
                export_rows.append((cid, 'after_mad', float(p[0]), float(p[1]), float(p[2]), np.nan, np.nan, dist_xy))

        if n_after_mad < min_neighbors:
            results.append((cid, X, Y, Z, np.nan, np.nan, n_neighbors, n_after_mad, 0, 'too_few_after_MAD'))
            dz.append(np.nan); dp.append(np.nan); continue

        plane4, inliers = ransac_plane(neigh_after, thresh=ransac_thresh, iters=ransac_iters, min_inliers=min_inliers)
        if plane4 is None:
            results.append((cid, X, Y, Z, np.nan, np.nan, n_neighbors, n_after_mad, 0, 'ransac_fail'))
            dz.append(np.nan); dp.append(np.nan); continue

        # Evaluate
        Zhat = eval_plane_at_xy(plane4, X, Y)
        dZ = Z - Zhat
        dperp = plane_point_distance(plane4, np.array([[X, Y, Z]])).item()
        n_inliers = int(inliers.sum())
        results.append((cid, X, Y, Z, float(dZ), float(dperp), n_neighbors, n_after_mad, n_inliers, 'ok'))
        dz.append(float(dZ)); dp.append(float(dperp))

        # Optional export: inliers
        if export_stage in ('inliers', 'all'):
            inlier_pts = neigh_after[inliers]
            a, b, c, d = plane4
            for p in inlier_pts:
                # residuals to the plane
                zhat_p = eval_plane_at_xy(plane4, p[0], p[1])
                dZp = p[2] - zhat_p
                dperp_p = plane_point_distance(plane4, np.array([[p[0], p[1], p[2]]])).item()
                dist_xy = math.hypot(p[0]-X, p[1]-Y)
                export_rows.append((cid, 'inliers', float(p[0]), float(p[1]), float(p[2]), float(dZp), float(dperp_p), dist_xy))

    return results, compute_metrics(dz, dp), export_rows

# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="CP vs point cloud check (auto NEH/ENH) with optional point export.")
    parser.add_argument('--points',
        default='/Users/jon/SkyMap Dropbox/Teammapp som tillhör SkyMap/SkyMap Innovations Projekt (1)/Jon projekt/Consto_support/Ringnes_2025-08-04 Adkomstvei_v2_pointcloud.laz')
    parser.add_argument('--cps',
        default='/Users/jon/SkyMap Dropbox/Teammapp som tillhör SkyMap/SkyMap Innovations Projekt (1)/Jon projekt/Consto_support/Ringnes_2025-08-04 Adkomstvei.csv')
    parser.add_argument('--cps-sep', default=';')
    parser.add_argument('--decimal', default=',')
    # Neighborhood (default: knn 200)
    parser.add_argument('--radius', type=float, default=None)
    parser.add_argument('--knn', type=int, default=200)
    # Robust/fit defaults (loosened)
    parser.add_argument('--mad_alpha', type=float, default=6.0)
    parser.add_argument('--ransac_thresh', type=float, default=0.08)
    parser.add_argument('--ransac_iters', type=int, default=400)
    parser.add_argument('--min_inliers', type=int, default=5)
    parser.add_argument('--min_neighbors', type=int, default=5)
    # Export options
    parser.add_argument('--export-points', action='store_true', help="Write cp_check_points.csv with used pointcloud points.")
    parser.add_argument('--export-stage', choices=['neighbors','after_mad','inliers','all'], default='inliers',
                        help="Which points to export (default: inliers).")
    args = parser.parse_args()

    # Load data
    pts = load_point_cloud(args.points, csv_decimal=args.decimal)
    ids, cps_ne, cps_en = load_cps_variants(args.cps, csv_sep=args.cps_sep, csv_decimal=args.decimal)

    # Params to evaluation
    eval_params = dict(
        radius=args.radius,
        knn=args.knn,
        mad_alpha=args.mad_alpha,
        ransac_thresh=args.ransac_thresh,
        ransac_iters=args.ransac_iters,
        min_inliers=args.min_inliers,
        min_neighbors=args.min_neighbors,
        export_stage=(args.export_stage if args.export_points else None),
    )

    # Evaluate NEH (preferred), then ENH
    res_ne, met_ne, exp_ne = evaluate_cps(pts, ids, cps_ne, **eval_params)
    res_en, met_en, exp_en = evaluate_cps(pts, ids, cps_en, **eval_params)

    chosen = 'NEH' if met_ne['ok_count'] >= met_en['ok_count'] else 'ENH'
    if chosen == 'NEH':
        results, metrics, export_rows = res_ne, met_ne, exp_ne
    else:
        results, metrics, export_rows = res_en, met_en, exp_en

    # Build per-CP dataframe (with counts)
    out_df = pd.DataFrame(
        results,
        columns=['id','X','Y','Z_CP','dZ','d_perp','n_neighbors','n_after_mad','n_inliers','status']
    )

    # Append TOTAL row
    total_row = {
        'id': 'TOTAL',
        'X': np.nan,
        'Y': np.nan,
        'Z_CP': np.nan,
        'dZ': metrics.get('bias_Z', np.nan),
        'd_perp': metrics.get('rmse_3D_perp', np.nan),
        'n_neighbors': metrics.get('n', np.nan),
        'n_after_mad': np.nan,
        'n_inliers': metrics.get('n', np.nan),
        'status': f"rmse_Z={metrics.get('rmse_Z',np.nan):.3f} P95_absZ={metrics.get('P95_absZ',np.nan):.3f}"
    }
    out_df = pd.concat([out_df, pd.DataFrame([total_row])], ignore_index=True)

    # Always save outputs next to point cloud
    points_dir = os.path.dirname(os.path.abspath(args.points))
    out_csv_path = os.path.join(points_dir, "cp_check_results.csv")
    summary_path = os.path.join(points_dir, "cp_check_results_summary.txt")
    out_df.to_csv(out_csv_path, index=False)

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"CHECK SUMMARY (chosen mapping: {chosen})\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    # Optional export of used pointcloud points
    if args.export_points and export_rows:
        export_df = pd.DataFrame(export_rows, columns=[
            'cp_id','stage','Xp','Yp','Zp','dZ_to_plane','d_perp_to_plane','dist_xy_to_cp'
        ])
        export_path = os.path.join(points_dir, "cp_check_points.csv")
        export_df.to_csv(export_path, index=False)
        extra = f"\nWrote exported points to: {export_path}"
    else:
        extra = ""

    # Console output
    print("=== SUMMARY ===")
    print(f"Chosen CP mapping: {chosen}")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"\nWrote per-CP results (with TOTAL row) to: {out_csv_path}")
    print(f"Wrote summary to: {summary_path}{extra}")

if __name__ == '__main__':
    main()