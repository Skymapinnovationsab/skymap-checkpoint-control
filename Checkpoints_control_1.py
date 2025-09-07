#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CP vs point cloud check (auto NEH/ENH), saves outputs next to the point cloud.
# Includes export of used points, selection filters, ADAPTIVE RADIUS and CENTROID CHECK.
# Author: ChatGPT for Jon Bengtsson (SkyMap)

import argparse, os, math, re
import numpy as np, pandas as pd

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
def _looks_like_headerless(df):
    def _numlike(s):
        s = str(s).strip().replace("'", "")
        return bool(re.fullmatch(r"[+-]?\d+([.,]\d+)?", s))
    try:
        return all(_numlike(c) for c in df.columns)
    except Exception:
        return False

# ---------- Loaders ----------
def load_point_cloud(path, csv_sep=None, csv_decimal=".", las_keep_class=None, las_keep_returns=None):
    ext = os.path.splitext(path.lower())[1]
    if ext in ['.las', '.laz']:
        if not HAVE_LASPY:
            raise RuntimeError("laspy not installed. Run: pip install 'laspy[lazrs]'")
        with laspy.open(path) as fh:
            las = fh.read()
        x, y, z = las.x, las.y, las.z
        mask = np.ones_like(x, dtype=bool)
        if las_keep_class is not None and hasattr(las, "classification"):
            keep = {int(v) for v in str(las_keep_class).split(',') if v.strip()}
            mask &= np.isin(las.classification, list(keep))
        if las_keep_returns is not None and hasattr(las, "return_number"):
            keep = {int(v) for v in str(las_keep_returns).split(',') if v.strip()}
            mask &= np.isin(las.return_number, list(keep))
        x, y, z = x[mask], y[mask], z[mask]
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
    """Robustly parse CPS file and return ids plus two CP arrays (NEH and ENH).
    Expects columns: id;E;N;H (header may be absent)."""
    # Sanitize separator in case quotes were provided via CLI
    if isinstance(csv_sep, str):
        csv_sep = csv_sep.strip().strip("'\"")
    # Try a few separators; try both with and without headers
    tried = []
    seps_to_try = [csv_sep, None, ';', ',', '\t', r'\s+']
    df = None
    for sp in seps_to_try:
        if sp in tried:
            continue
        tried.append(sp)
        try:
            # First try with header detection
            df_try = pd.read_csv(path, sep=sp, decimal=csv_decimal, engine="python")
            if df_try.shape[1] >= 4:
                df = df_try
                break
        except Exception:
            pass
        try:
            # If that fails, try without header
            df_try = pd.read_csv(path, sep=sp, decimal=csv_decimal, header=None, engine="python")
            if df_try.shape[1] >= 4:
                df = df_try
                break
        except Exception:
            continue
    if df is None or df.shape[1] < 4:
        raise RuntimeError(f"CPS parse error: <4 columns. Tried seps={tried}. Got shape {None if df is None else df.shape}")
    df = df.iloc[:, :4]
    df.columns = ["id", "A", "B", "H"]
    
    # Handle case where first row might be headers
    try:
        # Try to convert first row to float - if it fails, it's likely headers
        test_float = df["A"].iloc[0]
        float(test_float)
        # If successful, use all data
        ids = df["id"].astype(str).to_list()
        A = df["A"].to_numpy(float)  # E
        B = df["B"].to_numpy(float)  # N
        H = df["H"].to_numpy(float)
    except (ValueError, TypeError):
        # First row is likely headers, skip it
        df = df.iloc[1:]
        ids = df["id"].astype(str).to_list()
        A = df["A"].to_numpy(float)  # E
        B = df["B"].to_numpy(float)  # N
        H = df["H"].to_numpy(float)
    cps_ne = np.column_stack([B, A, H])  # X<-N(B), Y<-E(A)
    cps_en = np.column_stack([A, B, H])  # X<-E(A), Y<-N(B)
    return ids, cps_ne, cps_en

# ---------- Math ----------
def mad_based_filter(z, alpha=3.0):
    if len(z) == 0: return np.array([], dtype=bool)
    med = np.median(z)
    mad = np.median(np.abs(z - med))
    if mad == 0:
        # If MAD is zero, keep values equal to the median and drop deviations
        return np.abs(z - med) == 0
    zscore = 1.4826 * np.abs(z - med) / mad
    return zscore <= alpha

def fit_plane_ls(points):
    if points.shape[0] < 3: return None, None
    A = np.column_stack([points[:,:2], np.ones(len(points))])
    try:
        coef, *_ = np.linalg.lstsq(A, points[:,2], rcond=None)
    except np.linalg.LinAlgError:
        return None, None
    return (coef[0], coef[1], coef[2]), points[:,2] - A @ coef

def ransac_plane(points, thresh=0.03, iters=200, min_inliers=30, random_state=42):
    if points.shape[0] < 3: return None, None
    rng = np.random.default_rng(random_state)
    best_inliers = None; best_count = 0
    def plane_from_3(p0,p1,p2):
        v1,v2 = p1-p0, p2-p0
        normal = np.cross(v1,v2)
        if np.linalg.norm(normal) < 1e-12: return None
        a,b,c = normal; d = -np.dot(normal,p0)
        return a,b,c,d
    def dist(p,plane):
        a,b,c,d = plane
        return np.abs(a*p[:,0]+b*p[:,1]+c*p[:,2]+d)/math.sqrt(a*a+b*b+c*c)
    for _ in range(iters):
        tri = points[rng.choice(points.shape[0], 3, replace=False)]
        plane = plane_from_3(tri[0], tri[1], tri[2])
        if plane is None: continue
        inliers = dist(points, plane) <= thresh
        count = int(inliers.sum())
        if count > best_count and count >= min_inliers:
            best_count = count; best_inliers = inliers
    if best_inliers is None:
        coef,_ = fit_plane_ls(points)
        if coef is None: return None,None
        a,b,c = coef
        return (a,b,-1.0,c), np.ones(points.shape[0], dtype=bool)
    coef,_ = fit_plane_ls(points[best_inliers])
    if coef is None: return None,None
    a,b,c = coef
    return (a,b,-1.0,c), best_inliers

def eval_plane_at_xy(plane4,X,Y):
    a,b,c,d = plane4
    return (-(a*X + b*Y + d))/c if abs(c + 1.0) > 1e-6 else a*X + b*Y + d

def plane_point_distance(plane4, pts):
    a,b,c,d = plane4
    return np.abs(a*pts[:,0] + b*pts[:,1] + c*pts[:,2] + d) / math.sqrt(a*a + b*b + c*c)

def build_kdtree(points_xy):
    if not HAVE_SCIPY:
        raise RuntimeError("scipy not installed. Run: pip install scipy")
    return cKDTree(points_xy)

def query_neighbors(tree, pts_xy, X, Y, radius=None, knn=None):
    if radius is not None:
        idxs = tree.query_ball_point([X, Y], r=radius)
        return np.array(idxs, dtype=int)
    _, idxs = tree.query([X, Y], k=knn)
    return np.array([int(idxs)], dtype=int) if np.isscalar(idxs) else np.array(idxs, dtype=int)

def compute_metrics(dz, dp):
    arr = np.array(dz, float); arrp = np.array(dp, float)
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

# ---------- Evaluation with ADAPTIVE RADIUS & CENTROID ----------
def evaluate_cps(pts, ids, cps, *, radius=None, knn=None, mad_alpha=6.0,
                 ransac_thresh=0.08, ransac_iters=400, min_inliers=5, min_neighbors=5,
                 export_stage=None, xy_maxdist=None, z_window=None,
                 adaptive_radius=False, adaptive_radius_start=0.15, adaptive_radius_step=0.05, adaptive_radius_max=0.40,
                 centroid_max_offset=None):
    """export_stage: None|'neighbors'|'after_mad'|'inliers'|'all'"""
    tree = build_kdtree(pts[:, :2])
    results = []; dz = []; dp = []; export_rows = []

    radius_mode = radius is not None or adaptive_radius

    for cid, cp in zip(ids, cps):
        X, Y, Z = map(float, cp)

        # --- ADAPTIVE RADIUS NEIGHBOR PICKING (or fixed radius / kNN) ---
        current_radius = radius
        neigh = None; n_neighbors = 0

        def pick_neighbors(r=None):
            nonlocal n_neighbors
            if radius_mode:
                idxs = query_neighbors(tree, pts[:, :2], X, Y, radius=r, knn=None)
            else:
                idxs = query_neighbors(tree, pts[:, :2], X, Y, radius=None, knn=knn)
            n_neighbors = int(idxs.size)
            if n_neighbors == 0: 
                return None
            cand = pts[idxs].copy()

            # Hard XY and Z constraints applied BEFORE any robust filtering
            if xy_maxdist is not None:
                dxy = np.hypot(cand[:,0]-X, cand[:,1]-Y)
                cand = cand[dxy <= xy_maxdist]
            if z_window is not None:
                dzw = np.abs(cand[:,2] - Z)
                cand = cand[dzw <= z_window]

            return cand if cand.shape[0] > 0 else None

        if adaptive_radius:
            # Start at start-radius and increase until enough neighbors after prefilters
            r = adaptive_radius_start
            while r <= adaptive_radius_max:
                neigh = pick_neighbors(r=r)
                if neigh is not None and neigh.shape[0] >= min_neighbors:
                    current_radius = r
                    break
                r = round(r + adaptive_radius_step, 6)
            if neigh is None:
                # fall back to last attempt (still None) -> no neighbors
                results.append((cid,X,Y,Z,np.nan,np.nan,n_neighbors,0,0,'no_neighbors',np.nan,np.nan,np.nan,current_radius))
                dz.append(np.nan); dp.append(np.nan); continue
        else:
            neigh = pick_neighbors(r=current_radius)
            if neigh is None:
                results.append((cid,X,Y,Z,np.nan,np.nan,n_neighbors,0,0,'no_neighbors',np.nan,np.nan,np.nan,current_radius))
                dz.append(np.nan); dp.append(np.nan); continue

        # Optional export: neighbors (after prefilters)
        if export_stage in ('neighbors','all'):
            for p in neigh:
                dist_xy = math.hypot(p[0]-X, p[1]-Y)
                export_rows.append((cid,'neighbors',float(p[0]),float(p[1]),float(p[2]),np.nan,np.nan,dist_xy))

        # --- Robust filtering (MAD on Z) ---
        mask = mad_based_filter(neigh[:,2], alpha=mad_alpha)
        neigh_after = neigh[mask]
        n_after_mad = int(neigh_after.shape[0])
        if export_stage in ('after_mad','all'):
            for p in neigh_after:
                dist_xy = math.hypot(p[0]-X, p[1]-Y)
                export_rows.append((cid,'after_mad',float(p[0]),float(p[1]),float(p[2]),np.nan,np.nan,dist_xy))

        if n_after_mad < min_neighbors:
            results.append((cid,X,Y,Z,np.nan,np.nan,n_neighbors,n_after_mad,0,'too_few_after_MAD',np.nan,np.nan,np.nan,current_radius))
            dz.append(np.nan); dp.append(np.nan); continue

        # --- Plane fit & evaluation ---
        plane4, inliers = ransac_plane(neigh_after, thresh=ransac_thresh, iters=ransac_iters, min_inliers=min_inliers)
        if plane4 is None:
            results.append((cid,X,Y,Z,np.nan,np.nan,n_neighbors,n_after_mad,0,'ransac_fail',np.nan,np.nan,np.nan,current_radius))
            dz.append(np.nan); dp.append(np.nan); continue

        Zhat = eval_plane_at_xy(plane4, X, Y)
        dZ_cp = Z - Zhat
        dperp_cp = plane_point_distance(plane4, np.array([[X, Y, Z]])).item()
        inlier_pts = neigh_after[inliers]
        n_inliers = int(inliers.sum())

        # Centroid check
        cx = float(np.mean(inlier_pts[:,0])) if n_inliers>0 else np.nan
        cy = float(np.mean(inlier_pts[:,1])) if n_inliers>0 else np.nan
        centroid_dx = cx - X if n_inliers>0 else np.nan
        centroid_dy = cy - Y if n_inliers>0 else np.nan
        centroid_dxy = float(math.hypot(centroid_dx, centroid_dy)) if n_inliers>0 else np.nan

        status = 'ok'
        if centroid_max_offset is not None and n_inliers>0 and centroid_dxy > centroid_max_offset:
            status = 'centroid_far'

        results.append((cid,X,Y,Z,float(dZ_cp),float(dperp_cp),
                        n_neighbors,n_after_mad,n_inliers,status,
                        float(centroid_dx) if n_inliers>0 else np.nan,
                        float(centroid_dy) if n_inliers>0 else np.nan,
                        float(centroid_dxy) if n_inliers>0 else np.nan,
                        current_radius))

        dz.append(float(dZ_cp)); dp.append(float(dperp_cp))

        # Optional export: inliers
        if export_stage in ('inliers','all'):
            # residuals per inlier
            for p in inlier_pts:
                zhat_p = eval_plane_at_xy(plane4, p[0], p[1])
                dZp = p[2] - zhat_p
                dperp_p = plane_point_distance(plane4, np.array([[p[0],p[1],p[2]]])).item()
                dist_xy = math.hypot(p[0]-X, p[1]-Y)
                export_rows.append((cid,'inliers',float(p[0]),float(p[1]),float(p[2]),float(dZp),float(dperp_p),dist_xy))

    return results, compute_metrics(dz, dp), export_rows

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="CP vs point cloud check (auto NEH/ENH) with export, filters, adaptive radius, centroid check.")
    # Default paths (Växjö-projektet)
    ap.add_argument('--points', default='/Users/jon/SkyMap Dropbox/Teammapp som tillhör SkyMap/SkyMap Innovations Projekt (1)/Jon projekt/Vaxjo simhall/Kontroll_python/AO11282.2100 Växjö Nya Simhall_Vaxjo Simhall_v2 - 2025-09-02 1451_pointcloud.laz')
    ap.add_argument('--cps',    default='/Users/jon/SkyMap Dropbox/Teammapp som tillhör SkyMap/SkyMap Innovations Projekt (1)/Jon projekt/Vaxjo simhall/Kontroll_python/AO11282.2100 Växjö Nya Simhall_Vaxjo Simhall_v2 - 2025-09-01 1343_SWEREF991500.csv')
    ap.add_argument('--cps-sep', default=';')
    ap.add_argument('--decimal', default=',')
    # Neighborhood
    ap.add_argument('--radius', type=float, default=None, help="Fixed search radius (m). If omitted and --adaptive-radius used, adaptive strategy is applied.")
    ap.add_argument('--knn',    type=int,   default=200,  help="kNN if radius not used.")
    # Robust/fit defaults
    ap.add_argument('--mad_alpha',     type=float, default=6.0)
    ap.add_argument('--ransac_thresh', type=float, default=0.08)
    ap.add_argument('--ransac_iters',  type=int,   default=400)
    ap.add_argument('--min_inliers',   type=int,   default=5)
    ap.add_argument('--min_neighbors', type=int,   default=5)
    # LAS filters
    ap.add_argument('--las-keep-class',   default=None, help="Comma-separated LAS classes to keep (e.g. '2' for ground).")
    ap.add_argument('--las-keep-returns', default=None, help="Comma-separated return numbers to keep (e.g. '1,2').")
    # Selection filters (hard guards)
    ap.add_argument('--xy-maxdist', type=float, default=None, help="Max XY distance (m) from CP for neighbors (HARD cap).")
    ap.add_argument('--z-window',   type=float, default=None, help="Keep neighbors within ±Z (m) of CP height before fitting.")
    # Adaptive radius
    ap.add_argument('--adaptive-radius', action='store_true', help="Enable adaptive radius growth to reach min_neighbors.")
    ap.add_argument('--adaptive-radius-start', type=float, default=0.15)
    ap.add_argument('--adaptive-radius-step',  type=float, default=0.05)
    ap.add_argument('--adaptive-radius-max',   type=float, default=0.40)
    # Centroid guard
    ap.add_argument('--centroid-max-offset', type=float, default=0.15, help="Max allowed XY offset (m) between inlier centroid and CP; flags 'centroid_far'.")
    # Tolerance for CP acceptance (absolute dZ)
    ap.add_argument('--tolerance', type=float, default=0.025, help="Acceptable absolute dZ tolerance in meters (default 0.025).")
    # Export
    ap.add_argument('--export-points', action='store_true', help="Write cp_check_points.csv with used pointcloud points.")
    ap.add_argument('--export-stage', choices=['neighbors','after_mad','inliers','all'], default='inliers',
                    help="Which points to export (default: inliers).")
    args = ap.parse_args()

    # Load data
    pts = load_point_cloud(args.points, csv_decimal=args.decimal,
                           las_keep_class=args.las_keep_class, las_keep_returns=args.las_keep_returns)
    ids, cps_ne, cps_en = load_cps_variants(args.cps, csv_sep=args.cps_sep, csv_decimal=args.decimal)

    eval_params = dict(
        radius=args.radius, knn=args.knn, mad_alpha=args.mad_alpha,
        ransac_thresh=args.ransac_thresh, ransac_iters=args.ransac_iters,
        min_inliers=args.min_inliers, min_neighbors=args.min_neighbors,
        export_stage=(args.export_stage if args.export_points else None),
        xy_maxdist=args.xy_maxdist, z_window=args.z_window,
        adaptive_radius=args.adaptive_radius,
        adaptive_radius_start=args.adaptive_radius_start,
        adaptive_radius_step=args.adaptive_radius_step,
        adaptive_radius_max=args.adaptive_radius_max,
        centroid_max_offset=args.centroid_max_offset
    )

    # Evaluate NEH (preferred) then ENH
    res_ne, met_ne, exp_ne = evaluate_cps(pts, ids, cps_ne, **eval_params)
    res_en, met_en, exp_en = evaluate_cps(pts, ids, cps_en, **eval_params)
    chosen = 'NEH' if met_ne['ok_count'] >= met_en['ok_count'] else 'ENH'
    if chosen == 'NEH': results, metrics, export_rows = res_ne, met_ne, exp_ne
    else:               results, metrics, export_rows = res_en, met_en, exp_en

    # Build CSV (per CP)
    out_df = pd.DataFrame(results, columns=[
        'id','X','Y','Z_CP','dZ','d_perp',
        'n_neighbors','n_after_mad','n_inliers','status',
        'centroid_dx','centroid_dy','centroid_dxy','used_radius'
    ])
    # TOTAL row
    total_row = {
        'id':'TOTAL','X':np.nan,'Y':np.nan,'Z_CP':np.nan,
        'dZ':metrics.get('bias_Z',np.nan),
        'd_perp':metrics.get('rmse_3D_perp',np.nan),
        'n_neighbors':metrics.get('n',np.nan),
        'n_after_mad':np.nan,
        'n_inliers':metrics.get('n',np.nan),
        'status':f"rmse_Z={metrics.get('rmse_Z',np.nan):.3f} P95_absZ={metrics.get('P95_absZ',np.nan):.3f}",
        'centroid_dx':np.nan,'centroid_dy':np.nan,'centroid_dxy':np.nan,'used_radius':np.nan
    }
    out_df = pd.concat([out_df, pd.DataFrame([total_row])], ignore_index=True)

    # Compute within tolerance flag for per-CP rows (exclude TOTAL)
    within = (out_df['id'] != 'TOTAL') & (out_df['dZ'].abs() <= args.tolerance)
    out_df.insert(loc=5, column='within_tol', value=within.where(out_df['id'] != 'TOTAL', np.nan))

    # Annotate out-of-tolerance rows in status (keep other failure statuses intact)
    oob_mask = (out_df['id'] != 'TOTAL') & (~out_df['within_tol'].astype(bool)) & (out_df['status'] == 'ok')
    out_df.loc[oob_mask, 'status'] = 'out_of_tol'

    # Compute tolerance counts and update TOTAL row status
    within_count = int(out_df.loc[out_df['id'] != 'TOTAL', 'within_tol'].sum())
    total_count = int((out_df['id'] != 'TOTAL').sum())
    out_count = int(total_count - within_count)
    total_idx = out_df.index[out_df['id'] == 'TOTAL']
    if len(total_idx) == 1:
        i = total_idx[0]
        out_df.at[i, 'status'] = (
            f"rmse_Z={metrics.get('rmse_Z',np.nan):.3f} "
            f"P95_absZ={metrics.get('P95_absZ',np.nan):.3f} "
            f"within_tol={within_count} out_of_tol={out_count} tol={args.tolerance}"
        )

    # Save next to point cloud
    points_dir = os.path.dirname(os.path.abspath(args.points))
    out_csv_path = os.path.join(points_dir, "cp_check_results.csv")
    summary_path = os.path.join(points_dir, "cp_check_results_summary.txt")
    out_df.to_csv(out_csv_path, index=False)
    with open(summary_path,'w',encoding='utf-8') as f:
        f.write(f"CHECK SUMMARY (chosen mapping: {chosen})\n")
        for k,v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write(f"tolerance_abs_dZ: {args.tolerance}\n")
        f.write(f"within_tol_count: {within_count}\n")
        f.write(f"out_of_tol_count: {out_count}\n")

    # Optional export
    extra = ""
    if args.export_points:
        export_df = pd.DataFrame(export_rows, columns=[
            'cp_id','stage','Xp','Yp','Zp','dZ_to_plane','d_perp_to_plane','dist_xy_to_cp'
        ]) if export_rows else pd.DataFrame(columns=[
            'cp_id','stage','Xp','Yp','Zp','dZ_to_plane','d_perp_to_plane','dist_xy_to_cp'
        ])
        export_path = os.path.join(points_dir, "cp_check_points.csv")
        export_df.to_csv(export_path, index=False)
        if export_rows:
            extra = f"\nWrote exported points to: {export_path}"
        else:
            extra = f"\nExport requested but no points matched export-stage (file written with headers only): {export_path}"

    print("=== SUMMARY ===")
    print(f"Chosen CP mapping: {chosen}")
    for k,v in metrics.items():
        print(f"{k}: {v}")
    print(f"\nWrote per-CP results (with TOTAL row) to: {out_csv_path}")
    print(f"Wrote summary to: {summary_path}{extra}")

if __name__ == '__main__':
    main()