import os
import sys
import math
import numpy as np
import tempfile
import pandas as pd

# Load the target script via importlib since the filename contains spaces
import importlib.util


def load_module():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(base_dir, "Checkpoints_control_1.py")
    spec = importlib.util.spec_from_file_location("cp_check_module", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mad_based_filter_basic():
    mod = load_module()
    arr = np.array([0, 0, 0, 0, 10], dtype=float)
    mask = mod.mad_based_filter(arr, alpha=3.0)
    # Expect the outlier 10 to be filtered out
    assert mask.sum() == 4
    assert bool(mask[-1]) is False


def test_fit_plane_ls_recovery():
    mod = load_module()
    # Plane: z = 2x + 3y + 1
    rng = np.random.default_rng(0)
    xy = rng.uniform(-1, 1, size=(200, 2))
    z = 2 * xy[:, 0] + 3 * xy[:, 1] + 1
    pts = np.column_stack([xy, z])
    coef, resid = mod.fit_plane_ls(pts)
    assert coef is not None
    a, b, c = coef
    assert math.isclose(a, 2.0, rel_tol=1e-2, abs_tol=1e-2)
    assert math.isclose(b, 3.0, rel_tol=1e-2, abs_tol=1e-2)
    assert math.isclose(c, 1.0, rel_tol=1e-2, abs_tol=1e-2)


def test_ransac_plane_with_outliers():
    mod = load_module()
    rng = np.random.default_rng(1)
    # Inliers on plane z = x - y + 0.5 with small noise
    xy_in = rng.normal(0, 1, size=(150, 2))
    z_in = xy_in[:, 0] - xy_in[:, 1] + 0.5 + rng.normal(0, 0.005, size=150)
    inliers = np.column_stack([xy_in, z_in])
    # Outliers
    xy_out = rng.uniform(-5, 5, size=(50, 2))
    z_out = rng.uniform(-5, 5, size=50)
    outliers = np.column_stack([xy_out, z_out])
    pts = np.vstack([inliers, outliers])
    plane4, inlier_mask = mod.ransac_plane(pts, thresh=0.02, iters=300, min_inliers=50)
    assert plane4 is not None
    assert inlier_mask is not None
    # Evaluate at (0,0) should be close to z = 0.5
    z0 = mod.eval_plane_at_xy(plane4, 0.0, 0.0)
    assert abs(z0 - 0.5) < 0.05


def test_evaluate_cps_synthetic():
    mod = load_module()
    rng = np.random.default_rng(2)
    # Create synthetic point cloud on plane z = 0.1x + 0.2y + 10
    xy = rng.uniform(-2, 2, size=(1000, 2))
    z = 0.1 * xy[:, 0] + 0.2 * xy[:, 1] + 10 + rng.normal(0, 0.01, size=1000)
    pts = np.column_stack([xy, z])

    # Define CPs near the origin and another point
    ids = ["CP1", "CP2"]
    cps_ne = np.array([
        [0.1, -0.2, 10 + 0.1 * 0.1 + 0.2 * (-0.2)],
        [1.0, 1.0, 10 + 0.1 * 1.0 + 0.2 * 1.0],
    ], dtype=float)

    # Evaluate
    results, metrics, export_rows = mod.evaluate_cps(
        pts, ids, cps_ne,
        radius=None, knn=200, mad_alpha=6.0,
        ransac_thresh=0.03, ransac_iters=200, min_inliers=10, min_neighbors=10,
        export_stage='inliers', xy_maxdist=None, z_window=None,
        adaptive_radius=True, adaptive_radius_start=0.15, adaptive_radius_step=0.05, adaptive_radius_max=0.4,
        centroid_max_offset=0.5,
    )

    # Basic assertions
    assert len(results) == 2
    # Should have at least some exported inliers
    assert len(export_rows) > 0
    # Metrics should indicate finite values
    assert math.isfinite(metrics["rmse_Z"]) and math.isfinite(metrics["rmse_3D_perp"])


def test_load_cps_variants_semicolon():
    mod = load_module()
    # Create temporary CPS file with semicolon separator
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("1;100.0;200.0;50.0\n")
        f.write("2;101.0;201.0;51.0\n")
        f.write("3;102.0;202.0;52.0\n")
        temp_path = f.name
    
    try:
        ids, cps_ne, cps_en = mod.load_cps_variants(temp_path, csv_sep=';', csv_decimal='.')
        assert len(ids) == 3
        assert cps_ne.shape == (3, 3)
        assert cps_en.shape == (3, 3)
        # Check NEH mapping: X<-N, Y<-E, Z<-H
        assert np.allclose(cps_ne[0], [200.0, 100.0, 50.0])
        # Check ENH mapping: X<-E, Y<-N, Z<-H
        assert np.allclose(cps_en[0], [100.0, 200.0, 50.0])
    finally:
        os.unlink(temp_path)


def test_load_cps_variants_comma():
    mod = load_module()
    # Create temporary CPS file with comma separator
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("1,100.0,200.0,50.0\n")
        f.write("2,101.0,201.0,51.0\n")
        f.write("3,102.0,202.0,52.0\n")
        temp_path = f.name
    
    try:
        ids, cps_ne, cps_en = mod.load_cps_variants(temp_path, csv_sep=',', csv_decimal='.')
        assert len(ids) == 3
        assert cps_ne.shape == (3, 3)
        assert cps_ne.shape == (3, 3)
    finally:
        os.unlink(temp_path)


def test_load_cps_variants_auto_detect():
    mod = load_module()
    # Create temporary CPS file with semicolon separator
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("1;100.0;200.0;50.0\n")
        f.write("2;101.0;201.0;51.0\n")
        f.write("3;102.0;202.0;52.0\n")
        temp_path = f.name
    
    try:
        # Should auto-detect semicolon separator
        ids, cps_ne, cps_en = mod.load_cps_variants(temp_path, csv_sep=';', csv_decimal='.')
        assert len(ids) == 3
        assert cps_ne.shape == (3, 3)
        assert cps_en.shape == (3, 3)
    finally:
        os.unlink(temp_path)


def test_load_point_cloud_csv():
    mod = load_module()
    # Create temporary point cloud CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("x,y,z\n")
        f.write("1.0,2.0,3.0\n")
        f.write("4.0,5.0,6.0\n")
        f.write("7.0,8.0,9.0\n")
        temp_path = f.name
    
    try:
        pts = mod.load_point_cloud(temp_path, csv_sep=',', csv_decimal='.')
        assert pts.shape == (3, 3)
        assert np.allclose(pts[0], [1.0, 2.0, 3.0])
        assert np.allclose(pts[1], [4.0, 5.0, 6.0])
        assert np.allclose(pts[2], [7.0, 8.0, 9.0])
    finally:
        os.unlink(temp_path)


def test_load_point_cloud_headerless():
    mod = load_module()
    # Create temporary headerless point cloud CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("1.0,2.0,3.0\n")
        f.write("4.0,5.0,6.0\n")
        f.write("7.0,8.0,9.0\n")
        temp_path = f.name
    
    try:
        pts = mod.load_point_cloud(temp_path, csv_sep=',', csv_decimal='.')
        assert pts.shape == (3, 3)
        assert np.allclose(pts[0], [1.0, 2.0, 3.0])
    finally:
        os.unlink(temp_path)


def test_compute_metrics_basic():
    mod = load_module()
    # Test with finite values
    dz = [0.1, -0.2, 0.15, -0.05, 0.3]
    dp = [0.08, 0.18, 0.12, 0.04, 0.25]
    
    metrics = mod.compute_metrics(dz, dp)
    
    assert metrics['n'] == 5
    assert math.isclose(metrics['bias_Z'], 0.06, abs_tol=1e-10)
    assert math.isclose(metrics['rmse_Z'], 0.182, abs_tol=1e-3)
    assert math.isclose(metrics['rmse_3D_perp'], 0.153, abs_tol=1e-3)
    assert metrics['ok_count'] == 5


def test_compute_metrics_with_nans():
    mod = load_module()
    # Test with some NaN values
    dz = [0.1, np.nan, 0.15, -0.05, np.nan]
    dp = [0.08, 0.18, np.nan, 0.04, 0.25]
    
    metrics = mod.compute_metrics(dz, dp)
    
    assert metrics['n'] == 3  # Only finite values counted
    assert math.isclose(metrics['bias_Z'], 0.067, abs_tol=1e-3)
    assert math.isclose(metrics['rmse_Z'], 0.108, abs_tol=1e-3)
    assert math.isclose(metrics['rmse_3D_perp'], 0.160, abs_tol=1e-3)


def test_compute_metrics_empty():
    mod = load_module()
    # Test with empty arrays
    dz = []
    dp = []
    
    metrics = mod.compute_metrics(dz, dp)
    
    assert metrics['n'] == 0
    assert np.isnan(metrics['bias_Z'])
    assert np.isnan(metrics['rmse_Z'])
    assert metrics['ok_count'] == 0


def test_plane_point_distance():
    mod = load_module()
    # Plane: z = 2x + 3y + 1 (normalized: 2x + 3y - z + 1 = 0)
    plane4 = (2, 3, -1, 1)
    pts = np.array([[0, 0, 1], [1, 1, 6], [0.5, 0.5, 3]])
    
    distances = mod.plane_point_distance(plane4, pts)
    
    assert len(distances) == 3
    # First point should be on the plane (distance = 0)
    assert math.isclose(distances[0], 0.0, abs_tol=1e-10)
    # Other points should have positive distances
    assert distances[1] >= 0  # Allow for numerical precision
    assert distances[2] >= 0  # Allow for numerical precision


def test_eval_plane_at_xy():
    mod = load_module()
    # Plane: z = 2x + 3y + 1
    plane4 = (2, 3, -1, 1)
    
    # Test at origin
    z0 = mod.eval_plane_at_xy(plane4, 0, 0)
    assert math.isclose(z0, 1.0, abs_tol=1e-10)
    
    # Test at (1, 1)
    z1 = mod.eval_plane_at_xy(plane4, 1, 1)
    assert math.isclose(z1, 6.0, abs_tol=1e-10)


def test_error_handling_invalid_cps():
    mod = load_module()
    # Create temporary CPS file with insufficient columns
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("1,100.0\n")  # Only 2 columns
        f.write("2,101.0\n")
        temp_path = f.name
    
    try:
        with np.testing.assert_raises(RuntimeError):
            mod.load_cps_variants(temp_path, csv_sep=',', csv_decimal='.')
    finally:
        os.unlink(temp_path)


def test_error_handling_missing_file():
    mod = load_module()
    # Test with non-existent file
    with np.testing.assert_raises(FileNotFoundError):
        mod.load_point_cloud("/nonexistent/file.csv")


def test_mad_based_filter_edge_cases():
    mod = load_module()
    # Test with single value
    single = np.array([5.0])
    mask = mod.mad_based_filter(single, alpha=3.0)
    assert mask.sum() == 1
    assert mask[0] == True
    
    # Test with all identical values
    identical = np.array([3.0, 3.0, 3.0, 3.0])
    mask = mod.mad_based_filter(identical, alpha=3.0)
    assert mask.sum() == 4  # All should pass
    
    # Test with empty array
    empty = np.array([])
    mask = mod.mad_based_filter(empty, alpha=3.0)
    assert len(mask) == 0


def test_fit_plane_ls_edge_cases():
    mod = load_module()
    # Test with insufficient points
    insufficient = np.array([[1, 2, 3]])
    coef, resid = mod.fit_plane_ls(insufficient)
    assert coef is None
    assert resid is None
    
    # Test with collinear points (should still work)
    collinear = np.array([
        [0, 0, 1],
        [1, 0, 1],
        [2, 0, 1]
    ])
    coef, resid = mod.fit_plane_ls(collinear)
    assert coef is not None
    # Should fit a horizontal plane
    assert math.isclose(coef[2], 1.0, abs_tol=1e-10) 


