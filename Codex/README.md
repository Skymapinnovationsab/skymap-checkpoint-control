Change Detection for Orthophotos (Polygon SHP Export)

Overview
- Aligns two orthophotos of the same area
- Detects color-space changes robust to lighting (HSV-based)
- Thresholds automatically (Otsu) or with a user-specified value
- Removes speckles, vectorizes the change mask, and exports polygons to Shapefile

Installation
1) Ensure GDAL stack is available (required by rasterio/fiona).
2) Install Python packages:
   pip install -r requirements.txt

Usage
python change_detect.py \
  --before BEFORE.tif \
  --after AFTER.tif \
  --out changes.shp \
  --min-area-m2 100 \
  --min-pixels 64 \
  --simplify 0.5 \
  --smooth-buffer 0.5

Also supports GeoPackage and preview PNG:
python change_detect.py \
  --before BEFORE.tif \
  --after AFTER.tif \
  --out changes.gpkg:earthworks \
  --preview preview.png \
  --preview-background before \
  --min-area-m2 100

Export the binary change mask as GeoTIFF:
python change_detect.py \
  --before BEFORE.tif \
  --after AFTER.tif \
  --out changes.shp \
  --mask-tif change_mask.tif

Export the continuous change heatmap as GeoTIFF:
python change_detect.py \
  --before BEFORE.tif \
  --after AFTER.tif \
  --out changes.shp \
  --heatmap-tif change_heatmap.tif

Batch processing from CSV:
CSV requires headers: before, after, out
Optional headers: threshold, weights, min_pixels, min_area_m2, simplify, smooth_buffer, preview, preview_background, mask_tif, heatmap_tif

Example CSV (comma-separated):
before,after,out,preview,mask_tif,heatmap_tif,min_area_m2
/data/b1.tif,/data/a1.tif,/out/changes1.shp,/out/p1.png,/out/m1.tif,/out/h1.tif,100
/data/b2.tif,/data/a2.tif,/out/changes2.gpkg:layer2,,,/out/h2.tif,200

Run batch:
python change_detect.py --batch-csv jobs.csv --batch-continue --weights 2,1,0.5 --min-pixels 64

Key Options
- --threshold: Manual threshold in [0,1]. If omitted, Otsu auto-threshold is used.
- --weights: HSV weights for change metric, default 2,1,0.5 (hue emphasized).
- --min-pixels: Removes connected components smaller than N pixels before vectorization.
- --min-area-m2: Filters polygons by area (m²). If the CRS is geographic, areas are approximated by projecting to a local UTM zone.
- --simplify: Douglas-Peucker tolerance in map units (e.g., meters) to simplify boundaries.
- --smooth-buffer: Smooth outlines by buffering and unbuffering polygons by this distance (map units).
- --preview: Save a PNG overlay of change heatmap and polygons.
- --preview-background: Choose 'after' (default) or 'before' for the preview image background.
- --mask-tif: Save the binary change mask (1=change, 0=no change) as a GeoTIFF with georeferencing.
- --heatmap-tif: Save the continuous (float32 [0,1]) change heatmap as GeoTIFF.
- --batch-csv: Process multiple pairs from a CSV file (columns listed above).
- --batch-continue: Continue processing subsequent rows if a row fails.

Notes and Tips
- Input rasters should overlap spatially and ideally be high quality and well-georeferenced. The script automatically crops to the intersection area.
- If imagery is single-band or has an alpha band, the script adapts (uses first 3 bands as RGB; repeats grayscale to RGB).
- The HSV-based metric is fairly robust to brightness differences (shadows). You can tune weights (e.g., increase saturation weight for soil vs vegetation differences).
- For earthmoving activities, try higher --min-area-m2 to remove noise, and use a small --smooth-buffer (e.g., 0.5–1.0 m) to get cleaner outlines.

Output
- Vector file: ESRI Shapefile (*.shp) or GeoPackage (*.gpkg). For GPKG, you can specify a layer name with ":layer" (default: "changes").
- Attributes: id, area_m2 (approx m² for geographic CRS, exact for projected CRS).
- Optional PNG preview of changes overlaid on the "after" image with polygon outlines.

Troubleshooting
- If you get errors about drivers/CRS, ensure GDAL is correctly installed and that both rasters have valid CRS.
- If results are too noisy: increase --min-pixels and/or --min-area-m2, add --smooth-buffer, and consider a slightly higher manual --threshold (e.g., 0.25–0.35).
- If changes are missed: reduce --min-pixels and --min-area-m2, lower --threshold, or increase hue/saturation weights via --weights.
