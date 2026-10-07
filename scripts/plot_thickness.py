import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize
from matplotlib.ticker import FuncFormatter
import geopandas as gpd
import rioxarray
import xarray as xr
from datetime import datetime
import cmocean

# ==========================================
# Global Publication Quality Formatting Settings
# ==========================================
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['pdf.fonttype'] = 42  # Keeps text editable in vector outputs
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

def apply_imageschs(data, background, cmap, norm, weight=0.5, alpha=1.0):
    """
    Replicates MATLAB's TopoToolbox / CDT 'imageschs' multiplicative blending.
    Displays background hillshade where overlay data has NaNs.
    """
    data_arr = np.asarray(data)
    bg_arr = np.asarray(background)
    
    # 1. Normalize background grayscale map to [0, 1]
    bg_min, bg_max = np.nanmin(bg_arr), np.nanmax(bg_arr)
    bg_norm = (bg_arr - bg_min) / (bg_max - bg_min) if bg_max > bg_min else np.zeros_like(bg_arr)
    
    # 2. Map overlay data to RGB array
    rgb = cmap(norm(data_arr))[:, :, :3]
    
    # 3. Multiplicative shading factor
    shading_factor = (1.0 - weight) + weight * bg_norm
    
    # 4. Apply texture to the colored data
    textured_rgb = rgb * shading_factor[..., np.newaxis]
    
    # 5. Create pure grayscale RGB background
    bg_rgb = np.stack([bg_norm, bg_norm, bg_norm], axis=-1)
    
    # 6. Blend textured color with pure background based on alpha
    blended = (textured_rgb * alpha) + (bg_rgb * (1.0 - alpha))
    
    # 7. Restore pure grayscale background where data has NaNs/gaps
    nan_mask = np.isnan(data_arr)
    for c in range(3):
        blended[:, :, c][nan_mask] = bg_norm[nan_mask]
        
    # Render true background NaNs (e.g. ocean mask) as pure white
    bg_nan_mask = np.isnan(bg_arr)
    blended[bg_nan_mask] = 1.0
        
    return np.clip(blended, 0, 1)

def extract_transect(shp_path, raster_data, spacing=5.0):
    """Helper function to extract raster values along a LineString."""
    gdf = gpd.read_file(shp_path)
    line = gdf.geometry.iloc[0]
    distances = np.arange(0, line.length, spacing)
    points = [line.interpolate(d) for d in distances]
    
    x_coords = xr.DataArray([p.x for p in points], dims="distance")
    y_coords = xr.DataArray([p.y for p in points], dims="distance")
    
    dz_vals = raster_data.sel(x=x_coords, y=y_coords, method="nearest").values
    return distances, dz_vals, points, line

# ==========================================
# File Paths and Directories
# ==========================================
dem_dir = '/mnt/parscratch/users/gg1bjd/research/swift/data/raw/images/rema'
dem1_name = 'SETSM_s2s041_WV02_20171007_10300100708B5400_1030010071CE6F00_2m_lsf_seg2_coreg.tif'
dem2_name = 'SETSM_s2s041_WV03_20180915_1040010041957300_104001004203FF00_2m_lsf_seg1_coreg.tif'

dhdt_path = '/mnt/parscratch/users/gg1bjd/research/swift/data/raw/dz/dhdt/S65W058_2010-01-01_2015-01-01_dhdt.tif'
mosaic_raw_path = '/mnt/parscratch/users/gg1bjd/research/swift/data/raw/images/rema/JRI_2m_REMA_mosaic_geoid_corrected_and_ocean_masked.tif'

source_shp_path = '/mnt/parscratch/users/gg1bjd/research/swift/data/raw/vector/landslide/landslide_source_outline_2019-09-19_vs_2016-12-31.shp'
bounds_shp_path = '/mnt/parscratch/users/gg1bjd/research/swift/data/raw/vector/bounds/bounds.shp'
transect_across_path = '/mnt/parscratch/users/gg1bjd/research/swift/data/raw/vector/transect/across.shp'
transect_along_path = '/mnt/parscratch/users/gg1bjd/research/swift/data/raw/vector/transect/along.shp'

pub_dir = '/mnt/parscratch/users/gg1bjd/research/swift/data/pub/landslide_thickness'
mosaic_pub_dir = '/mnt/parscratch/users/gg1bjd/research/swift/data/pub/images/rema'
RESULTS_DIR = "/mnt/parscratch/users/gg1bjd/research/swift/results"

os.makedirs(pub_dir, exist_ok=True)
os.makedirs(mosaic_pub_dir, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Output cached files
cropped_dz_out = os.path.join(pub_dir, 'swift_landslide_thickness_20180915_is2_cropped.tif')
cropped_dem2_out = os.path.join(pub_dir, 'elevation_20180915_cropped.tif')
cropped_mosaic_out = os.path.join(mosaic_pub_dir, 'rema_2m_geoid_masked.tif')

dem1_date_str = '20171007'
dem2_date_str = '20180915'

# ==========================================
# Processing / Loading Data
# ==========================================
bounds_shp = gpd.read_file(bounds_shp_path)

if os.path.exists(cropped_dz_out) and os.path.exists(cropped_dem2_out):
    print("Cropped landslide outputs exist. Loading directly...")
    dz_cropped = rioxarray.open_rasterio(cropped_dz_out).squeeze()
    dem2_cropped = rioxarray.open_rasterio(cropped_dem2_out).squeeze()
else:
    print("Calculating elevation changes and volume...")
    dem1 = rioxarray.open_rasterio(os.path.join(dem_dir, dem1_name)).squeeze()
    dem2 = rioxarray.open_rasterio(os.path.join(dem_dir, dem2_name)).squeeze()
    
    dem1 = dem1.where(dem1 != -9999)
    dem2 = dem2.where(dem2 != -9999)
    
    dhdt = rioxarray.open_rasterio(dhdt_path).squeeze()
    dhdt = dhdt.where(dhdt != -9999)
    dhdt_resampled = dhdt.rio.reproject_match(dem1)
    
    date1 = datetime.strptime(dem1_date_str, '%Y%m%d')
    date2 = datetime.strptime(dem2_date_str, '%Y%m%d')
    time_diff_years = (date2 - date1).days / 365.25
    
    background_dh = dhdt_resampled * time_diff_years
    dz_landslide = dem2 - (dem1 + background_dh)
    
    source_shp = gpd.read_file(source_shp_path)
    dz_source = dz_landslide.rio.clip(source_shp.geometry, source_shp.crs)
    
    A = 4.0 # 2m x 2m pixel area
    valid_dz = np.nan_to_num(dz_source.values)
    total_VS_km3 = (np.sum(valid_dz) * A) * 1e-9
    
    vsp = 2500 * 1e-9
    print(f"Source Volume: {total_VS_km3:.6f} km³ ({total_VS_km3 / vsp:,.0f} Olympic Swimming Pools)")
    
    dz_cropped = dz_landslide.rio.clip(bounds_shp.geometry, bounds_shp.crs)
    dem2_cropped = dem2.rio.clip(bounds_shp.geometry, bounds_shp.crs)
    
    dz_cropped.rio.to_raster(cropped_dz_out, compress='deflate')
    dem2_cropped.rio.to_raster(cropped_dem2_out, compress='deflate')

# Clean non-data/outlier values (these NaNs are intentionally kept in the saved array)
dz_cropped = dz_cropped.where((dz_cropped != -9999) & (dz_cropped > -1000), np.nan)
dem2_cropped = dem2_cropped.where((dem2_cropped != -9999) & (dem2_cropped > -1000), np.nan)

# Load / Process Regional Background Mosaic DEM
if os.path.exists(cropped_mosaic_out):
    print("Cropped regional DEM mosaic exists. Loading directly...")
    mosaic_cropped = rioxarray.open_rasterio(cropped_mosaic_out).squeeze()
else:
    print("Processing regional background DEM mosaic...")
    mosaic = rioxarray.open_rasterio(mosaic_raw_path).squeeze()
    mosaic = mosaic.where((mosaic != -9999) & (mosaic > -1000), np.nan)
    
    mosaic_cropped = mosaic.rio.clip(bounds_shp.geometry, bounds_shp.crs)
    mosaic_cropped = mosaic_cropped.rio.reproject_match(dz_cropped)
    mosaic_cropped.rio.to_raster(cropped_mosaic_out, compress='deflate')

mosaic_cropped = mosaic_cropped.where((mosaic_cropped != -9999) & (mosaic_cropped > -1000), np.nan)

# ==========================================
# Transect Extraction
# ==========================================
print("Extracting transects...")
dist_A, dz_A, pts_A, line_A = extract_transect(transect_across_path, dz_cropped)
dist_B, dz_B, pts_B, line_B = extract_transect(transect_along_path, dz_cropped)

# ==========================================
# Plotting
# ==========================================
print("Generating publication-quality figure...")
ls = LightSource(azdeg=315, altdeg=45)

# 1. Generate primary hillshade from the local dem2
hs_dem2 = ls.hillshade(np.nan_to_num(dem2_cropped.values), vert_exag=1, dx=2, dy=2)

# 2. Generate backup hillshade from the broad mosaic DEM
hs_mosaic = ls.hillshade(np.nan_to_num(mosaic_cropped.values), vert_exag=1, dx=2, dy=2)

# 3. Merge hillshades: Use dem2 hillshade as the base, fill its gaps with the mosaic hillshade
dem2_nan_mask = np.isnan(dem2_cropped.values)
hillshade_bg = np.where(dem2_nan_mask, hs_mosaic, hs_dem2)

# 4. Restore NaNs where BOTH datasets are missing data (e.g., ocean mask)
mosaic_nan_mask = np.isnan(mosaic_cropped.values)
hillshade_bg[dem2_nan_mask & mosaic_nan_mask] = np.nan

cmap = cmocean.cm.balance
norm = Normalize(vmin=-150, vmax=150)

# Pass the unmodified dz_cropped.values (which still contains NaNs) to the imageschs function
blended_img = apply_imageschs(dz_cropped.values, hillshade_bg, cmap, norm, weight=0.6)

extent = [
    dz_cropped.x.min().item(), dz_cropped.x.max().item(),
    dz_cropped.y.min().item(), dz_cropped.y.max().item()
]

fig = plt.figure(figsize=(12, 7.5))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.2], height_ratios=[1, 1])

ax1 = fig.add_subplot(gs[:, 0])    # Map spans left column
ax2 = fig.add_subplot(gs[0, 1])    # Transect A-A' top right
ax3 = fig.add_subplot(gs[1, 1])    # Transect B-B' bottom right

# -- Panel 1: Map (ax1) --
ax1.imshow(blended_img, extent=extent, origin='upper', rasterized=True)

# Plot Transects on Map
ax1.plot(*line_A.xy, color='black', linewidth=1.5, linestyle='--')
ax1.text(pts_A[0].x, pts_A[0].y, "A", fontsize=11, fontweight='bold', ha='right', va='bottom', color='black')
ax1.text(pts_A[-1].x, pts_A[-1].y, "A'", fontsize=11, fontweight='bold', ha='left', va='top', color='black')

ax1.plot(*line_B.xy, color='black', linewidth=1.5, linestyle='--')
ax1.text(pts_B[0].x, pts_B[0].y, "B", fontsize=11, fontweight='bold', ha='right', va='bottom', color='black')
ax1.text(pts_B[-1].x, pts_B[-1].y, "B'", fontsize=11, fontweight='bold', ha='left', va='top', color='black')

# Format Axes (meters -> kilometers)
km_formatter = FuncFormatter(lambda x, pos: f"{x/1000:g}")
ax1.xaxis.set_major_formatter(km_formatter)
ax1.yaxis.set_major_formatter(km_formatter)

ax1.set_xlabel('Easting (km)')
ax1.set_ylabel('Northing (km)')

# Compact Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax1, shrink=0.5, aspect=20, pad=0.04)
cbar.set_label('Landslide-induced elevation change (m)')

# -- Transect Helper Function --
def plot_transect(ax, dist, dz, label_start, label_end):
    ax.fill_between(dist, dz, 0, where=(dz >= 0), color='#d73027', alpha=0.8, interpolate=True)
    ax.fill_between(dist, dz, 0, where=(dz < 0), color='#4575b4', alpha=0.8, interpolate=True)
    ax.plot(dist, dz, color='black', linewidth=1)
    
    ax.set_xlim(0, max(dist))
    ax.set_xlabel('Distance along transect (m)')
    ax.set_ylabel('Landslide-induced elevation change (m)')
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Label endpoints at bottom
    ax.text(0.01, 0.025, label_start, transform=ax.transAxes, fontsize=11, fontweight='bold', va='bottom')
    ax.text(0.99, 0.025, label_end, transform=ax.transAxes, fontsize=11, fontweight='bold', ha='right', va='bottom')

# -- Panel 2: Transect A-A' (ax2) --
plot_transect(ax2, dist_A, dz_A, "A", "A'")

# -- Panel 3: Transect B-B' (ax3) --
plot_transect(ax3, dist_B, dz_B, "B", "B'")

# -- Publication Panel Lettering --
for ax, letter in zip([ax1, ax2, ax3], ['a', 'b', 'c']):
    ax.text(0.02, 0.975, letter, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')

plt.tight_layout()

# ==========================================
# Export Multi-format High-Resolution Outputs
# ==========================================
output_base = f"{RESULTS_DIR}/landslide_thickness_{dem1_date_str}_{dem2_date_str}"
for ext in ['png', 'pdf', 'svg']:
    print(f"Saving {output_base}.{ext}...")
    plt.savefig(f"{output_base}.{ext}", bbox_inches='tight', dpi=600)