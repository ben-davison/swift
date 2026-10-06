import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LightSource
import matplotlib.patches as patches
import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import box
import pdemtools as pdt
import rioxarray
import xarray as xr
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# ==========================================
# 1. Configuration and Setup
# ==========================================

# Base Directories
S2_RAW_DIR = "/mnt/parscratch/users/gg1bjd/research/swift/data/raw/images/sentinel2"
S1_RAW_DIR = "/mnt/parscratch/users/gg1bjd/research/swift/data/raw/images/sentinel1"
BOUNDS_SHP = "/mnt/parscratch/users/gg1bjd/research/swift/data/raw/vector/bounds/bounds.shp"

PUB_S2_DIR = "/mnt/parscratch/users/gg1bjd/research/swift/data/pub/images/sentinel2"
PUB_S1_DIR = "/mnt/parscratch/users/gg1bjd/research/swift/data/pub/images/sentinel1"
PUB_REMA_DIR = "/mnt/parscratch/users/gg1bjd/research/swift/data/pub/images/rema"
PLOT_DIR = "/mnt/parscratch/users/gg1bjd/research/swift/results"

os.makedirs(PUB_S2_DIR, exist_ok=True)
os.makedirs(PUB_S1_DIR, exist_ok=True)
os.makedirs(PUB_REMA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

s2_b04_files = [
    os.path.join(S2_RAW_DIR, "S2B_MSIL2A_20180228T131859_N0500_R095_T21DVJ_20230731T044442.SAFE/GRANULE/L2A_T21DVJ_A005126_20180228T131925/IMG_DATA/R10m/T21DVJ_20180228T131859_B04_10m.jp2"),
    None,  # Panel 'b' uses Sentinel-1 backscatter instead
    os.path.join(S2_RAW_DIR, "S2B_MSIL2A_20181205T131909_N0500_R095_T21DVJ_20230613T173637.SAFE/GRANULE/L2A_T21DVJ_A009130_20181205T131903/IMG_DATA/R10m/T21DVJ_20181205T131909_B04_10m.jp2")
]

rema_points = [
    (-57.74164, -64.30168),
    (-57.72748, -64.30704),
    (-57.71393, -64.31269),
    (-57.759453, -64.330543)
]
rema_strip_id = "SETSM_s2s041_WV01_20190919_102001008B3FC900_102001008C269300_2m_lsf_seg1"

# Load the bounds shapefile
bounds_gdf = gpd.read_file(BOUNDS_SHP)
fontprops = fm.FontProperties(size=7.5)

def stretch_glacier_rgb(rgb_array, gamma=0.5, bottom_percentile=5, top_percentile=95):
    """Applies a per-band percentile clip and gamma stretch."""
    enhanced = np.zeros_like(rgb_array, dtype=np.float32)
    for i in range(3):
        band = rgb_array[i]
        valid_pixels = band[band > 0] 
        if len(valid_pixels) == 0:
            continue
            
        p_bottom, p_top = np.percentile(valid_pixels, (bottom_percentile, top_percentile))
        
        if p_top > p_bottom:
            stretched = (band - p_bottom) / (p_top - p_bottom)
        else:
            stretched = np.zeros_like(band)
            
        stretched = np.clip(stretched, 0, 1)
        bright = stretched ** gamma
        enhanced[i] = bright
        
    return (enhanced * 255).astype(np.uint8)

# ==========================================
# 2. Figure & GridSpec Setup
# ==========================================

fig = plt.figure(figsize=(6.5, 5.5)) 
gs = GridSpec(2, 12, figure=fig, hspace=0.15, wspace=0.6) 

top_axes = [fig.add_subplot(gs[0, i*4:(i+1)*4]) for i in range(3)]
bottom_axes = [fig.add_subplot(gs[1, i*3:(i+1)*3]) for i in range(4)]

# ==========================================
# 3. Top Row: Sentinel Imagery
# ==========================================

top_labels = ['a', 'b', 'c']

for idx in range(3):
    ax = top_axes[idx]
    
    if idx == 1:
        # Sentinel-1 Backscatter Image for Panel 'b'
        s1_filename = "20180414_038_0811_F2_db.tif"
        raw_s1_path = os.path.join(S1_RAW_DIR, s1_filename)
        pub_s1_path = os.path.join(PUB_S1_DIR, s1_filename)
        
        if os.path.exists(pub_s1_path):
            print(f"Loading existing cropped Sentinel-1 image: {pub_s1_path}")
            with rioxarray.open_rasterio(pub_s1_path) as src:
                s1_data = src.squeeze().values.astype(np.float32)
                s1_data[s1_data == -9999] = np.nan
                bnds = src.rio.bounds()
                extent_s1 = [bnds[0], bnds[2], bnds[1], bnds[3]]
        else:
            print(f"Processing, reprojecting, and cropping raw Sentinel-1 image from {raw_s1_path}...")
            s1_da = rioxarray.open_rasterio(raw_s1_path).squeeze()
            bounds_3031 = bounds_gdf.to_crs("EPSG:3031")
            
            if s1_da.rio.crs != "EPSG:3031":
                s1_da = s1_da.rio.reproject("EPSG:3031", resampling=Resampling.bilinear)
                
            clipped_s1 = s1_da.rio.clip(bounds_3031.geometry, bounds_3031.crs, drop=True)
            clipped_s1.rio.to_raster(pub_s1_path, driver="GTiff")
            
            s1_data = clipped_s1.values.astype(np.float32)
            s1_data[s1_data == -9999] = np.nan
            bnds = clipped_s1.rio.bounds()
            extent_s1 = [bnds[0], bnds[2], bnds[1], bnds[3]]
        
        extent_km = [e / 1000.0 for e in extent_s1]
        ax.imshow(s1_data, cmap='gray', extent=extent_km, interpolation='none')
        formatted_date = "2018-04-14"
        
    else:
        # Sentinel-2 Optical RGB Images for Panels 'a' and 'c'
        b04_path = s2_b04_files[idx]
        base_name = os.path.basename(b04_path)
        tile_name = base_name.split('_')[0]
        s2_date_str = base_name.split('_')[1][:8] 
        formatted_date = f"{s2_date_str[:4]}-{s2_date_str[4:6]}-{s2_date_str[6:8]}"
        
        out_filename = os.path.join(PUB_S2_DIR, f"s2_{tile_name}_{s2_date_str}_rgb_stretched_3031.tif")
        
        if os.path.exists(out_filename):
            print(f"Loading existing Sentinel-2 output: {out_filename}")
            with rioxarray.open_rasterio(out_filename) as src:
                plot_rgb = src.values
                bnds = src.rio.bounds()
                extent_s2 = [bnds[0], bnds[2], bnds[1], bnds[3]]
        else:
            print(f"Processing and reprojecting Sentinel-2 image {idx+1}...")
            b03_path = b04_path.replace('B04', 'B03')
            b02_path = b04_path.replace('B04', 'B02')
            
            b4_da = rioxarray.open_rasterio(b04_path).squeeze()
            b3_da = rioxarray.open_rasterio(b03_path).squeeze()
            b2_da = rioxarray.open_rasterio(b02_path).squeeze()
            
            rgb_da = xr.concat([b4_da, b3_da, b2_da], dim='band')
            
            rgb_3031 = rgb_da.rio.reproject("EPSG:3031", resampling=Resampling.bilinear)
            bounds_3031 = bounds_gdf.to_crs("EPSG:3031")
            clipped_rgb = rgb_3031.rio.clip(bounds_3031.geometry, bounds_3031.crs, drop=True)
            
            rgb_stack = clipped_rgb.values
            enhanced_rgb = stretch_glacier_rgb(rgb_stack)
            
            stretched_da = xr.DataArray(
                enhanced_rgb,
                coords=clipped_rgb.coords,
                dims=clipped_rgb.dims,
                attrs=clipped_rgb.attrs
            )
            stretched_da.rio.write_nodata(0, inplace=True)
            stretched_da.rio.to_raster(out_filename, driver="GTiff")
            
            plot_rgb = enhanced_rgb
            bnds = clipped_rgb.rio.bounds()
            extent_s2 = [bnds[0], bnds[2], bnds[1], bnds[3]]

        extent_km = [e / 1000.0 for e in extent_s2]
        ax.imshow(plot_rgb.transpose(1, 2, 0), extent=extent_km, interpolation='none')
    
    # Axis formatting
    ax.tick_params(axis='both', which='major', labelsize=7)
    
    # 1) Y-axis logic: Label and tick labels only on leftmost panel
    if idx == 0:
        ax.set_ylabel("Northing (km)", fontsize=8)
    else:
        ax.tick_params(labelleft=False)
        
    # 2) X-axis logic: Label only on the middle column
    if idx == 1:
        ax.set_xlabel("Easting (km)", fontsize=8)
    
    ax.text(0.05, 0.95, top_labels[idx], transform=ax.transAxes, fontsize=10, fontweight='bold', 
            va='top', ha='left', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
            
    ax.text(0.5, 0.95, formatted_date, transform=ax.transAxes, fontsize=8.5, 
            va='top', ha='center', bbox=dict(facecolor='lightgrey', alpha=0.7, edgecolor='none', pad=3))
            
    # 3) Scalebar logic: Bottom left, only on panel "a"
    if idx == 0:
        scalebar_s2 = AnchoredSizeBar(ax.transData, 2, '2 km', 'lower left', 
                                      pad=0.2, color='black', frameon=True, 
                                      size_vertical=0.15, fontproperties=fontprops)
        ax.add_artist(scalebar_s2)

# ==========================================
# 4. Bottom Row: REMA Processing
# ==========================================

print("Checking REMA strip...")
strip_path = os.path.join(PUB_REMA_DIR, f"{rema_strip_id}.tif")
search_bounds = tuple(bounds_gdf.to_crs(3031).total_bounds)

rema_date_str = rema_strip_id.split('_')[3]
rema_sensor = rema_strip_id.split('_')[2]
rema_labels = ['d', 'e', 'f', 'g']

if not os.path.exists(strip_path):
    print(f"Searching and downloading REMA strip {rema_strip_id}...")
    gdf_rema = pdt.search(
        dataset='rema',
        bounds=search_bounds,
        dates='20190918/20190920', 
        sensors=[rema_sensor],
        accuracy=2,
        min_aoi_frac=0.3
    )
    
    if gdf_rema is None or len(gdf_rema) == 0:
        raise ValueError("No REMA strips found in this date range and bounding box.")
        
    id_mask = gdf_rema.astype(str).apply(lambda col: col.str.contains(rema_strip_id)).any(axis=1)
    gdf_target = gdf_rema[id_mask]

    if len(gdf_target) == 0:
        raise ValueError(f"Found strips, but could not find the specific REMA strip ID: {rema_strip_id}")
        
    dem_da = pdt.load.from_search(gdf_target.iloc[0], bounds=search_bounds, bitmask=True)
    dem_da.rio.to_raster(strip_path, compress='lzw', predictor=3, zlevel=1)
else:
    print(f"REMA strip already exists at: {strip_path}")

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
xs, ys = transformer.transform([p[0] for p in rema_points], [p[1] for p in rema_points])
ls = LightSource(azdeg=315, altdeg=45)

for idx, (x, y) in enumerate(zip(xs, ys)):
    ax = bottom_axes[idx]
    
    out_hs_filename = os.path.join(PUB_REMA_DIR, f"rema_{rema_date_str}_hillshade_zoom_{idx+1}.tif")
    
    if os.path.exists(out_hs_filename):
        print(f"Loading existing REMA hillshade: {out_hs_filename}")
        with rasterio.open(out_hs_filename) as src:
            hs_uint8 = src.read(1)
    else:
        print(f"Processing REMA hillshade {idx+1}...")
        bbox = box(x - 150, y - 150, x + 150, y + 150)
        
        with rasterio.open(strip_path) as src:
            try:
                dem_crop, dem_transform = mask(src, [bbox], crop=True)
                dem_array = dem_crop[0]
                dem_masked = np.ma.masked_less_equal(dem_array, -9999) 
                
                hs = ls.hillshade(dem_masked, vert_exag=2, dx=2, dy=2)
                hs_uint8 = (hs * 255).astype(np.uint8)
                
                hs_meta = src.meta.copy()
                hs_meta.update({
                    "driver": "GTiff", "height": hs_uint8.shape[0], "width": hs_uint8.shape[1],
                    "transform": dem_transform, "count": 1, "dtype": 'uint8', "nodata": 0
                })
                
                with rasterio.open(out_hs_filename, "w", **hs_meta) as dest:
                    dest.write(hs_uint8, 1)
            except ValueError as e:
                print(f"Skipping point {idx+1} - Bounding box error: {e}")
                continue
                
    ax.imshow(hs_uint8, cmap='gray', extent=[x-150, x+150, y-150, y+150], interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.text(0.05, 0.95, rema_labels[idx], transform=ax.transAxes, fontsize=10, fontweight='bold', 
            va='top', ha='left', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
            
    # 3) Scalebar logic: Bottom left, only on panel "d"
    if idx == 0:
        scalebar_rema = AnchoredSizeBar(ax.transData, 100, '100 m', 'lower left', 
                                        pad=0.2, color='black', frameon=True, 
                                        size_vertical=4, fontproperties=fontprops)
        ax.add_artist(scalebar_rema)

    # 4) Draw the red bounding box on Panel "a" (top_axes[0])
    x_km = x / 1000.0
    y_km = y / 1000.0
    
    rect = patches.Rectangle((x_km - 0.15, y_km - 0.15), 0.3, 0.3, 
                             linewidth=1.2, edgecolor='red', facecolor='none')
    top_axes[0].add_patch(rect)
    
    top_axes[0].text(x_km + 0.18, y_km + 0.18, rema_labels[idx], 
                     color='red', fontsize=8.5, fontweight='bold', 
                     ha='left', va='bottom')

# ==========================================
# 5. Final Formatting and Save
# ==========================================
print(f"Saving final figure to {PLOT_DIR}...")
fig.savefig(os.path.join(PLOT_DIR, "overview.png"), dpi=600, bbox_inches='tight')
fig.savefig(os.path.join(PLOT_DIR, "overview.svg"), dpi=600, bbox_inches='tight')
fig.savefig(os.path.join(PLOT_DIR, "overview.pdf"), dpi=600, bbox_inches='tight')
plt.show()