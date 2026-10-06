import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import savgol_filter, find_peaks
from functools import lru_cache
import rasterio
import rioxarray as rxr
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point, Polygon
import cmocean
from cmcrameri import cm as cmc
import pdemtools as pdt
import glob
import re

# ==========================================
# 1. PATH DEFINITIONS
# ==========================================
BASE_DIR = "/mnt/parscratch/users/gg1bjd/research/swift/data"
RAW_VEL_S2 = f"{BASE_DIR}/raw/vel/sentinel2"
RAW_VEL_LS = f"{BASE_DIR}/raw/vel/landsat"
BOUNDS_SHP = f"{BASE_DIR}/raw/vector/bounds/bounds.shp"
FLOWLINE_SHP = f"{BASE_DIR}/raw/vector/flowline/flowline.shp"
#BEDMACHINE_NC = f"{BASE_DIR}/raw/bedmachine/NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc"
BEDMACHINE_NC = f"{BASE_DIR}/raw/bedmachine/BedMachineAntarctica-v3.nc"
REMA_OUT = f"{BASE_DIR}/raw/images/rema/rema_10m_geoid_masked.tif"
TERMINUS_SHP = f"{BASE_DIR}/raw/vector/terminus/swift_terminus.shp"
TERMINUS_CSV = f"{BASE_DIR}/raw/vector/terminus/swift.csv"
DZ_PUB_DIR = f"{BASE_DIR}/pub/dz"
ELEV_CSV_OUT = f"{DZ_PUB_DIR}/flowline_elevation_timeseries.csv"

PUB_VEL_DIR = f"{BASE_DIR}/pub/vel"
PUB_IMG_DIR = f"{BASE_DIR}/pub/images/sentinel1"
RESULTS_DIR = "/mnt/parscratch/users/gg1bjd/research/swift/results"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PUB_VEL_DIR, exist_ok=True)
os.makedirs(PUB_IMG_DIR, exist_ok=True)

PRE_MEAN_S = f"{PUB_VEL_DIR}/pre_landslide_mean_speed.tif"
PRE_MEAN_U = f"{PUB_VEL_DIR}/pre_landslide_mean_vx.tif"
PRE_MEAN_V = f"{PUB_VEL_DIR}/pre_landslide_mean_vy.tif"

S1_S_CHANGE_OUT = f"{PUB_VEL_DIR}/s1_speed_change_pct.tif"
S1_BG_OUT = f"{PUB_IMG_DIR}/s1_backscatter_cropped.tif"

S1_RAW_S = f"{BASE_DIR}/raw/vel/sentinel1/20180414_20180426_038_0811_F2_speed.tif"
S1_RAW_U = f"{BASE_DIR}/raw/vel/sentinel1/20180414_20180426_038_0811_F2_U.tif"
S1_RAW_V = f"{BASE_DIR}/raw/vel/sentinel1/20180414_20180426_038_0811_F2_V.tif"
S1_RAW_DB = f"{BASE_DIR}/raw/images/sentinel1/20180402_20180414_038_0811_F2_db.tif"

ZARR_ANTARCTICA = "/mnt/parscratch/users/gg1bjd/Data/Velocity/Antarctica/multisource_zarr/antarctica_multisource_velocity_optimized.zarr"

# Velocity file lists
SE2_list = [
    'S_50m_20160218_20160309', 'S_50m_20160119_20160309', 'S_50m_20160119_20160218',
    'S_50m_20160109_20160309', 'S_50m_20160109_20160218', 'S_50m_20180324_20180403'
]
LST_list = [
    'S_50m_20170824_20171027', 'S_50m_20170824_20170925', 'S_50m_20161209_20170211',
    'S_50m_20161209_20161225', 'S_50m_20161006_20161225', 'S_50m_20161006_20161209',
    'S_50m_20160922_20161008', 'S_50m_20160821_20160906', 'S_50m_20160108_20160328',
    'S_50m_20160101_20160218', 'S_50m_20160101_20160202', 'S_50m_20160101_20160117',
    'S_50m_20151216_20160218', 'S_50m_20151216_20160202', 'S_50m_20151216_20160117',
    'S_50m_20151216_20160101', 'S_50m_20151105_20160108', 'S_50m_20151029_20160117',
    'S_50m_20151029_20160101', 'S_50m_20151029_20151216', 'S_50m_20151020_20160108',
    'S_50m_20151020_20151105', 'S_50m_20150911_20151029', 'S_50m_20150826_20150911',
    'S_50m_20141001_20141102', 'S_50m_20140924_20141026', 'S_50m_20140908_20141026',
    'S_50m_20140908_20140924', 'S_50m_20140111_20140401', 'S_50m_20131210_20140111',
    'S_50m_20131030_20140118'
]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def load_and_resample(file_path, match_grid, nodata=-9999):
    da = rxr.open_rasterio(file_path, masked=True).squeeze()
    da = da.where(da != nodata)
    return da.rio.reproject_match(match_grid)

def generate_flowline_points(shp_path, spacing=500):
    gdf = gpd.read_file(shp_path)
    line_crs = gdf.crs if gdf.crs is not None else "EPSG:3031"
    line = gdf.geometry.iloc[0]
    distances = np.arange(0, line.length, spacing)
    points = [line.interpolate(distance) for distance in distances]
    return gpd.GeoDataFrame(geometry=points, crs=line_crs)

def hillshade(dem, azimuth=180, angle_altitude=45, vert_exag=8.0, dx=10.0, dy=10.0):
    ls = LightSource(azdeg=azimuth, altdeg=angle_altitude)
    return ls.hillshade(dem, vert_exag=vert_exag, dx=dx, dy=dy)

#ls = LightSource(azdeg=315, altdeg=45)

def apply_imageschs(data, background, cmap, norm, weight=0.5, alpha=1.0):
    """
    Replicates MATLAB's TopoToolbox / CDT 'imageschs' multiplicative blending.
    
    Parameters
    ----------
    data : 2D numpy array or DataArray
        The spatial overlay data (e.g. ice speed). NaNs display as pure background.
    background : 2D numpy array or DataArray
        The DEM hillshade array or SAR backscatter image.
    cmap : Matplotlib Colormap
    norm : Matplotlib Normalize object
    weight : float, default 0.5
        Controls texture contrast (0.2 = subtle texture, 0.7 = strong relief shadows).
    """
    data_arr = np.asarray(data)
    bg_arr = np.asarray(background)
    
    # 1. Normalize background grayscale map to [0, 1]
    bg_min, bg_max = np.nanmin(bg_arr), np.nanmax(bg_arr)
    bg_norm = (bg_arr - bg_min) / (bg_max - bg_min) if bg_max > bg_min else np.zeros_like(bg_arr)
    
    # 2. Map overlay data to RGB array
    rgb = cmap(norm(data_arr))[:, :, :3]
    
    # 3. Multiplicative shading factor (modulates light without touching hue/saturation)
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
        
    return np.clip(blended, 0, 1)


# ==========================================
# 3. ZARR EXTRACTION UTILITIES
# ==========================================
@lru_cache(maxsize=4)
def get_cached_timeseries_zarr(zarr_path):
    print(f"Opening and caching time-series Zarr store: {zarr_path}")
    return xr.open_zarr(zarr_path, consolidated=True).sortby('time')

def _load_input_to_gdf(loc_input):
    if isinstance(loc_input, gpd.GeoDataFrame):
        return loc_input.to_crs("EPSG:4326")
    raise ValueError("Unsupported input format.")

def get_multi_glacier_timeseries(location_input, buffer=500, sources=None, gap_fill=24, win_raw=25, win_daily=25, poly=2):
    results = {}
    gdf = _load_input_to_gdf(location_input)
    if gdf.empty: return {"error": "Input file contains no geometries."}

    try:
        ds = get_cached_timeseries_zarr(ZARR_ANTARCTICA)
    except Exception as e:
        return {"error": f"Could not open multi-source data store: {str(e)}"}

    for idx, row in gdf.iterrows():
        site_name = f"Site_{idx}"
        site_data = _process_single_site_multi(ds, row.geometry, "EPSG:3031", buffer, sources, gap_fill, win_raw, win_daily, poly)
        results[site_name] = site_data

    return results

def _process_single_site_multi(ds, geometry, target_crs, buffer, sources, gap_fill, win_raw, win_daily, poly):
    temp_gdf = gpd.GeoDataFrame({'geometry': [geometry]}, crs="EPSG:4326").to_crs(target_crs)
    proj_geom = temp_gdf.geometry.iloc[0]
    
    x_min, x_max = ds.x.min().item(), ds.x.max().item()
    y_min, y_max = ds.y.min().item(), ds.y.max().item()
    if y_min > y_max: y_min, y_max = y_max, y_min

    px, py = proj_geom.centroid.x, proj_geom.centroid.y
    if not (x_min <= px <= x_max) or not (y_min <= py <= y_max):
        return {"status": "error", "message": "Location outside data coverage."}
    
    is_single_pixel = False
    if isinstance(proj_geom, Point):
        if buffer <= 0: is_single_pixel = True 
        else: minx, miny, maxx, maxy = proj_geom.buffer(buffer).bounds
    else:
        if buffer > 0: proj_geom = proj_geom.buffer(buffer)
        minx, miny, maxx, maxy = proj_geom.bounds

    if not is_single_pixel:
        y_slice = slice(maxy, miny) if ds.y[0] > ds.y[-1] else slice(miny, maxy)
        try:
            subset = ds.sel(x=slice(minx, maxx), y=y_slice)
            if subset.x.size == 0 or subset.y.size == 0: is_single_pixel = True 
        except Exception: is_single_pixel = True

    if is_single_pixel:
        try: subset = ds.sel(x=proj_geom.centroid.x, y=proj_geom.centroid.y, method='nearest')
        except Exception as e: return {"status": "error", "message": f"Pixel selection failed: {e}"}
            
    if 'time_bnds' in subset.data_vars or 'time_bnds' in subset.coords:
        tb = subset['time_bnds']
        if len(tb.dims) >= 2:
            bnd_dim = [d for d in tb.dims if d != 'time'][0]
            t0, t1 = tb.isel({bnd_dim: 0}), tb.isel({bnd_dim: 1})
            dt_days = (t1 - t0) / np.timedelta64(1, 'D')
            subset = subset.assign(time_separation=dt_days)
        else:
            subset = subset.assign(time_separation=xr.full_like(subset['speed'], 12.0))
        subset = subset.drop_vars('time_bnds')
    else:
        subset = subset.assign(time_separation=xr.full_like(subset['speed'], 12.0))

    if is_single_pixel:
        df = subset[['speed', 'error', 'data_source', 'time_separation']].to_dataframe()
        df['valid_count'] = subset['speed'].notnull().astype(int).to_series()
    else:
        subset_df = subset[['speed', 'error', 'data_source', 'time_separation']].to_dataframe().reset_index()
        df = subset_df.groupby('time').agg({
            'speed': 'median', 'error': 'median', 'data_source': 'first', 'time_separation': 'first'
        })
        df['valid_count'] = subset_df.groupby('time')['speed'].count()

    if sources is not None and len(sources) > 0:
        df = df[df['data_source'].astype(str).isin(sources)]
        
    if df.empty or df['speed'].dropna().empty:
        return {"status": "error", "message": "No valid data or all selected sources masked/NaN"}
    
    df['time_separation'] = df['time_separation'] - 1.0
    df['time_separation'] = df['time_separation'].apply(lambda x: x if x > 0 else 0.5).fillna(12.0)
    df = df.sort_index()
    if df.index.duplicated().any(): df = df.groupby(level=0).first()

    df.loc[(df['speed'] < -100) | (df['speed'] > 100000), 'speed'] = np.nan
    rolling_mean = df['speed'].rolling(window=5, center=True, min_periods=1).mean()
    rolling_std = df['speed'].rolling(window=5, center=True, min_periods=1).std()
    fallback_std = df['speed'].std()
    if pd.isna(fallback_std) or fallback_std == 0: fallback_std = 1.0
    rolling_std = rolling_std.fillna(fallback_std).replace(0, fallback_std)
    outliers = (df['speed'] - rolling_mean).abs() > (3 * rolling_std)
    df.loc[outliers, 'speed'] = np.nan

    exact_idx = df.index
    daily_idx = pd.date_range(start=exact_idx.min().floor('D'), end=exact_idx.max().ceil('D'), freq='D')
    full_idx = exact_idx.union(daily_idx).sort_values()
    df_daily = df.reindex(full_idx) 
    valid_dates_mask = df_daily['speed'].notnull()
    
    def clean_nans(data_series):
        if hasattr(data_series, 'values'): data_series = data_series.values 
        if len(data_series) == 0: return []
        return [x if (pd.notnull(x) and (isinstance(x, str) or np.isfinite(x))) else None for x in data_series]

    output_data = {
        "dates": full_idx.strftime('%Y-%m-%dT%H:%M:%S').tolist(), 
        "error": clean_nans(np.round(df_daily['error'].astype(float), 2)),
        "dt": clean_nans(np.round(df_daily['time_separation'].astype(float), 1)),
        "data_source": clean_nans(df_daily['data_source']), 
        "count": df_daily['valid_count'].fillna(0).astype(int).tolist()
    }

    current_speed_series = df['speed']
    daily_temp = current_speed_series.reindex(full_idx)
    daily_filled = daily_temp.interpolate(method='time', limit=gap_fill)
    processed_raw_series = df_daily['speed'] 
    
    try:
        temp_series = daily_filled.interpolate(method='time', limit_direction='both')
        curr_len = len(temp_series)
        effective_window = win_raw if curr_len >= win_raw else curr_len
        if effective_window % 2 == 0: effective_window -= 1 
        if effective_window >= 3:
            smoothed_values = savgol_filter(temp_series.values, window_length=effective_window, polyorder=poly)
            processed_raw_series = pd.Series(smoothed_values, index=full_idx).where(valid_dates_mask)
    except Exception: pass

    capped_separation = df['time_separation'].clip(upper=gap_fill)
    time_sep_days = pd.to_timedelta(capped_separation, unit='D')
    starts = df.index - (time_sep_days / 2)
    ends   = df.index + (time_sep_days / 2)
    
    daily_stack = []
    for i in range(len(df)):
        if pd.isna(current_speed_series.iloc[i]): continue 
        s_date, e_date = starts.iloc[i].floor('D'), ends.iloc[i].ceil('D')
        val_to_use = current_speed_series.iloc[i]
        try:
            if pd.notnull(processed_raw_series.loc[df.index[i]]): val_to_use = processed_raw_series.loc[df.index[i]]
        except: pass
        dt_val = df['time_separation'].iloc[i]
        if pd.isna(dt_val) or dt_val < 1: dt_val = 1.0 
        weight_val = 1.0 / dt_val
        date_rng = pd.date_range(start=s_date, end=e_date, freq='D')
        if not date_rng.empty:
            daily_stack.append(pd.DataFrame({'date': date_rng, 'speed': val_to_use, 'weight': weight_val}))

    if daily_stack:
        big_df = pd.concat(daily_stack)
        big_df['weighted_speed'] = big_df['speed'] * big_df['weight']
        grouped = big_df.groupby('date')
        daily_ts = (grouped['weighted_speed'].sum() / grouped['weight'].sum()).reindex(full_idx)
    else:
        daily_ts = pd.Series(dtype=float, index=full_idx)

    daily_ts_filled = daily_ts.interpolate(method='time', limit=gap_fill)
    daily_final = daily_ts.copy() 
    try:
        temp_series_daily = daily_ts_filled.interpolate(method='time', limit_direction='both')
        curr_len_d = len(temp_series_daily)
        eff_win_daily = win_daily if curr_len_d >= win_daily else curr_len_d
        if eff_win_daily % 2 == 0: eff_win_daily -= 1
        if eff_win_daily >= 3:
            smooth_vals_daily = savgol_filter(temp_series_daily.values, window_length=eff_win_daily, polyorder=poly)
            daily_final = pd.Series(smooth_vals_daily, index=full_idx)
            daily_final[daily_ts_filled.isna()] = np.nan
    except Exception: pass
        
    output_data['speed'] = {
        "raw": clean_nans(np.round(processed_raw_series.astype(float), 2)), 
        "smoothed": clean_nans(np.round(daily_final.astype(float), 2))            
    }

    return {"status": "success", "data": output_data}


# ==========================================
# 4. PROCESSING (MAPS)
# ==========================================
bounds_gdf = gpd.read_file(BOUNDS_SHP)

# --- A. Master Grid Setup ---
master_ref = rxr.open_rasterio(f"{RAW_VEL_S2}/S/{SE2_list[0]}.tif").squeeze().drop_vars("band", errors="ignore")
master_grid = master_ref.rio.clip(bounds_gdf.geometry, bounds_gdf.crs)

# --- B. Process or Load Pre-Landslide Mean Velocities & DEM ---
if os.path.exists(PRE_MEAN_S) and os.path.exists(REMA_OUT):
    print("Loading cached pre-landslide means and DEM...")
    pre_S = rxr.open_rasterio(PRE_MEAN_S).squeeze()
    pre_U = rxr.open_rasterio(PRE_MEAN_U).squeeze()
    pre_V = rxr.open_rasterio(PRE_MEAN_V).squeeze()
    rema_dem = rxr.open_rasterio(REMA_OUT).squeeze()
else:
    print("Processing pre-landslide velocities and DEM...")
    bounds_extent = tuple(bounds_gdf.total_bounds) 
    
    rema = pdt.load.mosaic(
        dataset='rema',
        resolution=10,
        bounds=bounds_extent
    ) 
    
    geoid = pdt.data.geoid_from_bedmachine(BEDMACHINE_NC, rema)
    rema_geoid = rema.pdt.geoid_correct(geoid)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rema_masked = rema_geoid.pdt.mask_ocean(near_sealevel_thresh_m=5)
    
    rema_dem = rema_masked.rio.reproject_match(master_grid)
    rema_dem.rio.to_raster(REMA_OUT)

    stacks = {'S': [], 'U': [], 'V': []}
    for var in ['S', 'U', 'V']:
        for f in SE2_list:
            fpath = f"{RAW_VEL_S2}/{var}/{f.replace('S_', f'{var}_')}.tif"
            if os.path.exists(fpath): stacks[var].append(load_and_resample(fpath, master_grid))
        for f in LST_list:
            fpath = f"{RAW_VEL_LS}/{var}/{f.replace('S_', f'{var}_')}.tif"
            if os.path.exists(fpath): stacks[var].append(load_and_resample(fpath, master_grid))
            
    elev_mask = (rema_dem <= 1100) & (rema_dem >= 5)
    
    # FIX: Replaced trim_mean with np.nanmedian to handle NaNs properly while rejecting outliers
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning) # Ignore all-NaN slice warnings
        pre_S = xr.DataArray(np.nanmedian(np.stack(stacks['S']), axis=0), coords=master_grid.coords, dims=['y', 'x']).where(elev_mask)
        pre_U = xr.DataArray(np.nanmedian(np.stack(stacks['U']), axis=0), coords=master_grid.coords, dims=['y', 'x']).where(elev_mask)
        pre_V = xr.DataArray(np.nanmedian(np.stack(stacks['V']), axis=0), coords=master_grid.coords, dims=['y', 'x']).where(elev_mask)

    pre_S.rio.to_raster(PRE_MEAN_S)
    pre_U.rio.to_raster(PRE_MEAN_U)
    pre_V.rio.to_raster(PRE_MEAN_V)

# --- C. Process or Load Post-Landslide Speed Change & S1 DB ---
if os.path.exists(S1_S_CHANGE_OUT) and os.path.exists(S1_BG_OUT):
    print("Loading cached S1 speed change and backscatter...")
    s1_change_pct = rxr.open_rasterio(S1_S_CHANGE_OUT).squeeze()
    s1_db = rxr.open_rasterio(S1_BG_OUT).squeeze()
else:
    print("Processing S1 speed change and backscatter...")
    s1_s = load_and_resample(S1_RAW_S, master_grid)
    s1_db = load_and_resample(S1_RAW_DB, master_grid)
    
    s1_change_pct = ((s1_s - pre_S) / pre_S) * 100
    
    s1_change_pct.rio.to_raster(S1_S_CHANGE_OUT)
    s1_db.rio.to_raster(S1_BG_OUT)

# ==========================================
# 5. PROCESSING (TIMESERIES)
# ==========================================
flowline_pts = generate_flowline_points(FLOWLINE_SHP, spacing=250)
num_pts = len(flowline_pts)
cmap_amp = cmocean.cm.amp
colors_flowline = [cmap_amp(i / num_pts) for i in range(num_pts)]

# --- Define Global Temporal Bounds ---
start_date = pd.to_datetime('2018-01-01')
end_date = pd.to_datetime('2019-07-31')
landslide_date = pd.to_datetime('2018-04-14 06:38:32')

# --- Process / Load Terminus Data ---
print("Loading terminus vector and CSV data...")
term_gdf = gpd.read_file(TERMINUS_SHP)
if term_gdf.crs and term_gdf.crs.to_epsg() == 4326:
    term_gdf = term_gdf.to_crs(epsg=3031)

# Clip the terminus shapefile to the bounds shapefile
bounds_gdf_proj = bounds_gdf.to_crs(term_gdf.crs)
term_gdf = gpd.clip(term_gdf, bounds_gdf_proj)

term_gdf['Date'] = pd.to_datetime(term_gdf['Date'])
# Filter shapefile geometries to our specific date window
term_gdf = term_gdf[(term_gdf['Date'] >= start_date) & (term_gdf['Date'] <= end_date)]
term_gdf['date_num'] = mdates.date2num(term_gdf['Date'])

term_df = pd.read_csv(TERMINUS_CSV)
term_df['Date'] = pd.to_datetime(term_df[['Year', 'Month', 'Day']])
term_df['date_num'] = mdates.date2num(term_df['Date'])

# Create unified colormap norm for both the shapefile and CSV
#vmin_date = min(term_gdf['date_num'].min(), term_df['date_num'].min())
#vmax_date = max(term_gdf['date_num'].max(), term_df['date_num'].max())
vmin_date = mdates.date2num(start_date)
vmax_date = mdates.date2num(end_date)
date_norm = Normalize(vmin=vmin_date, vmax=vmax_date)
date_cmap = plt.cm.viridis

TS_CSV_OUT = f"{PUB_VEL_DIR}/flowline_timeseries.csv"

if os.path.exists(TS_CSV_OUT):
    print(f"Loading cached timeseries from CSV: {TS_CSV_OUT}")
    df = pd.read_csv(TS_CSV_OUT)
    ts_results = {}
    for site in df['site_id'].unique():
        site_df = df[df['site_id'] == site]
        ts_results[site] = {
            "status": "success",
            "data": {
                "dates": site_df['date'].tolist(),
                "speed": {
                    "raw": site_df['raw_speed'].tolist(),
                    "smoothed": site_df['smoothed_speed'].tolist()
                },
                "error": site_df['error'].tolist()
            }
        }
else:
    print("Extracting timeseries from Zarr store (this may take a moment)...")
    ts_results = get_multi_glacier_timeseries(
        location_input=flowline_pts, 
        buffer=500, 
        sources=["SHIFT"]
    )
    
    print(f"Saving timeseries to CSV: {TS_CSV_OUT}")
    rows = []
    for site, res in ts_results.items():
        if res.get('status') == 'success':
            d = res["data"]
            for j in range(len(d["dates"])):
                rows.append({
                    'date': d["dates"][j],
                    'site_id': site,
                    'raw_speed': d["speed"]["raw"][j],
                    'smoothed_speed': d["speed"]["smoothed"][j],
                    'error': d["error"][j]
                })
    pd.DataFrame(rows).to_csv(TS_CSV_OUT, index=False)
    
    
# Elevation timeseries
if os.path.exists(ELEV_CSV_OUT):
    print(f"Loading cached elevation timeseries from CSV: {ELEV_CSV_OUT}")
    elev_df = pd.read_csv(ELEV_CSV_OUT)
    elev_df['Date'] = pd.to_datetime(elev_df['Date'])
else:
    print("Extracting elevation timeseries from REMA DEM strips...")
    rema_dir = f"{BASE_DIR}/raw/images/rema"
    rema_files = glob.glob(os.path.join(rema_dir, "SETSM_*.tif"))
    
    rows = []
    for fpath in rema_files:
        fname = os.path.basename(fpath)
        # Parse date (YYYYMMDD) from filename
        match = re.search(r'(\d{8})', fname)
        if not match:
            continue
        date_str = match.group(1)
        file_date = pd.to_datetime(date_str, format='%Y%m%d')

        try:
            with rxr.open_rasterio(fpath, masked=True) as da:
                da = da.squeeze()
                # Ensure CRS alignment with flowline points
                if da.rio.crs != flowline_pts.crs:
                    da = da.rio.reproject(flowline_pts.crs)
                
                for i, pt in enumerate(flowline_pts.geometry):
                    site_name = f"Site_{i}"
                    # Define 20x20m box (10m buffer in each direction)
                    x_min, x_max = pt.x - 10, pt.x + 10
                    y_min, y_max = pt.y - 10, pt.y + 10
                    
                    try:
                        sub = da.rio.clip_box(minx=x_min, miny=y_min, maxx=x_max, maxy=y_max)
                        mean_elev = float(sub.mean(skipna=True).values)
                        if not np.isnan(mean_elev):
                            rows.append({
                                'site_id': site_name,
                                'site_idx': i,
                                'distance_m': i * 250,
                                'Date': file_date,
                                'elevation': mean_elev
                            })
                    except Exception:
                        continue
        except Exception as e:
            print(f"Warning: Could not process {fname}: {e}")

    elev_df = pd.DataFrame(rows)
    
    if not elev_df.empty:
        # Sort chronologically per site and calculate elevation change relative to first measurement
        elev_df = elev_df.sort_values(['site_id', 'Date']).reset_index(drop=True)
        elev_df['elevation_change'] = elev_df.groupby('site_id')['elevation'].transform(lambda x: x - x.iloc[0])
        elev_df.to_csv(ELEV_CSV_OUT, index=False)
        print(f"Saved elevation timeseries to CSV: {ELEV_CSV_OUT}")
    else:
        print("No valid elevation data found across the provided DEMs.")

# ==========================================
# 6. PLOTTING
# ==========================================
print("Generating figure...")
fig = plt.figure(figsize=(7.5, 7), dpi=600, layout='constrained')

# 6 rows x 2 columns setup
gs = GridSpec(
    6, 2, 
    figure=fig, 
    width_ratios=[1, 2],    # Timeseries col is 2x width of Map col
    wspace=0.2,            # Space between cols for vertical colorbars
    hspace=0.1             # Space between rows
)

# Left Column: Maps (each spans 3 rows)
ax1 = fig.add_subplot(gs[0:3, 0])  # Panel A
ax2 = fig.add_subplot(gs[3:6, 0])  # Panel B

# Right Column: Timeseries (each spans 2 rows)
ax3 = fig.add_subplot(gs[0:2, 1])  # Panel C
ax4 = fig.add_subplot(gs[2:4, 1])  # Panel D
ax5 = fig.add_subplot(gs[4:6, 1])  # Panel E

extent = [master_grid.x.min(), master_grid.x.max(), master_grid.y.min(), master_grid.y.max()]

# --- Axes Formatting Helper for Panels A & B ---
def format_map_axes(ax):
    # Convert map coordinates (meters) to kilometers for the tick labels
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x/1000:.0f}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{y/1000:.0f}"))
    ax.set_xlabel("Easting (km)", fontsize=8)
    ax.set_ylabel("Northing (km)", fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=8)

# --- Panel A: Top Left (Pre-landslide Velocity Map) ---
hs = hillshade(rema_dem.values)

# Extract colors starting at 15% (skips the pitch black) to 100%
batlow_colors = cmc.batlow(np.linspace(0.15, 1.0, 256))
batlow_light = LinearSegmentedColormap.from_list('batlow_light', batlow_colors)

# Apply imageschs multiplicative blend
blended_map_a = apply_imageschs(
    pre_S.values, 
    hs, 
    cmap=batlow_light, 
    norm=Normalize(vmin=0, vmax=150), 
    weight=0.55,  # Tune between 0.3 (subtle) and 0.7 (punchy)
    alpha=0.85
)

ax1.imshow(blended_map_a, extent=extent, origin='upper')

sm1 = plt.cm.ScalarMappable(cmap=cmc.batlow, norm=Normalize(vmin=0, vmax=150))
sm1.set_array([])
cax1 = inset_axes(ax1, width="6%", height="60%", loc='center left',
                  bbox_to_anchor=(1.05, 0.5, 1, 0.5), bbox_transform=ax1.transAxes, borderpad=0)
cbar1 = plt.colorbar(sm1, cax=cax1, orientation='vertical')
cbar1.set_label('Speed (m yr$^{-1}$)', fontsize=8)
cbar1.ax.tick_params(labelsize=7)

# Overlay terminus lines with date color coding
for idx, row in term_gdf.iterrows():
    geom = row.geometry
    c = date_cmap(date_norm(row.date_num))
    if geom.geom_type == 'MultiLineString':
        for line in geom.geoms:
            ax1.plot(*line.xy, color=c, linewidth=0.25, zorder=6)
    else:
        ax1.plot(*geom.xy, color=c, linewidth=1.5, zorder=6)

# Create Date Colorbar on the LEFT side of panel A so it doesn't clash with the speed colorbar
sm_date = plt.cm.ScalarMappable(cmap=date_cmap, norm=date_norm)
sm_date.set_array([])
cax_date = inset_axes(ax1, width="6%", height="60%", loc='center left',
                      bbox_to_anchor=(1.05, 0.1, 1, 0.5), bbox_transform=ax1.transAxes, borderpad=0)
cbar_date = plt.colorbar(sm_date, cax=cax_date, orientation='vertical')
cbar_date.set_label('Terminus Date', fontsize=8)
cbar_date.ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#cax_date.yaxis.set_ticks_position('left')
#cax_date.yaxis.set_label_position('left')
cbar_date.ax.tick_params(labelsize=7)

step = 7 
X, Y = np.meshgrid(pre_U.x, pre_U.y)
ax1.quiver(X[::step, ::step], Y[::step, ::step], pre_U.values[::step, ::step], pre_V.values[::step, ::step], 
           color='k', alpha=0.7, scale=2500, width=0.003)

for i, point in enumerate(flowline_pts.geometry):
    ax1.plot(point.x, point.y, marker='o', markersize=3, color=colors_flowline[i], markeredgecolor='k', markeredgewidth=0.3)

format_map_axes(ax1)

sb_len = 2000  
sb_x = extent[0] + (extent[1] - extent[0]) * 0.05
sb_y = extent[2] + (extent[3] - extent[2]) * 0.05
ax1.plot([sb_x, sb_x + sb_len], [sb_y, sb_y], color='k', linewidth=2, solid_capstyle='butt')
ax1.text(sb_x + sb_len/2, sb_y + (extent[3] - extent[2]) * 0.02, '2 km', 
         color='k', fontsize=8, ha='center', va='bottom', fontweight='bold')
ax1.set_xlim([extent[0], extent[1]])
ax1.set_ylim([extent[2], extent[3]])

# --- Panel B: Top Right (Speed Change Map over SAR Backscatter) ---
blended_map_b = apply_imageschs(
    s1_change_pct.values, 
    s1_db.values, 
    cmap=cmocean.cm.balance, 
    norm=Normalize(vmin=-750, vmax=750), 
    weight=0.55,
    alpha=0.75
)

ax2.imshow(blended_map_b, extent=extent, origin='upper')

sm2 = plt.cm.ScalarMappable(cmap=cmocean.cm.balance, norm=Normalize(vmin=-750, vmax=750))
sm2.set_array([])
cax2 = inset_axes(ax2, width="6%", height="60%", loc='center left',
                  bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax2.transAxes, borderpad=0)
cbar2 = plt.colorbar(sm2, cax=cax2, orientation='vertical', extend='both')
cbar2.set_label('Speed Change (%)', fontsize=8)
cbar2.ax.tick_params(labelsize=7)

format_map_axes(ax2)

# --- Panel C: Bottom (Timeseries) ---
for i, idx in enumerate(flowline_pts.index):
    site_name = f"Site_{idx}"
    site_data = ts_results.get(site_name)
    
    if site_data and site_data.get("status") == "success":
        data = site_data["data"]
        
        dates = pd.to_datetime(data["dates"])
        raw_speed = np.array(data["speed"]["raw"], dtype=float)
        smoothed_speed = np.array(data["speed"]["smoothed"], dtype=float)
        error = np.array(data["error"], dtype=float)
        
        valid_raw = ~np.isnan(raw_speed)
        valid_smooth = ~np.isnan(smoothed_speed)

        ax3.errorbar(dates[valid_raw], raw_speed[valid_raw], yerr=error[valid_raw], 
                     fmt='o', markersize=2, color=colors_flowline[i], alpha=0.3, elinewidth=0.5)
        ax3.plot(dates[valid_smooth], smoothed_speed[valid_smooth], 
                 color=colors_flowline[i], linewidth=1)
        
# --- Kinematic Wave Analysis ---
peak_dates = []
peak_speeds = []
distances = []
site_ids = []

# Define search window to isolate the landslide-induced wave
wave_search_start = pd.to_datetime('2018-04-06')
wave_search_end = pd.to_datetime('2018-05-15')
wave_search_start = pd.to_datetime('2018-03-15')
wave_search_end = pd.to_datetime('2018-07-15')

for i, idx in enumerate(flowline_pts.index):
    site_name = f"Site_{idx}"
    site_data = ts_results.get(site_name)
    
    if site_data and site_data.get("status") == "success":
        dates = pd.to_datetime(site_data["data"]["dates"])
        smoothed = np.array(site_data["data"]["speed"]["smoothed"], dtype=float)
        
        # Mask using the newly defined isolated search window
        mask = (dates >= wave_search_start) & (dates <= wave_search_end) & (~np.isnan(smoothed))
        
        if np.any(mask):
            valid_dates = dates[mask]
            valid_smoothed = smoothed[mask]
            
            #max_idx = np.argmax(valid_smoothed)
            #peak_date = valid_dates[max_idx]
            peaks, properties = find_peaks(valid_smoothed, prominence=10)
            
            if len(peaks) > 0:
                # If multiple peaks exist in the window, pick the most prominent one
                #best_peak_idx = peaks[np.argmax(properties["prominences"])]
                this_peak_speeds = valid_smoothed[peaks]
                best_peak_idx = peaks[np.argmax(this_peak_speeds)]
                
                peak_dates.append(valid_dates[best_peak_idx])
                peak_speeds.append(valid_smoothed[best_peak_idx])
            
            # Check if the peak is hitting the edge of our search window
            #if peak_date == valid_dates.min() or peak_date == valid_dates.max():
            #    print(f"Warning: {site_name} peak hit the search boundary ({peak_date}). Discarding.")
            #    continue # Skip this site
                
            #peak_dates.append(peak_date)
            #peak_speeds.append(valid_smoothed[max_idx])
            distances.append(i * 250)
            site_ids.append(site_name)

if len(peak_dates) > 1:
    # 1. Line of best fit for Panel C (Amplitude vs Date)
    x_mdates = mdates.date2num(peak_dates)
    p_amp = np.polyfit(x_mdates, peak_speeds, 1)
    
    # Plot dashed grey line through the peaks
    x_line = np.linspace(min(x_mdates), max(x_mdates), 100)
    y_line = np.polyval(p_amp, x_line)
    ax3.plot(mdates.num2date(x_line), y_line, color='grey', linestyle='--', linewidth=2, zorder=15)
    
    # 2. Calculate Kinematic Wave Speed (Distance vs Time)
    base_date = min(peak_dates)
    days_since = np.array([(d - base_date).total_seconds() / (24*3600) for d in peak_dates])
    
    # Linear fit: Distance (m) = Wave_Speed (m/day) * Time (days) + C
    p_dist = np.polyfit(days_since, distances, 1)
    wave_speed_m_day = p_dist[0]
    
    # Print metrics to console and add a clean text box to the plot
    print(f"\n--- Kinematic Wave Metrics ---")
    print(f"Wave Speed: {wave_speed_m_day:.1f} m/day")
    
    # Place text anchor relative to the upper-most right-most end of the line
    line_end_date = mdates.num2date(x_line[-1])
    line_end_speed = y_line[-1]
    
    text_str = f"Wave Speed: {wave_speed_m_day:.1f} m day$^{{-1}}$"
    ax3.text(0.98, 0.95, text_str, transform=ax3.transAxes, fontsize=7, 
             ha='right', va='top', zorder=20)
    
    WAVE_METRICS_CSV = f"{PUB_VEL_DIR}/wave_metrics.csv"
    print(f"Saving wave metrics to CSV: {WAVE_METRICS_CSV}")
    metrics_df = pd.DataFrame({
        'site_id': site_ids,
        'distance_m': distances,
        'peak_date': peak_dates,
        'peak_speed': peak_speeds
    })
    metrics_df.to_csv(WAVE_METRICS_CSV, index=False)

plot_landslide_date = landslide_date - pd.Timedelta(days=5)
ax3.axvline(plot_landslide_date, color='k', linestyle='--', linewidth=1.5, zorder=10)
ax3.text(plot_landslide_date - pd.Timedelta(days=3), ax3.get_ylim()[1], 'Landslide', 
         rotation=90, va='top', ha='right', fontsize=8)

ax3.set_xlim([start_date, end_date])
ax3.set_ylabel("Speed (m yr$^{-1}$)", fontsize=8)
ax3.tick_params(axis='both', which='major', labelsize=8)
ax3.tick_params(axis='x', which='major', labelsize=8, labelrotation=45)
ax3.grid(True, linestyle=':', alpha=0.6)

# --- Panel D: Bottom (Terminus Relative Position Timeseries) ---
# Sort by date so the line connects properly chronologically
term_df_sorted = term_df.sort_values(by='Date')

# Connecting line underneath
ax4.plot(term_df_sorted['Date'], term_df_sorted['Terminus position relative to most recent observation (m)'],
         color='grey', linewidth=0.8, linestyle='-', zorder=2)

# Scatter points coloured by the exact date (matching Panel A colormap)
ax4.scatter(term_df_sorted['Date'], term_df_sorted['Terminus position relative to most recent observation (m)'],
            c=term_df_sorted['date_num'], cmap=date_cmap, norm=date_norm, 
            s=25, edgecolor='k', linewidth=0.4, zorder=3)

# Adding the landslide marker to panel d for consistency
ax4.axvline(plot_landslide_date, color='k', linestyle='--', linewidth=1.5, zorder=10)
ax4.text(plot_landslide_date - pd.Timedelta(days=3), 750, 'Landslide', 
         rotation=90, va='top', ha='right', fontsize=8)

ax4.set_xlim([start_date, end_date])  # Or comment this line to let ax4 autoscale the date range
ax4.set_ylim([0, 750])
ax4.set_ylabel("Rel. Terminus\nPosition (m)", fontsize=8)
ax4.tick_params(axis='both', which='major', labelsize=8)
ax4.tick_params(axis='x', which='major', labelsize=8, labelrotation=45)
ax4.grid(True, linestyle=':', alpha=0.6)

# --- Panel E: Bottom (Elevation Change Timeseries) ---
if 'elev_df' in locals() and not elev_df.empty:
    for i, idx in enumerate(flowline_pts.index):
        site_name = f"Site_{idx}"
        site_elev = elev_df[elev_df['site_id'] == site_name].sort_values('Date')
        
        if not site_elev.empty:
            ax5.plot(site_elev['Date'], site_elev['elevation_change'], 
                     color=colors_flowline[i], linewidth=1, marker='o', markersize=2.5, alpha=0.8)

ax5.axvline(plot_landslide_date, color='k', linestyle='--', linewidth=1.5, zorder=10)
# Dynamic y position for landslide text
y_max_e = ax5.get_ylim()[1] if not elev_df.empty else 1
ax5.text(plot_landslide_date - pd.Timedelta(days=3), y_max_e, 'Landslide', 
         rotation=90, va='top', ha='right', fontsize=8)

#ax5.set_xlim([start_date, end_date]) # comment to show full timeseriies/uncomment to clip to Jan-18 to Jul-19
ax5.set_ylabel("$\Delta z$ (m)", fontsize=8)
ax5.tick_params(axis='both', which='major', labelsize=8)
ax5.grid(True, linestyle=':', alpha=0.6)

for ax, letter in zip([ax1, ax2, ax3, ax4, ax5], ['a', 'b', 'c', 'd', 'e']):
    ax.text(0.02, 0.95, letter, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

output_base = f"{RESULTS_DIR}/response"
for ext in ['png', 'pdf', 'svg']:
    plt.savefig(f"{output_base}.{ext}", bbox_inches='tight', dpi=600)
