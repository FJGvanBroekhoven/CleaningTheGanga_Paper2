# -*- coding: utf-8 -*-
"""
Author : Frank van Broekhoven
=======================
Prepare groundwater observation well data for the MODFLOW model of the
Ganges–Yamuna interfluve.

Overview
--------
Raw groundwater level measurements (metres below ground level, mbgl) from
the State Ground Water Department are processed into clean time series and
metadata files for use in model calibration. Processing steps:

    Step 1 – Load and reshape raw data
        The input Excel file contains one row per well with measurement dates
        as column headers. Data are melted to long format, dates parsed, and
        a unique well identifier assigned to each well.

    Step 2 – Build metadata and time series DataFrames
        A wide-format time series DataFrame (date × well_id) and a metadata
        GeoDataFrame (location, observation period, summary statistics) are
        derived from the long-format data.

    Step 3 – Extract surface elevation from DEM
        Well locations are reprojected to EPSG:32643 and sampled against the
        HydroSHEDS DEM to retrieve ground surface elevation [m MSL] at each
        well.

    Step 4 – Convert water levels from mbgl to m MSL
        Groundwater levels are converted from depth below ground level (mbgl)
        to elevation above mean sea level (m MSL) by subtracting the observed
        depth from the surface elevation:
            head_msl = elevation - depth_mbgl

    Step 5 – Filter wells by record length and export
        Monthly time series: wells with ≥ 50 observations are retained.
        Yearly time series : wells with ≥ 10 years of data are retained.
        Both time series and metadata are exported as CSV and GeoJSON.

References
----------
State Ground Water Department (n.d.) – Groundwater level monitoring data,
    Hindon catchment, Uttar Pradesh
"""

#%% =============================================================================
# Imports
# =============================================================================
import os
import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import Point


#%% =============================================================================
# Paths
# =============================================================================

main_dir = r'C:\GitHub\Temp_CleaningTheGanga_Paper2'

# --- Input ---
wells_path = os.path.join(main_dir, '1_model_input', '1_original_data', 'Observation_wells', 'State_GWD_Hindon_WL.xlsx')
dem_path   = os.path.join(main_dir, '1_model_input', '1_original_data', 'HydroSHEDS', 'HydroSHEDS_DEM_merged_cropped.tif')

# --- Output ---
outpath_meta         = os.path.join(main_dir, '1_model_input', '3_model_input_data', 'metadata_selected_observation_wells.geojson')
outpath_data         = os.path.join(main_dir, '1_model_input', '3_model_input_data', 'timeseries_data_selected_observation_wells.csv')
outpath_meta_yearly  = os.path.join(main_dir, '1_model_input', '3_model_input_data', 'metadata_selected_observation_wells_yearly.geojson')
outpath_data_yearly  = os.path.join(main_dir, '1_model_input', '3_model_input_data', 'timeseries_data_selected_observation_wells_yearly.csv')

#%% =============================================================================
# Step 1 – Load and reshape raw data
# =============================================================================

# --- Load Excel file ---
# Each row represents one well; date columns are identified by the '-' separator
# in the column header (e.g. 'Jan-2005').
df = pd.read_excel(wells_path)

# --- Assign unique well identifier ---
# A zero-padded index is used as the well_id to ensure consistent ordering
# and stable cross-referencing between metadata and time series files.
df['well_id'] = 'well_' + df.index.astype(str).str.zfill(3)

# --- Melt from wide to long format ---
date_cols = [col for col in df.columns if '-' in col]
long_df = df.melt(
    id_vars=['well_id', 'Y', 'X'],
    value_vars=date_cols,
    var_name='date',
    value_name='value',
)

# --- Parse dates and coerce non-numeric values to NaN ---
long_df['date']  = pd.to_datetime(long_df['date'], format='%b-%Y')
long_df['value'] = pd.to_numeric(long_df['value'], errors='coerce')


#%% =============================================================================
# Step 2 – Build metadata and time series DataFrames
# =============================================================================

# --- Wide-format monthly time series (date × well_id) [mbgl] ---
timeseries_df = long_df.pivot(index='date', columns='well_id', values='value')

# --- Metadata GeoDataFrame with summary statistics per well ---
long_df_valid = long_df.dropna(subset=['value'])
meta_df = (
    long_df_valid
    .groupby('well_id')
    .agg(
        y=('Y', 'first'),
        x=('X', 'first'),
        start_date=('date', 'min'),
        end_date=('date', 'max'),
        count_of_observations=('value', 'count'),
        min_value=('value', 'min'),
        max_value=('value', 'max'),
        mean_value=('value', 'mean'),
    )
    .reset_index()
)

# Build GeoDataFrame and reproject from geographic (EPSG:4326) to UTM 43N
geometry = [Point(xy) for xy in zip(meta_df.x, meta_df.y)]
meta_gdf = gpd.GeoDataFrame(meta_df, geometry=geometry, crs="EPSG:4326")
meta_gdf = meta_gdf.set_index('well_id').to_crs("EPSG:32643")


#%% =============================================================================
# Step 3 – Extract surface elevation from HydroSHEDS DEM
# =============================================================================

# Sample the DEM at each well location to obtain ground surface elevation
# [m MSL]. The DEM must share the CRS of the well locations (EPSG:32643).
with rasterio.open(dem_path) as src:
    if src.crs.to_epsg() != 32643:
        raise ValueError(f"DEM CRS ({src.crs}) does not match well CRS (EPSG:32643).")
    coords     = [(x, y) for x, y in zip(meta_gdf.geometry.x, meta_gdf.geometry.y)]
    elevations = [val[0] for val in src.sample(coords)]

meta_gdf['elevation'] = elevations
elevation_dict = meta_gdf['elevation'].to_dict()


#%% =============================================================================
# Step 4 – Convert water levels from mbgl to m MSL
# =============================================================================

# head_msl [m MSL] = surface_elevation [m MSL] − depth_to_water [mbgl]
# This conversion is applied to both the monthly and yearly time series.

# --- Monthly time series [m MSL] ---
timeseries_df_msl = timeseries_df.copy()
for well_id in timeseries_df.columns:
    if well_id in elevation_dict:
        timeseries_df_msl[well_id] = elevation_dict[well_id] - timeseries_df[well_id]

# --- Yearly time series: resample monthly to annual mean, then convert ---
timeseries_df_yearly = timeseries_df.resample('Y').mean()
timeseries_df_yearly_msl = timeseries_df_yearly.copy()
for well_id in timeseries_df_yearly.columns:
    if well_id in elevation_dict:
        timeseries_df_yearly_msl[well_id] = elevation_dict[well_id] - timeseries_df_yearly[well_id]

# --- Update metadata GeoDataFrame with yearly summary statistics ---
meta_yearly_gdf = meta_gdf.copy()
for well_id in meta_yearly_gdf.index:
    well_data = timeseries_df_yearly[well_id]
    meta_yearly_gdf.at[well_id, 'count_of_observations'] = well_data.count()
    meta_yearly_gdf.at[well_id, 'min_value']             = well_data.min()
    meta_yearly_gdf.at[well_id, 'max_value']             = well_data.max()
    meta_yearly_gdf.at[well_id, 'mean_value']            = well_data.mean()


#%% =============================================================================
# Step 5 – Filter wells by record length and export
# =============================================================================

# --- Selection thresholds ---
MIN_MONTHLY_OBS = 50   # minimum number of monthly observations to retain a well
MIN_YEARLY_OBS  = 10   # minimum number of yearly observations to retain a well

# --- Monthly: retain wells with at least MIN_MONTHLY_OBS observations ---
selection_meta         = meta_gdf[meta_gdf['count_of_observations'] >= MIN_MONTHLY_OBS]
selected_timeseries    = timeseries_df_msl[selection_meta.index]

# --- Yearly: retain wells with at least MIN_YEARLY_OBS years of data ---
selection_meta_yearly  = meta_yearly_gdf[meta_yearly_gdf['count_of_observations'] >= MIN_YEARLY_OBS]
selected_timeseries_yearly = timeseries_df_yearly_msl[selection_meta_yearly.index]

#%% --- Quick Plots ----

selection_meta.plot(column='count_of_observations', legend=True)
selection_meta_yearly.plot(column='count_of_observations', legend=True)

selected_timeseries.plot(marker='x')
selected_timeseries_yearly .plot(marker='x')


#%% --- Export ---
selection_meta.to_file(outpath_meta, driver='GeoJSON')
selected_timeseries.to_csv(outpath_data)
selection_meta_yearly.to_file(outpath_meta_yearly, driver='GeoJSON')
selected_timeseries_yearly.to_csv(outpath_data_yearly)

print(f"Monthly:  {len(selection_meta)} wells selected  → {outpath_meta}")
print(f"          time series          → {outpath_data}")
print(f"Yearly:   {len(selection_meta_yearly)} wells selected  → {outpath_meta_yearly}")
print(f"          time series          → {outpath_data_yearly}")
