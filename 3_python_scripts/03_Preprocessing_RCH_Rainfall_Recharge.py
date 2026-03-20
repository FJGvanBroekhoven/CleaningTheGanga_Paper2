# -*- coding: utf-8 -*-
"""
Author : Frank van Broekhoven
=======================
Derive spatially and temporally explicit rainfall recharge from the 
CRU TS dataset for the MODFLOW model of the Ganges–Yamuna interfluve, 
covering 1800–2024 at monthly resolution.
 
Overview
--------
Rainfall recharge is computed from the CRU TS v4.09 precipitation (PRE) and
potential evapotranspiration (PET) datasets (Harris et al., 2020) and
resampled to the 1 km × 1 km model grid using bilinear interpolation.
 
The water balance partitioning follows:
    P   = precipitation [mm/month]
    RR  = rainfall runoff = P × runoff_factor
    RCH = max(P − RR − PET, 0)   [only non-negative recharge retained]
    AET = P − RR − RCH            [actual evapotranspiration as residual]
 
All variables are converted from mm/month to m³/month for model input.
 
For the period 1800–1900 (before CRU TS coverage), rainfall recharge is
extended using the long-term monthly average of the full CRU TS record,
assuming no significant pre-1900 climate shifts.
 
References
----------
Harris et al. (2020) – CRU TS v4: An improved version of the gridded climate
    dataset. Scientific Data, 7, 109.
 
"""
 
#%% =============================================================================
# Imports
# =============================================================================
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
import os
 
 
#%% =============================================================================
# Paths
# =============================================================================

main_dir = r'C:\GitHub\Temp_CleaningTheGanga_Paper2'
 
# --- Input ---
mask_path     = os.path.join(main_dir, '1_model_input', '1_original_data', 'Model_Mask.tif')
PRE_path      = os.path.join(main_dir, '1_model_input', '1_original_data', 'CRUTS_v4_09', 'cru_ts4.09.1901.2024.pre.dat.nc')
PET_path      = os.path.join(main_dir, '1_model_input', '1_original_data', 'CRUTS_v4_09', 'cru_ts4.09.1901.2024.pet.dat.nc')

# --- Output ---
output_path   = os.path.join(main_dir, '1_model_input', '2_temp_data', 'CRUTS_derived_rainfall_recharge.nc')


#%% =============================================================================
# Step 1 – Load and prepare CRU TS data
# =============================================================================
 
# --- Load precipitation [mm/month] and PET [mm/day → mm/month] ---
# CRU TS PRE is already in mm/month; PET is in mm/day and is converted by
# multiplying by the average number of days per month (365.25 / 12).
P   = xr.open_dataset(PRE_path)['pre']              # [mm/month]
PET = xr.open_dataarray(PET_path) * (365.25 / 12)   # [mm/day] → [mm/month]
 
CRUTS = xr.Dataset({'P': P, 'PET': PET})
 
# --- Adjust time coordinate to the first of each month ---
# CRU TS timestamps are set to the 15th or 16th of each month; these are
# shifted to the 1st for consistency with other model input datasets.
time_coords          = CRUTS['time'].to_index()
adjusted_time_coords = time_coords.to_period('M').to_timestamp()
CRUTS                = CRUTS.assign_coords(time=adjusted_time_coords)
 
 
#%% =============================================================================
# Step 2 – Reproject and resample to model grid
# =============================================================================
 
# --- Load model mask (defines target CRS, extent and resolution) ---
mask_da = xr.open_dataarray(mask_path)
mask_da.rio.write_crs("EPSG:32643", inplace=True)
 
# --- Reproject CRU TS from geographic (EPSG:4326) to model CRS (EPSG:32643) ---
# Bilinear interpolation is used to downsample from the ~0.5° CRU TS grid
# to the 1 km model grid.
CRUTS.rio.write_crs("epsg:4326", inplace=True)
CRUTS = CRUTS.rename({'lat': 'y', 'lon': 'x'})
CRUTS = CRUTS.rio.reproject("EPSG:32643")
CRUTS = CRUTS.rio.reproject_match(mask_da, resampling=Resampling.bilinear)
 
 
#%% =============================================================================
# Step 3 – Compute water balance components
# =============================================================================
 
# Runoff factor: fraction of precipitation assumed to become surface runoff
# before it can contribute to groundwater recharge.
runoff_factor = 0.025 # so 2.5 % of precipitation is assumed to be surface runoff
 
# RR  = rainfall runoff [mm/month]
# RCH = groundwater recharge [mm/month]; only non-negative values retained
# AET = actual evapotranspiration [mm/month] as residual of the water balance
CRUTS['RR']  = CRUTS['P'] * runoff_factor
CRUTS['RCH'] = (CRUTS['P'] - CRUTS['RR'] - CRUTS['PET']).clip(min=0)
CRUTS['AET'] = CRUTS['P'] - CRUTS['RR'] - CRUTS['RCH']
 
 
#%% =============================================================================
# Step 4 – Extend all variables back to 1800 using long-term monthly averages
# =============================================================================
 
# For the period 1800–1900, CRU TS data are not available. All water balance
# variables (P, PET, RR, RCH, AET) are extended using their long-term monthly
# average over the full CRU TS record (1901–2024)
new_time_range = pd.date_range('1800-01-01', '1900-12-31', freq='MS')
n_pre_years    = len(new_time_range) // 12
 
CRUTS_extended = xr.Dataset()
for var in CRUTS.data_vars:
    # Compute mean value for each calendar month across the full CRU TS period
    monthly_means = CRUTS[var].groupby('time.month').mean('time')
 
    # Tile the 12-month climatology across all pre-instrumental years
    pre_1901 = xr.DataArray(
        np.tile(monthly_means.data, (n_pre_years, 1, 1)),
        dims=['time', 'y', 'x'],
        coords={'time': new_time_range, 'y': CRUTS[var].y, 'x': CRUTS[var].x},
    )
 
    # Concatenate pre-1901 climatology with the CRU TS record (1901–2024)
    CRUTS_extended[var] = xr.concat([pre_1901, CRUTS[var]], dim='time')
 
 
#%% =============================================================================
# Step 5 – Convert units and export
# =============================================================================
 
# --- Convert from mm/month to m³/month ---
# Each model cell is 1 000 m × 1 000 m = 1 × 10⁶ m²; dividing by 1 000
# converts mm to m, giving m³/month per cell: factor = 1e6 / 1e3 = 1 000.
conversion_factor = 1_000_000 / 1_000   # mm/month → m³/month for a 1 km² cell
CRUTSm3month = CRUTS_extended * conversion_factor
 
for var_name in CRUTSm3month.data_vars:
    CRUTSm3month[var_name].attrs['units'] = 'm3/month'
 
# --- Export to NetCDF ---
CRUTSm3month.to_netcdf(output_path)
print(f"CRU TS recharge dataset saved to: {output_path}")

