# -*- coding: utf-8 -*-
"""
Author : Frank van Broekhoven
=======================
Compute spatially and temporally explicit industrial groundwater demand for
the MODFLOW model of the Ganges–Yamuna interfluve, covering 1800–2020 at monthly resolution.
 
Overview
--------
All industrial water use is assumed to be sourced from groundwater. Monthly
demand [m³/month] per model cell is computed in two steps:
 
    Step 1 – Rasterize Current Industrial Abstractions
        Point locations of industrial facilities and their reported discharge
        rates [m³/day] (Uttar Pradesh Pollution Control Board, n.d.) are
        rasterized onto the model grid. Multiple facilities falling within the
        same cell are summed. 
 
    Step 2 – Hindcast Using Logistic Growth Curve (UN Water, 2022)
        A logistic growth curve is fitted to national Indian groundwater
        withdrawal data (digitised from Figure 1.2 in UN Water, 2022) and
        normalised to the final observation year (2020). The resulting monthly
        index (1800–2020) is multiplied by the current abstraction raster to
        hindcast industrial demand back to 1800. The current spatial distribution 
        of industrial abstraction is assumed to remain constant over time; 
        only the magnitude is scaled by the hindcast index.
 
References
----------
Uttar Pradesh Pollution Control Board (n.d.)
UN Water (2022) – Groundwater: Making the Invisible Visible. UNESCO.
 
"""
 
#%% =============================================================================
# Imports
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import os
import rioxarray  # noqa: F401 – registers .rio accessor
from rasterio.transform import from_bounds
from scipy.optimize import curve_fit
 
 
#%% =============================================================================
# Paths
# =============================================================================

main_dir = r'C:\GitHub\Temp_CleaningTheGanga_Paper2'
 
# --- Input ---
mask_path     = os.path.join(main_dir, '1_model_input', '1_original_data', 'Model_Mask.tif')
industry_path = os.path.join(main_dir, '1_model_input', '1_original_data', 'Industry.geojson')

# --- Output ---
output_path   = os.path.join(main_dir, '1_model_input', '2_temp_data', 'industrial_demand.nc')
 
 
#%% =============================================================================
# Step 1 – Rasterize Current Industrial Abstractions
# =============================================================================
 
# --- Load industry point locations ---
# Each point represents an industrial facility with a reported discharge rate
# [m³/day] from the Uttar Pradesh Pollution Control Board.
industry = gpd.read_file(industry_path).to_crs(epsg=32643)
 
# --- Load model mask (defines grid extent, resolution and CRS) ---
mask = rioxarray.open_rasterio(mask_path)
 
# --- Rasterize point discharges onto the model grid ---
# Points are assigned to the nearest grid cell; multiple facilities within
# the same cell are summed. The transform is derived from the mask grid bounds
# and resolution so that point coordinates map correctly to pixel indices.
height, width = mask.sizes['y'], mask.sizes['x']
x_coords      = mask.x.values
y_coords      = mask.y.values
x_res         = (x_coords[-1] - x_coords[0]) / (len(x_coords) - 1)
y_res         = (y_coords[0]  - y_coords[-1]) / (len(y_coords) - 1)
 
transform = from_bounds(
    x_coords[0]  - x_res / 2,
    y_coords[-1] - abs(y_res) / 2,
    x_coords[-1] + x_res / 2,
    y_coords[0]  + abs(y_res) / 2,
    width,
    height,
)
 
rasterized = np.zeros((height, width), dtype=np.float32)
for geom, value in zip(industry.geometry, industry['Discharge m3/day']):
    col, row = ~transform * (geom.x, geom.y)
    rasterized[round(row), round(col)] += value
 

# industrial_demand_m3month: current industrial abstraction [m³/month] per cell
# Converted from m³/day using the average number of days per month.
DAYS_PER_MONTH = 365.25 / 12   # Average number of days per month [days/month]
industrial_demand_m3month = xr.DataArray(
    rasterized * DAYS_PER_MONTH,
    coords={'y': mask.y, 'x': mask.x},
    dims=['y', 'x'],
)
 
 
#%% =============================================================================
# Step 2 – Hindcast Using Logistic Growth Curve (UN Water, 2022)
# =============================================================================
 
# --- Fit logistic curve to national GW withdrawal observations ---
# Data points digitised from Figure 1.2 in UN Water (2022). The logistic curve
# captures the rapid growth in Indian groundwater abstraction during the second
# half of the 20th century and the levelling off towards 2017.
obs_years      = np.array([1950, 1960, 1970, 1980, 1990, 2000, 2004, 2007, 2010, 2017])
obs_withdrawal = np.array([15,   20,   70,   100,  150,  210,  230,  248,  250,  250])
 
def logistic(t, L, k, t0):
    """Logistic (sigmoid) growth function."""
    return L / (1 + np.exp(-k * (t - t0)))
 
popt, _ = curve_fit(logistic, obs_years, obs_withdrawal, p0=[260, 0.1, 1980]) # p0 - [L (km³/yr), k (growth rate), t0 (inflection year)]
L, k, t0 = popt
print(f"Logistic fit: L={L:.2f}, k={k:.4f}, t0={t0:.1f}")
 
# Generate yearly predictions 1800–2020
yearly_years      = np.arange(1800, 2021)
yearly_withdrawal = logistic(yearly_years, L, k, t0)
 
# --- Build monthly hindcast index, normalised to 2020 ---
# The index equals 1 at the final year (2020) so that the rasterized current
# abstraction is preserved at the end of the simulation period.
df = pd.DataFrame(
    {'withdrawal': yearly_withdrawal},
    index=pd.to_datetime(yearly_years, format='%Y'),
)
df['hindcast_index'] = df['withdrawal'] / df['withdrawal'].iloc[-1]
 
# Forward-fill yearly values to monthly frequency (step function)
df_monthly = df[['hindcast_index']].resample('MS').ffill()
 
# da_monthly: monthly hindcast index (1800–2020), normalised to 2020 = 1
da_monthly = xr.DataArray(
    df_monthly['hindcast_index'],
    dims=['time'],
    coords={'time': df_monthly.index},
)
 
# --- Apply hindcast index to current abstraction raster ---
# industrial_demand_da: industrial GW demand [m³/month] per cell, shape (n_months, ny, nx)
# The spatial pattern is assumed constant over time; only the magnitude varies.
industrial_demand_da = industrial_demand_m3month * da_monthly
 
 
#%% =============================================================================
# Export to NetCDF
# =============================================================================
 
industrial_demand_da.to_netcdf(output_path)
print(f"Industrial demand saved to: {output_path}")

