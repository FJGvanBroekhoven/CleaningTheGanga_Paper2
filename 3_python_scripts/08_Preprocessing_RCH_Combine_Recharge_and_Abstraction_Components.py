# -*- coding: utf-8 -*-
"""
Author : Frank van Broekhoven
=======================
Combine all groundwater recharge and abstraction components into a single
yearly dataset for the MODFLOW model of the Ganges–Yamuna interfluve,
covering 1800–2016.
 
Overview
--------
Individual monthly recharge and abstraction NetCDF files (produced by
upstream processing scripts) are read, aligned, masked to the model domain,
and resampled from monthly to yearly totals [m³/year]. Abstraction components
are stored as negative values. The combined dataset is exported as a single
NetCDF file for use as MODFLOW model forcing.
 
Components
----------
Recharge:
    rainfall_rch          – Rainfall recharge
    canal_rch             – Canal leakage recharge
    irrigation_return_rch – Irrigation return flow recharge
    municipal_return_rch  – Municipal wastewater return flow recharge
 
Abstraction (stored as negative values):
    irrigation_demand_abs – Groundwater abstraction for irrigation
    municipal_demand_abs  – Groundwater abstraction for municipal supply
    industrial_demand_abs – Groundwater abstraction for industrial use
 
"""
 
#%% =============================================================================
# Imports
# =============================================================================
import numpy as np
import xarray as xr
import rioxarray
import os
 
 
#%% =============================================================================
# Paths
# =============================================================================
 
main_dir = r'C:\GitHub\Temp_CleaningTheGanga_Paper2'
 
# --- Input ---
mask_path                  = os.path.join(main_dir, '1_model_input', '1_original_data', 'Model_Mask.tif')
rainfall_rch_path          = os.path.join(main_dir, '1_model_input', '2_temp_data', 'CRUTS_derived_rainfall_recharge.nc')
canal_rch_path             = os.path.join(main_dir, '1_model_input', '2_temp_data', 'irrigation_canal_leakage.nc')
irrigation_return_rch_path = os.path.join(main_dir, '1_model_input', '2_temp_data', 'irrigation_returnflow.nc')
municipal_return_rch_path  = os.path.join(main_dir, '1_model_input', '2_temp_data', 'Municipal_return_flow.nc')
irrigation_demand_abs_path = os.path.join(main_dir, '1_model_input', '2_temp_data', 'irrigation_demand.nc')
municipal_demand_abs_path  = os.path.join(main_dir, '1_model_input', '2_temp_data', 'Municipal_water_demand.nc')
industrial_demand_abs_path = os.path.join(main_dir, '1_model_input', '2_temp_data', 'industrial_demand.nc')
 
# --- Output ---
output_path = os.path.join(main_dir, '1_model_input', '3_model_input_data', 'recharge_and_abstraction_components_yearly.nc')
 
 
#%% =============================================================================
# Step 1 – Load all components
# =============================================================================
 
# --- Model mask ---
mask = xr.open_dataarray(mask_path).squeeze(drop='band')
 
# --- Recharge components ---
# Transposing to (time, y, x) ensures consistent dimension ordering across all arrays.
rainfall_rch      = xr.open_dataset(rainfall_rch_path)['RCH']
canal_rch         = xr.open_dataarray(canal_rch_path)
irrigation_return = xr.open_dataarray(irrigation_return_rch_path).squeeze(drop='band').transpose('time', 'y', 'x')
municipal_return  = xr.open_dataset(municipal_return_rch_path).squeeze(drop='band')['recharge']
 
# --- Abstraction components ---
irrigation_demand = xr.open_dataarray(irrigation_demand_abs_path).squeeze(drop='band').transpose('time', 'y', 'x')
municipal_demand  = xr.open_dataarray(municipal_demand_abs_path).squeeze('band', drop=True)
industry_demand   = xr.open_dataarray(industrial_demand_abs_path).transpose('time', 'y', 'x')
 
 
#%% =============================================================================
# Step 2 – Combine, mask, clip to model period and resample to yearly
# =============================================================================
 
# --- Assemble into a single Dataset ---
# Abstraction components are negated so that all variables share the same
# sign convention: positive = flux into groundwater, negative = flux out.
components = xr.Dataset({
    'rainfall_rch':          rainfall_rch,
    'canal_rch':             canal_rch,
    'irrigation_return_rch': irrigation_return,
    'municipal_return_rch':  municipal_return,
    'irrigation_demand_abs': irrigation_demand * -1,
    'municipal_demand_abs':  municipal_demand  * -1,
    'industrial_demand_abs': industry_demand   * -1,
    })
 
# --- Mask to model domain ---
# Cells outside the model domain (mask ≠ 1) are set to NaN.
components = xr.where(mask == 1, components, np.nan)
 
# --- Clip to simulation period 1800–2016 ---
components = components.sel(time=slice('1800', '2016'))
 
# --- Resample from monthly to yearly totals [m³/year] ---
# skipna=False ensures that years with any missing monthly data are not
# silently summed from incomplete records.
components_yearly = components.resample(time='YE').sum(dim='time', skipna=False)
 
# update units m3/year
for var_name in components_yearly.data_vars:
    components_yearly[var_name].attrs['units'] = 'm3/year'
 
 
#%% =============================================================================
# Export to NetCDF
# =============================================================================

# reduce file size
# Convert to float32
components_yearly = components_yearly.astype("float32")
# Compression settings
encoding = {var: {"zlib": True, "complevel": 1 } for var in components_yearly.data_vars}

components_yearly.to_netcdf(output_path, encoding=encoding)
print(f"Combined yearly components saved to: {output_path}")




