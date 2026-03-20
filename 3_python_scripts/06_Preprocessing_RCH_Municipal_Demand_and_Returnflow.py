# -*- coding: utf-8 -*-
"""
Author : Frank van Broekhoven
=======================================
Compute spatially and temporally explicit municipal groundwater demand and
return flow recharge for the MODFLOW model of the Ganges–Yamuna interfluve, 
covering 1800–2020 at monthly resolution.
 
Overview
--------
This script covers two processing steps:
 
    Step 1 – Municipal Groundwater Demand
        All domestic water supply is assumed to be sourced from groundwater.
        Monthly demand [m³/month] per model cell is computed as:
            demand = population_per_cell × per_capita_water_use
        where:
          - Current (2017) population distribution comes from WorldPop
            (Bondarenko et al., 2025) at 1 km resolution.
          - Historical population is hindcast spatially using HYDE 3.2
            (Klein Goldewijk et al., 2017), normalised to 2017 and multiplied
            by the WorldPop baseline. Years 2018–2020 are appended directly
            from annual WorldPop rasters.
          - Per capita water use is taken from Joseph et al. (2021) for
            1975–2015 and linearly extrapolated back to 1800 (assuming
            25 litres/person/day at 1800).
 
    Step 2 – Municipal Return Flow Recharge
        Municipal wastewater is assumed to either discharge to rivers or to
        ponds, depending on proximity to the river network:
          - Within 2.5 km of a river  → wastewater reaches rivers;
              leakage to groundwater factor = 0.10
              discharge to river factor     = 0.85
              evaporation factor            = 0.05
          - Beyond 2.5 km from a river → wastewater discharges to ponds;
              pond infiltration factor      = 0.55
              pond evaporation factor       = 0.30
              (direct) leakage factor       = 0.10 (applied everywhere)
        The river proximity mask is derived by rasterizing a 2.5 km buffer
        around HydroSHEDS river polylines onto the model grid.
 
References
----------
Bondarenko et al. (2025) – WorldPop gridded population
Klein Goldewijk et al. (2017) – HYDE 3.2 historical land use and population
Joseph et al. (2021) – Per capita domestic water use, Urban Water Journal
"""

#%% =============================================================================
# Imports
# =============================================================================
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rioxarray
import xarray as xr
from rasterio.features import rasterize
from shapely.geometry import mapping


#%% =============================================================================
# Paths
# =============================================================================
 
main_dir = r'C:\GitHub\Temp_CleaningTheGanga_Paper2'
 
# --- Input ---
mask_path           = os.path.join(main_dir, '1_model_input', '1_original_data', 'Model_Mask.tif')
population_2017_path = os.path.join(main_dir, '1_model_input', '1_original_data', 'WorldPop', 'ind_ppp_2017_1km_Aggregated_UNadj.tif')
hyde_path           = os.path.join(main_dir, '1_model_input', '1_original_data', 'HYDE3_2', 'HYDE3_2_population_data_1800-2017.nc')
water_use_path      = os.path.join(main_dir, '1_model_input', '1_original_data', 'domestic_water_use_per_capita.csv')
rivers_path         = os.path.join(main_dir, '1_model_input', '1_original_data', 'HydroSHEDS', 'HydroRIVERS_crop.geojson')
 
# WorldPop rasters for years beyond the HYDE dataset (2018–2020)
worldpop_extra_years = {
    2018: '1_model_input/1_original_data/WorldPop/ind_ppp_2018_1km_Aggregated_UNadj.tif',
    2019: '1_model_input/1_original_data/WorldPop/ind_ppp_2019_1km_Aggregated_UNadj.tif',
    2020: '1_model_input/1_original_data/WorldPop/ind_ppp_2020_1km_Aggregated_UNadj.tif',
}
 
# --- Output ---
demand_output_path      = os.path.join(main_dir, '1_model_input', '2_temp_data', 'Municipal_water_demand.nc')
returnflow_output_path  = os.path.join(main_dir, '1_model_input', '2_temp_data', 'Municipal_return_flow.nc')
 
#%% ============================================================================
# Step 1 – Municipal Groundwater Demand
# =============================================================================
 
# --- Load model mask ---
mask = xr.open_dataarray(mask_path).isel(band=0)
mask.rio.write_crs("EPSG:32643", inplace=True)
 
# --- WorldPop 2017: baseline spatial population distribution ---
# No-data values in WorldPop are encoded as -99999; these are replaced with 0.
population_2017 = rioxarray.open_rasterio(population_2017_path)
population_2017 = population_2017.where(population_2017 >= 0, 0)
population_2017.rio.write_crs("epsg:4326", inplace=True)
population_2017 = population_2017.rio.reproject_match(mask)
 
#%% --- HYDE 3.2: spatially explicit population hindcast (1800–2017) ---
# The HYDE 3.2 dataset is normalised to the 2017 baseline (index = 1 in 2017)
# and reprojected to the model grid. Multiplying by the WorldPop 2017 raster
# then gives a spatially downscaled population count for each historical year.
# --- HYDE reference year ---
HYDE_REFERENCE_YEAR = 2017    # WorldPop baseline year; HYDE index = 1 at this year
hyde = xr.open_dataset(hyde_path, decode_coords="all")
hyde_index = hyde / hyde.sel(time=HYDE_REFERENCE_YEAR)
hyde_index.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
hyde_index.rio.write_crs("epsg:4326", inplace=True)
hyde_index = hyde_index.rio.reproject("EPSG:32643")
hyde_index = hyde_index.rio.reproject_match(population_2017)
 
# population_per_year: spatially downscaled population count per cell [persons]
# at decadal HYDE time steps (1800–2017)
population_per_year = (hyde_index * population_2017)['population_count']
 
# --- Interpolate HYDE from decadal to annual frequency ---
# Convert integer year coordinates to datetime, then linearly interpolate.
years_as_dates      = pd.to_datetime(population_per_year.indexes['time'].values, format='%Y')
population_per_year = population_per_year.assign_coords(time=years_as_dates)
annual_time         = pd.date_range(
    start=population_per_year.time.min().values,
    end=population_per_year.time.max().values,
    freq='YS',
)
population_per_year = population_per_year.interp(time=annual_time)
 
#%% --- Append WorldPop years 2018–2020 (beyond HYDE coverage) ---
# Direct WorldPop rasters are used for years not covered by HYDE 3.2.
for year, rel_path in worldpop_extra_years.items():
    pop = rioxarray.open_rasterio(os.path.join(main_dir, rel_path))
    pop = pop.where(pop >= 0, 0)
    pop.rio.write_crs("epsg:4326", inplace=True)
    pop = pop.rio.reproject_match(mask)
    pop = pop.expand_dims(time=pd.to_datetime([year], format='%Y'))
    population_per_year = xr.concat([population_per_year, pop], dim='time')
 
#%% --- Per capita water use (Joseph et al., 2021) ---
# Annual per capita domestic water use [m³/person/year] from literature
# (1975–2015). Values are extrapolated back to 1800 using a linear trend
# with a fixed anchor of 25 litres/person/day at 1800 (≈ 9.125 m³/person/year).
water_use_daily = pd.read_csv(water_use_path, delimiter=';', decimal=',', parse_dates=['Year'])
water_use_daily.rename(columns={'Year': 'time'}, inplace=True)
water_use_daily.set_index('time', inplace=True)
 
water_demand_per_capita = water_use_daily['Per-capita domestic water use (m3/person)']
water_demand_da = xr.DataArray(
    water_demand_per_capita,
    dims=["time"],
    coords={"time": water_demand_per_capita.index},
)
 
# Align per capita time series to the population time axis
water_demand_da = water_demand_da.sel(time=population_per_year.time)
 
# Annual municipal demand [m³/year] per cell = population × per capita use
water_demand_per_pixel = population_per_year * water_demand_da
 
#%% --- Resample annual demand to monthly frequency ---
# Yearly totals are divided equally across the 12 months of each year.
monthly_time = pd.date_range(
    start=water_demand_per_pixel.time.min().values,
    end=pd.Timestamp(water_demand_per_pixel.time.max().values).replace(month=12, day=31),
    freq='MS',
)
# demand_monthly: monthly municipal groundwater demand [m³/month] per model cell
demand_monthly = water_demand_per_pixel.reindex(time=monthly_time, method="ffill") / 12
 
#%% --- Export municipal demand to NetCDF ---
demand_monthly.to_netcdf(demand_output_path)
print(f"Municipal demand saved to: {demand_output_path}")
 

#%% =============================================================================
# Step 2 – Municipal Return Flow Recharge
# =============================================================================
 
# --- River proximity mask ---
# A binary raster is derived by rasterizing a 2.5 km buffer around the
# HydroSHEDS river network. Cells within the buffer (value = 1) are assumed
# to discharge wastewater to rivers; cells outside (value = 0) discharge to
# ponds. CRS is inherited from the model mask (EPSG:32643).
demand = demand_monthly
 
# --- River proximity threshold ---
RIVER_BUFFER_M = 2500         # Buffer radius around rivers [m]; cells within
                               # this distance are assumed to drain to rivers
rivers = gpd.read_file(rivers_path)
rivers_buffer = rivers.geometry.buffer(RIVER_BUFFER_M)
 
x   = demand_monthly.x.values
y   = demand_monthly.y.values
dx  = x[1] - x[0]
dy  = y[0] - y[1]
transform = rasterio.transform.from_origin(x.min(), y.max(), dx, dy)
 
shapes = [(mapping(geom), 1) for geom in rivers_buffer.geometry]
rivers_raster_arr = rasterize(
    shapes,
    out_shape=(len(y), len(x)),
    transform=transform,
    fill=0,
    dtype=np.uint8,
)
 
# rivers_raster: 1 = within RIVER_BUFFER_M of a river, 0 = beyond buffer
rivers_raster = xr.DataArray(
    rivers_raster_arr,
    dims=("y", "x"),
    coords={"y": y, "x": x},
)
 
#%% --- Compute return flow components ---
# Leakage to groundwater occurs everywhere at LEAKAGE_FACTOR_RIVER (0.10).
# In cells beyond the river buffer, additional recharge occurs via pond
# infiltration at POND_RECHARGE_FACTOR (0.55).

# --- Return flow factors ---
LEAKAGE_FACTOR_RIVER = 0.10   # Groundwater leakage fraction for wastewater
                               # discharged to rivers (applied everywhere)
POND_RECHARGE_FACTOR = 0.55   # Infiltration fraction for wastewater discharged
                               # to ponds (cells > 2.5 km from river)
POND_ET_FACTOR       = 0.30   # Evaporation fraction from ponds
RIVER_DISCHARGE_FACTOR = 0.85 # Fraction of wastewater that reaches the river
                               # in cells within the river buffer
DEMAND_ET_FACTOR     = 0.05   # Fraction of municipal demand lost to
                               # evapotranspiration (direct usage losses)

leakage       = demand_monthly * LEAKAGE_FACTOR_RIVER
pond_recharge = demand_monthly * (1 - rivers_raster) * POND_RECHARGE_FACTOR
 
# municipal_return_flow: total groundwater recharge from municipal wastewater [m³/month]
municipal_return_flow = leakage + pond_recharge
 
# Remaining water balance components (for mass-balance checks or other uses)
to_rivers = demand_monthly * rivers_raster * RIVER_DISCHARGE_FACTOR   # wastewater reaching rivers [m³/month]
ET_usage  = demand_monthly * DEMAND_ET_FACTOR                         # evaporation from direct use losses [m³/month]
ET_pond   = demand_monthly * (1 - rivers_raster) * POND_ET_FACTOR     # evaporation from ponds [m³/month]
ET_total  = ET_usage + ET_pond                                        # total municipal ET [m³/month]
 
#%% --- Combine return flow components into a single Dataset ---
municipal_outflow = xr.Dataset({
    'recharge':   municipal_return_flow,   # Groundwater recharge from municipal return flow [m³/month]
    'discharge':  to_rivers,               # Wastewater discharged to rivers [m³/month]
    'ET':         ET_total,                # Total evapotranspiration from municipal water use [m³/month]
})
 
#%% --- Export return flow dataset to NetCDF ---
municipal_outflow.to_netcdf(returnflow_output_path)
print(f"Municipal return flow saved to: {returnflow_output_path}")

