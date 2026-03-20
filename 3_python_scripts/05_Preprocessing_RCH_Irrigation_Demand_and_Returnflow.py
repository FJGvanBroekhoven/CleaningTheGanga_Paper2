# -*- coding: utf-8 -*-
"""
Author : Frank van Broekhoven
=======================
Compute spatially and temporally explicit irrigation groundwater demand and
return flow for the MODFLOW 6 model of the Hindon catchment (Upper Ganges–Yamuna interfluve, India), 
covering 1800–present at monthly resolution.

Overview
--------
This script combines three processing steps:

    Step 1 – Irrigated Area Estimation
        Derives groundwater- and canal-irrigated agricultural area [ha] per
        model cell and month using:
          - Copernicus/VITO Global Land Cover (2015) for agricultural extent
          - Census 2011 rasters for irrigation-source fractions
          - A logistic growth curve fitted to UN Water (2022) national GW
            withdrawal data to hindcast GW-irrigated area back to 1800
          - HYDE 3.2 (Klein Goldewijk et al., 2017) IGB-averaged total
            irrigated area to hindcast total irrigated area back to 1800

    Step 2 – Monthly Crop Coefficient Rasters
        Rasterizes district-level monthly crop coefficients onto the model grid:
          - District polygons (IND_adm2) merged with IWMI crop-coefficient table
          - Nearest-neighbour fill for model cells without district coverage

    Step 3 – Irrigation Demand & Return Flow
        Computes volumetric irrigation demand and return flow [m³/month]:
          - Crop water demand = PET (CRU TS) × monthly crop coefficient
          - Irrigation demand accounts for return flow and rainfall runoff
          - Final outputs written to NetCDF

References
----------
Buchhorn et al. (2020) – Copernicus Global Land Cover
Government of India (2011) – Census of India
UN Water (2022) – Groundwater: Making the Invisible Visible. UNESCO.
Harris et al. (2020) – CRU TS v4
Cai et al. (2010) – IWMI Research Report 140
INTACH (2017) 
Klein Goldewijk et al. (2017) – HYDE 3.2 historical land use dataset
Gupta & Deshpande (2004) in Deltares report (van der Vat)

"""

#%% =============================================================================
# Imports
# =============================================================================
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import rasterio
import rioxarray
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from scipy.optimize import curve_fit

#%% =============================================================================
# Paths
# =============================================================================

main_dir = r'C:\GitHub\Temp_CleaningTheGanga_Paper2'

# --- Input ---
mask_path          = os.path.join(main_dir, '1_model_input', '1_original_data', 'Model_Mask.tif')
landuse_path       = os.path.join(main_dir, '1_model_input', '1_original_data', 'Land_Cover_Copernicus_2015.tif')
canal_irri_path    = os.path.join(main_dir, '1_model_input', '1_original_data', 'Census_2011', 'Canal_irrigated.tif')
gw_irri_path       = os.path.join(main_dir, '1_model_input', '1_original_data', 'Census_2011', 'GW_irrigated.tif')
districts_path     = os.path.join(main_dir, '1_model_input', '1_original_data', 'Districts_IND_adm2.geojson')
cropcoef_csv_path  = os.path.join(main_dir, '1_model_input', '1_original_data', 'monthly_crop_coefficients_per_district.csv')
tot_irri_path      = os.path.join(main_dir, '1_model_input', '1_original_data', 'HYDE3_2', 'HYDE3_2_total_irrigated_data_1800-2017.nc')
CRUTS_path         = os.path.join(main_dir, '1_model_input', '2_temp_data', 'CRUTS_derived_rainfall_recharge.nc')

# --- Output ---
irrigation_demand_output_path     = os.path.join(main_dir, '1_model_input', '2_temp_data', 'irrigation_demand.nc')
irrigation_returnflow_output_path = os.path.join(main_dir, '1_model_input', '2_temp_data', 'irrigation_returnflow.nc')

#%% =============================================================================
# Step 1 – Irrigated area per cell per month
# =============================================================================

# --- Load model mask ---
mask = rioxarray.open_rasterio(mask_path).squeeze(drop='band')
mask.rio.write_crs("EPSG:32643", inplace=True)

#%% --- Agriculture fraction from Copernicus/VITO land cover (class 40 = cropland) ---
# current land use: agriculture percentage per cell.
# Fine-resolution pixels are aggregated to the coarse model grid by computing
# the fraction of fine pixels that belong to class 40 within each coarse cell.
with rasterio.open(landuse_path) as fine_src:
    fine_data = fine_src.read(1)

with rasterio.open(mask_path) as coarse_src:
    coarse_data = coarse_src.read(1)

x_factor = int(fine_data.shape[1] / coarse_data.shape[1])
y_factor = int(fine_data.shape[0] / coarse_data.shape[0])

fine_trimmed = fine_data[
    : coarse_data.shape[0] * y_factor,
    : coarse_data.shape[1] * x_factor,
]
fine_reshaped = fine_trimmed.reshape(
    coarse_data.shape[0], y_factor,
    coarse_data.shape[1], x_factor,
)

# agriculture_percentage: fraction of each model cell covered by cropland [0–100 %]
agriculture_percentage = np.mean(fine_reshaped == 40, axis=(1, 3)) * 100

# Print the resulting percentage array
plt.figure(figsize=(8, 6))
im = plt.imshow(agriculture_percentage, cmap='YlGn')  # Optional: use a green-ish colormap
cbar = plt.colorbar(im)
cbar.set_label('Percentage of agriculture per pixel')
# Axis labels (optional)
plt.xlabel('X pixel')
plt.ylabel('Y pixel')
plt.title('Agricultural Coverage')
plt.tight_layout()
plt.show()

#%% --- Census 2011 irrigation-source fractions ---
# canal_irri: fraction of agricultural area irrigated by canal water per cell
# GW_irri   : fraction of agricultural area irrigated by groundwater per cell
canal_irri = rioxarray.open_rasterio(canal_irri_path).squeeze()
GW_irri    = rioxarray.open_rasterio(gw_irri_path).squeeze()

# plots of Census 2011 irrigation-source fractions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# --- Canal irrigation subplot ---
im1 = axes[0].imshow(canal_irri, cmap='Blues')
cbar1 = plt.colorbar(im1, ax=axes[0])
cbar1.set_label('Percentage per cell irrigated with canal water')
axes[0].set_title('Canal Irrigation')
axes[0].set_xlabel('X pixel')
axes[0].set_ylabel('Y pixel')

# --- Groundwater irrigation subplot ---
im2 = axes[1].imshow(GW_irri, cmap='Oranges')
cbar2 = plt.colorbar(im2, ax=axes[1])
cbar2.set_label('Percentage per cell irrigated with groundwater')
axes[1].set_title('Groundwater Irrigation')
axes[1].set_xlabel('X pixel')
axes[1].set_ylabel('Y pixel')

plt.tight_layout()
plt.show()

#%% --- Hindcast index 1: logistic curve (UN Water 2022) for GW-irrigated area ---
# A logistic curve is fitted to national Indian GW withdrawal data (digitised
# from Figure 1.2 in UN Water 2022) to hindcast the temporal trend in
# groundwater-irrigated area back to 1800. The curve is normalised to the
# 2011 census year (LOGISTIC_REF_YEAR) so that the index equals 1 at that year.
obs_years      = np.array([1950, 1960, 1970, 1980, 1990, 2000, 2004, 2007, 2010, 2017])
obs_withdrawal = np.array([15,   20,   70,   100,  150,  210,  230,  248,  250,  250])

def logistic(t, L, k, t0):
    """Logistic (sigmoid) growth function used to fit GW withdrawal trend."""
    return L / (1 + np.exp(-k * (t - t0)))

popt, _ = curve_fit(logistic, obs_years, obs_withdrawal, p0=[260, 0.1, 1980])
L, k, t0 = popt

yearly_years      = np.arange(1800, 2021)
yearly_withdrawal = logistic(yearly_years, L, k, t0)

# Plot the data and fitted curve
plt.figure(figsize=(10, 6))
plt.scatter(obs_years , obs_withdrawal, color='red', label='Original Data')
plt.plot(yearly_years, yearly_withdrawal, color='blue', label='Logistic Fit')
# Add fit parameters as text on the plot
param_text = f"L = {L:.2f}\nk = {k:.4f}\nt₀ = {t0:.1f}"
plt.text(1820, 200, param_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
# Labels and formatting
plt.xlabel('Year')
plt.ylabel('Rate of Withdrawal (km³/year)')
plt.title('Water Withdrawal Rate with Logistic Fit')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Normalise to the census reference year and resample to monthly frequency
LOGISTIC_REF_YEAR      = 2011    # Census reference year for normalisation
df_gw = pd.DataFrame(
    {'withdrawal': yearly_withdrawal},
    index=pd.to_datetime(yearly_years, format='%Y'),
)
ref_value               = logistic(np.array([LOGISTIC_REF_YEAR]), L, k, t0)[0]
df_gw['hindcast_index'] = df_gw['withdrawal'] / ref_value
df_gw_monthly           = df_gw[['hindcast_index']].resample('MS').ffill()

# gw_hindcast_index: monthly index (1800–2020), normalised to 2011 census = 1
# Used to scale GW-irrigated area over time.
gw_hindcast_index = xr.DataArray(
    df_gw_monthly['hindcast_index'],
    dims=['time'],
    coords={'time': df_gw_monthly.index},
)

#%% --- Hindcast index 2: HYDE 3.2 IGB spatial mean for total irrigated area ---
# Total irrigated area is hindcast using the HYDE 3.2 dataset
# (Klein Goldewijk et al., 2017). The spatially averaged total irrigated area
# across the Indo-Gangetic Basin (IGB) at each time step is used as a temporal
# index. The dataset is normalised to its final time step (2017) and then
# linearly interpolated from decadal to monthly frequency.
tot_irri = xr.open_dataset(tot_irri_path)['total_irrigated']
tot_irri.rio.write_crs("epsg:4326", inplace=True)
tot_irri = tot_irri.rename({'lat': 'y', 'lon': 'x'})

# Compute IGB spatial mean at each available time step and normalise to 2017
tot_irri_mean = tot_irri.mean(dim=['x', 'y'], skipna=True)
tot_irri_index_1d = tot_irri_mean / tot_irri_mean.isel(time=-1)   # normalise to 2017

# Convert integer year coordinates to datetime and interpolate to monthly
years_as_dates    = pd.to_datetime(tot_irri_index_1d.indexes['time'].values, format='%Y')
tot_irri_index_1d = tot_irri_index_1d.assign_coords(time=years_as_dates)

new_time_yearly   = pd.date_range(
    start=tot_irri_index_1d.time.min().values,
    end=tot_irri_index_1d.time.max().values,
    freq='YS',
)
new_time_monthly  = pd.date_range(
    start=tot_irri_index_1d.time.min().values,
    end=tot_irri_index_1d.time.max().values,
    freq='MS',
)

# Linear interpolation: decadal → yearly → monthly
tot_irri_index_yearly  = tot_irri_index_1d.interp(time=new_time_yearly)
tot_irri_index_monthly = tot_irri_index_yearly.interp(time=new_time_monthly)

# hyde_hindcast_index: monthly index (1800–2017), normalised to 2017 IGB mean = 1
# Used to scale total irrigated area (canal + GW) over time.
hyde_hindcast_index = tot_irri_index_monthly
#hyde_hindcast_index.plot()


#%% --- Irrigated area [ha] per cell per month ---
# GW-irrigated ha uses the logistic hindcast (UN Water 2022):
#   GW_irrigated_ha = agriculture_fraction × GW_fraction × gw_hindcast_index × cell_area
# Total irrigated ha uses the HYDE 3.2 spatial-mean hindcast:
#   total_irrigated_ha = agriculture_fraction × (canal + GW)_fraction × hyde_hindcast_index × cell_area
CELL_SIZE_HA           = 100     # Cell area [ha] (1 000 m × 1 000 m)
GW_irrigated_ha = agriculture_percentage / 100 * GW_irri * gw_hindcast_index * CELL_SIZE_HA
GW_irrigated_ha = GW_irrigated_ha.where(mask == 1)

total_irrigated_ha = agriculture_percentage / 100 * (canal_irri + GW_irri) * hyde_hindcast_index * CELL_SIZE_HA
total_irrigated_ha = total_irrigated_ha.where(mask == 1)



#%% =============================================================================
# Step 2 – Monthly crop coefficient rasters
# =============================================================================

# District-level monthly crop coefficients (IWMI Research Report 140,
# Cai et al. 2010) combined with district crop distribution (INTACH 2017)
# are rasterized onto the model grid. Cells inside the model domain that fall
# outside any district polygon are filled using nearest-neighbour interpolation.

# --- Load mask DataArray and district polygons ---
mask_da = xr.open_dataarray(mask_path).squeeze(drop='band')
mask_da = mask_da.rio.set_spatial_dims(x_dim="x", y_dim="y")

districts = gpd.read_file(districts_path)
districts['NAME_2'] = districts['NAME_2'].replace('Hardwar', 'Haridwar')  # name fix

cropcoef = pd.read_csv(cropcoef_csv_path, delimiter=";")

# Merge district geometries with crop coefficient table
gdf = districts.merge(cropcoef, left_on="NAME_2", right_on="District", how="outer")
mask_da.rio.write_crs(gdf.crs, inplace=True)
gdf = gdf.to_crs(mask_da.rio.crs)

transform = mask_da.rio.transform()
out_shape = (mask_da.sizes["y"], mask_da.sizes["x"])

#%% --- Rasterize each month onto model grid ---
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

rasters = []
for m in MONTHS:
    shapes = [(geom, val) for geom, val in zip(gdf.geometry, gdf[m])]
    arr = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=np.nan,
        dtype="float32",
    )
    rasters.append(
        xr.DataArray(arr, coords={"y": mask_da.y, "x": mask_da.x}, dims=("y", "x"))
    )

raster_da = xr.concat(rasters, dim="month").assign_coords(month=MONTHS)
raster_da.rio.write_crs(mask_da.rio.crs, inplace=True)

# --- Mask to model domain and nearest-neighbour fill ---
# Cells within the model domain but outside any district polygon receive the
# value of the nearest district cell.
raster_masked = raster_da.where(mask_da == 1)
valid_mask    = mask_da.values == 1

filled_list = []
for m in raster_masked.month.values:
    arr     = raster_masked.sel(month=m).values
    missing = np.isnan(arr) & valid_mask
    if np.any(missing):
        inds   = distance_transform_edt(np.isnan(arr), return_distances=False, return_indices=True)
        arr    = arr.copy()
        arr[missing] = arr[tuple(inds[:, missing])]
    filled_list.append(
        xr.DataArray(arr, coords={"y": raster_da.y, "x": raster_da.x}, dims=("y", "x"))
    )

# monthly_cropcoef: monthly crop coefficient [-] per model cell, shape (12, ny, nx)
monthly_cropcoef = xr.concat(filled_list, dim="month").assign_coords(month=MONTHS)
monthly_cropcoef.rio.write_crs(raster_da.rio.crs, inplace=True)
monthly_cropcoef.name  = "monthly_crop_coefficients"
monthly_cropcoef.attrs["units"] = "[-]"

#Plot monthly maps
months = monthly_cropcoef.month.values
fig, axes = plt.subplots(3, 4, figsize=(16,10), constrained_layout=True)
axes = axes.flatten()
vmin = float(monthly_cropcoef.min())
vmax = float(monthly_cropcoef.max())
for i, m in enumerate(months):
    ax = axes[i]
    monthly_cropcoef.sel(month=m).plot(
        ax=ax,
        add_colorbar=False,
        vmin=vmin,
        vmax=vmax,
        cmap="Greens"
    )
    ax.set_title(str(m))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_frame_on(False)
cbar = fig.colorbar(
    axes[0].collections[0],
    ax=axes,
    shrink=0.8
    )
cbar.set_label("crop coefficient [-]")
plt.show()


#%% =============================================================================
# Step 3 – Irrigation demand and return flow
# =============================================================================

# --- Load CRU TS meteorological data ---
CRUTS = xr.open_dataset(CRUTS_path)

# Convert PET and precipitation from m³/month (model units) to mm/month
PET = CRUTS['PET'] * 1000 / 1000000
P   = CRUTS['P']   * 1000 / 1000000

# Add abbreviated month label to align with the 'month' dimension of crop coefficients
PET = PET.assign_coords(month=PET['time'].dt.strftime('%b'))

# --- Crop water demand and irrigation demand [mm/month] ---
# Crop water demand = PET × district-level monthly crop coefficient
crop_water_demand = PET * monthly_cropcoef.sel(month=PET['month'])

# Irrigation demand accounts for:
#   - return flow: demand is grossed up by 1/(1 - f_rf) to cover losses
#   - effective rainfall: precipitation minus surface runoff offsets demand
RETURNFLOW_FACTOR      = 0.162   # Fraction of irrigation returning to GW
                                  # (Gupta & Deshpande, 2004 via Deltares report)
RAINFALL_RUNOFF_FACTOR = 0.025   # Fraction of precipitation lost as runoff
                                 
irrigation_demand = crop_water_demand * 1 / (1 - RETURNFLOW_FACTOR) - P * (1 - RAINFALL_RUNOFF_FACTOR)
irrigation_demand = irrigation_demand.clip(min=0)   # irrigation demand cannot be negative
irrigation_demand = irrigation_demand.drop_vars('month')

'''
# --- Extend demand back to 1800 using monthly climatology ---
# The period before the CRU TS record is filled with the long-term monthly
# mean of the available record, assuming a stationary pre-industrial climate.
monthly_means  = irrigation_demand.groupby('time.month').mean('time')
new_time_range = pd.date_range('1800-01-01', '1900-12-31', freq='MS')

new_data = np.tile(monthly_means.data, (len(new_time_range) // 12, 1, 1))
new_da   = xr.DataArray(
    new_data,
    dims=['time', 'y', 'x'],
    coords={'time': new_time_range, 'y': irrigation_demand.y, 'x': irrigation_demand.x},
)

extended_irrigation_demand = xr.concat([new_da, irrigation_demand], dim='time')
'''

# --- Convert to volumetric demand and multiply by irrigated area ---
# mm/month → m³/month/ha: divide by 1000 (mm→m) × 10 000 (m² per ha)
irrigation_factor = irrigation_demand / 1000 * 10000

# actual_irrigation_demand    : GW abstraction for irrigation [m³/month] per cell
# actual_irrigation_returnflow: return flow to GW from all irrigation [m³/month] per cell
actual_irrigation_demand     = GW_irrigated_ha    * irrigation_factor
actual_irrigation_returnflow = total_irrigated_ha * irrigation_factor * RETURNFLOW_FACTOR

# --- Quick diagnostic plots ---
actual_irrigation_demand.mean(dim='time').plot()
actual_irrigation_demand.mean(dim=['x', 'y']).plot()

actual_irrigation_returnflow.mean(dim='time').plot()
actual_irrigation_returnflow.mean(dim=['x', 'y']).plot()


#%% =============================================================================
# Export to NetCDF
# =============================================================================

actual_irrigation_demand.to_netcdf(irrigation_demand_output_path)
actual_irrigation_returnflow.to_netcdf(irrigation_returnflow_output_path)

print(f"Irrigation demand saved to     : {irrigation_demand_output_path}")
print(f"Irrigation return flow saved to: {irrigation_returnflow_output_path}")