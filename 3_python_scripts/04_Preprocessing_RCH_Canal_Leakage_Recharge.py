# -*- coding: utf-8 -*-
"""
Author : Frank van Broekhoven
=======================
Compute spatially and temporally explicit canal leakage recharge for the
MODFLOW model of the Ganges–Yamuna interfluve, covering 1800–2024 at
monthly resolution.
 
Overview
--------
Canal leakage recharge is estimated by multiplying canal area per model cell
by a literature-based leakage rate. The processing is split into three steps:
 
    Step 1 – Canal Area Per Cell
        Canal locations, types and construction years are read from the
        India-WRIS dataset (India-WRIS, n.d.). Canal width is assigned per
        canal type (ranging from 5 m to 100 m). Canal length per model cell
        is computed by clipping each canal polyline to the cell polygon.
        Canal area per cell = length × width.
 
    Step 2 – Temporal Activation by Construction Year
        Each model cell's canal area is activated from the year the canal
        was constructed. Canal leakage is computed from 1830 onwards, when
        the Eastern Yamuna Canal (1830) and Upper Ganga Canal (1856) were constructed. 
        Pre-1830 canals are assumed to have negligible contribution to recharge.
 
    Step 3 – Resample to Monthly Frequency
        The yearly canal area DataArray is repeated 12 times per year to
        produce a monthly time series
        
    Step 4 – Calculate canal leakage form canal area and leakage rate and Export
        The monthly canal array DataArray is multiplied by a literature-based 
        leakage rate factor (Raza et al., 2013), which is exported to NetCDF.
 
References
----------
India-WRIS (n.d.)
Raza et al. (2013) 
 
"""
 
#%% =============================================================================
# Imports
# =============================================================================
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
import rasterio
import rasterio.features
import os
from shapely.geometry import box
 
 
#%% =============================================================================
# Paths
# =============================================================================
 
main_dir = r'C:\GitHub\Temp_CleaningTheGanga_Paper2'
 
# --- Input ---
mask_path   = os.path.join(main_dir, '1_model_input', '1_original_data', 'Model_Mask.tif')
canals_path = os.path.join(main_dir, '1_model_input', '1_original_data', 'irrigation_canals.geojson')
 
# --- Output ---
output_path = os.path.join(main_dir, '1_model_input', '2_temp_data', 'irrigation_canal_leakage.nc')
 
 
#%% =============================================================================
# Step 1 – Canal Area Per Cell
# =============================================================================
 
# --- Load canal network and model mask ---
canals = gpd.read_file(canals_path)
mask   = rioxarray.open_rasterio(mask_path).isel(band=0)
 
# --- Assign canal width by canal type ---
# Width is assumed to vary between 5 m and 100 m depending on canal order/type,
# based on typical dimensions in the Indo-Gangetic Basin (India-WRIS, n.d.).
def canal_type_to_width(order):
    """Return assumed canal width [m] for a given canal type code."""
    return {1: 100, 2: 50, 3: 25, 4: 5, 5: 5, 6: 5, 8: 50, 11: 25}.get(order, 2)
 
canals['width'] = canals['can_type'].apply(canal_type_to_width)
 
# --- Compute canal length per model cell [m] ---
# Each canal polyline is clipped to the bounding box of each grid cell.
# A spatial index is used to avoid redundant intersection tests.
def line_length_per_pixel(da: xr.DataArray, gdf: gpd.GeoDataFrame) -> xr.DataArray:
    """
    Compute the total length [m] of vector lines within each raster cell.
 
    Parameters
    ----------
    da  : xr.DataArray  Reference raster defining the grid (y, x dims).
    gdf : gpd.GeoDataFrame  Line geometries to measure; must share CRS with da.
 
    Returns
    -------
    xr.DataArray, shape (ny, nx)  Total line length [m] per cell.
    """
    x_coords = da['x'].values
    y_coords = da['y'].values
    dx = np.abs(x_coords[1] - x_coords[0])
    dy = np.abs(y_coords[1] - y_coords[0])
 
    pixel_shapes  = []
    pixel_indices = []
    for j, y in enumerate(y_coords):
        for i, x in enumerate(x_coords):
            pixel = box(x - dx / 2, y - dy / 2, x + dx / 2, y + dy / 2)
            pixel_shapes.append(pixel)
            pixel_indices.append((j, i))
 
    gdf_sindex   = gdf.sindex
    length_array = np.zeros((len(y_coords), len(x_coords)))
 
    for geom, (j, i) in zip(pixel_shapes, pixel_indices):
        candidates = gdf.iloc[list(gdf_sindex.intersection(geom.bounds))]
        total_length = 0.0
        for line in candidates.geometry:
            if not line.intersects(geom):
                continue
            clipped = line.intersection(geom)
            if clipped.is_empty:
                continue
            if clipped.geom_type == 'LineString':
                total_length += clipped.length
            elif clipped.geom_type == 'MultiLineString':
                total_length += sum(seg.length for seg in clipped.geoms)
        length_array[j, i] = total_length
 
    return xr.DataArray(
        length_array,
        coords={'y': y_coords, 'x': x_coords},
        dims=('y', 'x'),
        name='line_length_per_pixel',
    )
 
 
# --- Rasterize a numeric canal attribute onto the model grid ---
# Canals are sorted ascending by value before rasterizing so that higher-value
# features overwrite lower ones (last-write wins in rasterio.features.rasterize).
def rasterize_canal_attribute(da: xr.DataArray, gdf: gpd.GeoDataFrame, attr: str) -> xr.DataArray:
    """
    Rasterize a numeric attribute from canal polylines onto the model grid.
 
    Parameters
    ----------
    da   : xr.DataArray      Reference raster defining the grid (y, x dims).
    gdf  : gpd.GeoDataFrame  Canal geometries with the attribute column.
    attr : str               Column name in gdf to rasterize.
 
    Returns
    -------
    xr.DataArray, shape (ny, nx)  Rasterized attribute values; 0 where no canal.
    """
    x_coords = da['x'].values
    y_coords = da['y'].values
    dx = np.abs(x_coords[1] - x_coords[0])
    dy = np.abs(y_coords[1] - y_coords[0])
 
    transform = rasterio.transform.from_origin(
        west=x_coords[0]  - dx / 2,
        north=y_coords[0] + dy / 2,
        xsize=dx,
        ysize=dy,
    )
    shapes = zip(gdf.geometry, gdf[attr])
    raster = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=(len(y_coords), len(x_coords)),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype='float32',
    )
    return xr.DataArray(
        raster,
        coords={'y': y_coords, 'x': x_coords},
        dims=('y', 'x'),
        name=f'{attr}_raster',
    )
 
 
# Compute canal length per cell [m]
length_da = line_length_per_pixel(mask, canals)
 
# Rasterize canal width [m] — sort ascending so wider canals overwrite narrower
canals_sorted_width = canals.sort_values(by='width', ascending=True)
width_da = rasterize_canal_attribute(mask, canals_sorted_width, attr='width')
 
# Rasterize construction year — sort descending so earliest year is kept per cell
# (i.e. the oldest canal in a cell determines when leakage begins)
canals_sorted_year = canals.sort_values(by='construction_year', ascending=False)
construction_year_da = rasterize_canal_attribute(mask, canals_sorted_year, attr='construction_year')
construction_year_da = xr.where(construction_year_da != 0, construction_year_da, np.nan)
 
# Canal area per cell [m²] = length [m] × width [m]
# Cells with no canal are set to NaN.
canal_area_da = width_da * length_da
canal_area_da = xr.where(canal_area_da != 0, canal_area_da, np.nan)
 
 
#%% =============================================================================
# Step 2 – Temporal Activation by Construction Year (1800–2024)
# =============================================================================
 
# Each cell's canal area is activated from its construction year onward.
# Canal leakage is modelled from 1800, but meaningful contributions begin with
# the Eastern Yamuna Canal (1830) and Upper Ganga Canal (1856). Cells whose
# construction year exceeds the current simulation year are set to zero.
years = np.arange(1800, 2025)
 
area_per_year_da = xr.DataArray(
    np.zeros((len(years), canal_area_da.sizes['y'], canal_area_da.sizes['x'])),
    dims=('year', 'y', 'x'),
    coords={'year': years, 'y': canal_area_da['y'], 'x': canal_area_da['x']},
)
 
for year in years:
    # Retain canal area only for canals constructed on or before this year
    area_per_year_da.loc[dict(year=year)] = canal_area_da.where(
        construction_year_da <= year, 0
    )
 
 
#%% =============================================================================
# Step 3 – Resample to Monthly Frequency
# =============================================================================
 
# Each annual canal area value is repeated for all 12 months of that year,
# producing a monthly time series with constant area within each year.
monthly_dates = pd.date_range(
    start=f'{years[0]}-01-01',
    end=f'{years[-1]}-12-31',
    freq='MS',
)
 
# da_monthly: canal area [m²] per cell per month, shape (n_months, ny, nx)
da_monthly = xr.DataArray(
    np.repeat(area_per_year_da.values, 12, axis=0),
    coords={'time': monthly_dates, 'y': area_per_year_da.y, 'x': area_per_year_da.x},
    dims=['time', 'y', 'x'],
)


#%% =============================================================================
# Step 4 – Resample to Monthly Frequency and Export to NetCDF
# =============================================================================

# The leakage factor is on Raza et al. (2013), who reported seepage losses of 0.32 L s⁻¹ per 100 m of unlined canal. 
# Assuming an average canal width of 25 m in that study, 
# this corresponds to an approximate leakage rate of 11 mm day⁻¹ m⁻² of canal area.
# and thus  0.3366 m/month per m2 canal area
canal_leakage_factor =  0.3366 # m/month canal leakage per m2 canal area

# calculate canal leakage recharge
canal_leakage_recharge = da_monthly * canal_leakage_factor 
 
# --- Export to NetCDF ---
canal_leakage_recharge.to_netcdf(output_path)
print(f"Canal leakage recharge saved to: {output_path}")

