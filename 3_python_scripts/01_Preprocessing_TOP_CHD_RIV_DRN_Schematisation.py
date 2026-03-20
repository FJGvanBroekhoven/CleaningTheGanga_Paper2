# -*- coding: utf-8 -*-
"""
Author : Frank van Broekhoven
===================================
Generate static spatial input rasters for the MODFLOW 6 boundary condition
packages (DRN, RIV, CHD) and TOP package for Upper Ganges–Yamuna interfluve.

Overview
--------
This script derives the spatial schematisation of surface-water boundary
conditions from river network and topographic data. It produces a single
NetCDF dataset containing the area and/or head level for each boundary
condition type used in the MODFLOW 6 model.
The script starts from the raw HydroSHEDS DEM tiles, which are merged,
reprojected, and resampled to the model grid before being used.

Model configuration
-------------------
The MODFLOW 6 model uses a transient, single-layer, regular grid of
231 rows × 108 columns with 1000 m × 1000 m cells. Boundaries are
assigned based on Strahler stream order and spatial location:
    
    TOP (DIS package)
        The resampled mean of the HydroSHEDS DEM

    DRN (Drain package)
        Small streams and drains (Strahler order ≤ 2, width < 20 m) that
        typically dry up outside the monsoon season. The DRN package allows
        only exfiltration (groundwater discharge to the drain).

    RIV (River package)
        The Hindon River and its main tributaries (20 m ≤ width ≤ 95 m,
        Strahler order 3–4). The RIV package allows both infiltration and
        exfiltration.

    CHD (General Head Boundary package)
        Major rivers forming the model's lateral boundaries: the Ganges
        (eastern) and Yamuna (western) rivers (width > 95 m, Strahler
        order ≥ 5). The southern no-natural-boundary edge is also
        assigned a GHB.

Processing steps
----------------
1. Merge the two HydroSHEDS DEM tiles (n20e070, n30e070), reproject to
   EPSG:32643.
2. Resample the merged DEM to the model grid using mean and minimum
   aggregation; export both resampled DEMs as GeoTIFFs.
3. Load input data: river network (HydroRIVERS), model mask, and Hindon
   catchment mask.
4. Compute river length per model cell by clipping each river segment
   to the corresponding cell polygon using a spatial index.
5. Assign river width from a Strahler-order lookup table and rasterize
   onto the model grid.
6. Derive river bed area (width × length) and bed level (minimum DEM).
7. Split into DRN, RIV, and CHD based on width thresholds.
8. Assign boundaries to the Hindon sub-catchment, the wider model domain,
   and the Ganges/Yamuna/southern boundary zones using the mask values.
9. Export the combined boundary condition dataset to NetCDF.

Data sources
------------
HydroSHEDS   – Lehner et al. (2008)
"""

#%% =============================================================================
# Imports
# =============================================================================
import os
import numpy as np
import geopandas as gpd
import xarray as xr
import rioxarray
import rasterio
import rasterio.features
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from shapely.geometry import box


#%% =============================================================================
# Paths
# =============================================================================

main_dir = r'C:\GitHub\Temp_CleaningTheGanga_Paper2'

# --- Input ---
dem_tile_1_path = os.path.join(main_dir, '1_model_input', '1_original_data', 'HydroSHEDS', 'n20e070_con.tif')
dem_tile_2_path = os.path.join(main_dir, '1_model_input', '1_original_data','HydroSHEDS', 'n30e070_con.tif')
path_rivers     = os.path.join(main_dir, '1_model_input', '1_original_data','HydroSHEDS', 'HydroRIVERS_crop.geojson')
mask_path       = os.path.join(main_dir, '1_model_input', '1_original_data', 'Model_Mask.tif')
Hindon_path     = os.path.join(main_dir, '1_model_input', '1_original_data', 'Hindon_catchment_mask.tif')

# --- Output ---
dem_mean_path   = os.path.join(main_dir, '1_model_input', '3_model_input_data', 'DIS_TOP.tif')
outpath_ds      = os.path.join(main_dir, '1_model_input', '3_model_input_data', 'DRN_RIV_CHD.nc')



#%% =============================================================================
# Step 1 – Merge HydroSHEDS DEM tiles
# =============================================================================
# The HydroSHEDS DEM covering the model domain is distributed across two
# 10° × 10° tiles (n20e070 and n30e070). They are concatenated along the
# y-dimension, sorted to ensure a monotonic coordinate axis, and reprojected
# from geographic (EPSG:4326) to the model CRS (EPSG:32643).

dem_1 = rioxarray.open_rasterio(dem_tile_1_path)
dem_2 = rioxarray.open_rasterio(dem_tile_2_path)

# Concatenate tiles along the latitude (y) axis and sort coordinates
combined_dem = xr.concat([dem_1, dem_2], dim='y')
combined_dem = combined_dem.sortby('x').sortby('y')

# Reproject to model CRS from "EPSG:4326" to "EPSG:32643"
combined_dem = combined_dem.rio.reproject("EPSG:32643")

#combined_dem.plot()
#plt.title('Merged HydroSHEDS DEM (EPSG:32643)')
#plt.show()


#%% =============================================================================
# Step 2 – Resample DEM to model grid (mean and minimum)
# =============================================================================
# The merged fine-resolution DEM is resampled to the 1 000 m model grid using
# two aggregation methods:
#   - Mean resampling  → used as model cell top elevation (TOP package)
#   - Minimum resampling → used as river/drain stage approximation
# Both outputs are saved as GeoTIFFs for use in this and other scripts.

# Load model mask to define target grid extent, resolution, and CRS
mask = rioxarray.open_rasterio(mask_path)
# Set the CRS ("EPSG:32643")
mask.rio.write_crs("EPSG:32643", inplace=True)
print(mask.rio.crs)

# Mean resampling: representative cell elevation for model top
dem_resampled_mean = combined_dem.rio.reproject_match(mask, resampling=Resampling.average)
dem_resampled_mean.rio.to_raster(dem_mean_path)
print(f"Mean resampled DEM saved to: {dem_mean_path}")

# Minimum resampling: elevation for river/drain bed level
dem_resampled_min = combined_dem.rio.reproject_match(mask, resampling=Resampling.min)
#dem_resampled_min.rio.to_raster(dem_min_path)
#print(f"Minimum resampled DEM saved to: {dem_min_path}")

# Quick plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
dem_resampled_mean.squeeze().plot(ax=axes[0], robust=True)
axes[0].set_title('DEM – mean resampled [m a.s.l.]')
dem_resampled_min.squeeze().plot(ax=axes[1], vmin=200, vmax=300)
axes[1].set_title('DEM – minimum resampled [m a.s.l.]')
plt.tight_layout()
plt.show()


#%% =============================================================================
# Step 3 – Load remaining input data
# =============================================================================

# --- River network (HydroRIVERS) ---
# Sort by Strahler order so higher-order (wider) rivers overwrite lower-order
# ones when rasterizing widths.
rivers = gpd.read_file(path_rivers)
rivers = rivers.sort_values(by='ORD_STRA', ascending=True)

# Manual correction: the Kali Nadi south-east of Ghaziabad has ORD_STRA = 5
# but ORD_CLAS = 4 in the source data, which would incorrectly assign it to
# the GHB package. Reclassify to ORD_STRA = 4 so it is treated as RIV.
rivers.loc[(rivers['ORD_STRA'] == 5) & (rivers['ORD_CLAS'] == 4), 'ORD_STRA'] = 4

# --- Hindon catchment mask ---
# Binary mask: 1 = inside Hindon sub-catchment, 0 = outside
Hindon = rioxarray.open_rasterio(Hindon_path)


#%% =============================================================================
# Step 4 – Compute river length per model cell
# =============================================================================
# Each river segment is clipped to the bounding box of each model cell.
# The total clipped length of all segments within a cell is summed.

def line_length_per_pixel(da: xr.DataArray, gdf: gpd.GeoDataFrame) -> xr.DataArray:
    """
    Calculate the total river line length [m] within each model grid cell.

    For each cell, the function clips all river geometries that intersect the
    cell's bounding box and sums the resulting lengths.

    Parameters
    ----------
    da : xr.DataArray
        2-D DataArray defining the model grid (must have 'x' and 'y' coords
        in a projected CRS with units of metres).
    gdf : gpd.GeoDataFrame
        River line geometries, reprojected to the same CRS as `da`.

    Returns
    -------
    xr.DataArray, shape (ny, nx)
        Total river length [m] per model cell.
    """
    x_coords = da['x'].values
    y_coords = da['y'].values
    dx = np.abs(x_coords[1] - x_coords[0])
    dy = np.abs(y_coords[1] - y_coords[0])

    # Build cell polygon list with (row, col) index for array assignment
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
        candidate_idx = list(gdf_sindex.intersection(geom.bounds))
        candidates    = gdf.iloc[candidate_idx]

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
        name='river_length_per_cell',
    )


# length_da: total river length [m] within each model cell
length_da = line_length_per_pixel(mask.isel(band=0), rivers)


#%% =============================================================================
# Step 5 – Rasterize river width onto model grid
# =============================================================================
# Each river segment is assigned a representative width from the Strahler-order
# lookup table. Segments are burned onto the model grid; where multiple segments
# overlap a cell, the last-written value is retained (segments are pre-sorted by
# ascending order, so wider rivers overwrite narrower ones).

def strahler_to_width(order: int) -> int:
    """
    Return representative channel width [m] for a given Strahler stream order.

    Parameters
    ----------
    order : int
        Strahler stream order (1–6).

    Returns
    -------
    int
        Representative channel width [m] from STRAHLER_WIDTH lookup.
        Returns 2 m for orders not present in the table.
    """
    return STRAHLER_WIDTH.get(order, 2)

# Strahler-order to representative channel width [m] lookup.
# Based on typical channel widths in the Indo-Gangetic Plain.
STRAHLER_WIDTH = {1: 5, 2: 10, 3: 25, 4: 50, 5: 100, 6: 200}
rivers['width'] = rivers['ORD_STRA'].apply(strahler_to_width)


def rasterize_river_width(
    da: xr.DataArray,
    gdf: gpd.GeoDataFrame,
    attr: str = 'width',
) -> xr.DataArray:
    """
    Burn a river attribute (default: channel width) onto the model grid.

    Parameters
    ----------
    da : xr.DataArray
        Reference DataArray defining grid extent, resolution, and coordinates.
    gdf : gpd.GeoDataFrame
        River geometries with the attribute column to be rasterized.
    attr : str, optional
        Name of the GeoDataFrame column to burn. Default is 'width'.

    Returns
    -------
    xr.DataArray, shape (ny, nx)
        Rasterized attribute values; cells without any river segment receive 0.
    """
    x_coords = da['x'].values
    y_coords = da['y'].values
    dx = np.abs(x_coords[1] - x_coords[0])
    dy = np.abs(y_coords[1] - y_coords[0])

    transform = rasterio.transform.from_origin(
        west=x_coords[0] - dx / 2,
        north=y_coords[0] + dy / 2,
        xsize=dx,
        ysize=dy,
    )

    raster = rasterio.features.rasterize(
        shapes=zip(gdf.geometry, gdf[attr]),
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


# width_da: representative river width [m] per model cell
width_da = rasterize_river_width(mask.isel(band=0), rivers, attr='width')


#%% =============================================================================
# Step 6 – Derive river bed area and bed level
# =============================================================================

# river_area_da: river bed area per cell [m²] = width × in-cell river length.
# Cells with no river segment (area = 0) are set to NaN.
river_area_da = width_da * length_da
river_area_da = xr.where(river_area_da != 0, river_area_da, np.nan)

# river_level: river stage [m a.s.l.] = minimum DEM elevation per cell.
# The minimum DEM value approximates the river bed / thalweg elevation, which
# is used as the boundary head for RIV, DRN, and CHD packages
river_level = xr.where(river_area_da > 0, dem_resampled_min, np.nan)


#%% =============================================================================
# Step 7 – Assign boundary condition types (DRN / RIV / CHD)
# =============================================================================
# Boundaries are split by river width, which serves as a proxy for stream order
# and the degree of groundwater–surface water connectivity:
#   width < 20 m           → DRN  (small streams / drains; exfiltration only)
#   20 m ≤ width ≤ 95 m   → RIV  (Hindon River and major tributaries)
#   width > 95 m           → CHD  (Ganges and Yamuna boundary rivers)

# Width thresholds [m] separating DRN / RIV / GHB boundary condition types.
# Based on Strahler-order-derived widths: orders 1–2 → DRN, 3–4 → RIV,
# ≥ 5 → GHB (Ganges, Yamuna).
WIDTH_DRN_MAX = 20    # Cells with width < 20 m → DRN
WIDTH_RIV_MAX = 95    # Cells with 20 m ≤ width ≤ 95 m → RIV; > 95 m → GHB

DRN_area  = xr.where(width_da < WIDTH_DRN_MAX, river_area_da, np.nan)
DRN_level = xr.where(width_da < WIDTH_DRN_MAX, river_level,   np.nan)

RIV_area  = xr.where((width_da >= WIDTH_DRN_MAX) & (width_da <= WIDTH_RIV_MAX), river_area_da, np.nan)
RIV_level = xr.where((width_da >= WIDTH_DRN_MAX) & (width_da <= WIDTH_RIV_MAX), river_level,   np.nan)

CHD_level = xr.where(width_da > WIDTH_RIV_MAX, river_level, np.nan)


#%% =============================================================================
# Step 8 – Build output dataset: split by sub-catchment and boundary zone
# =============================================================================
# Boundaries are further split using the Hindon catchment mask and the model
# mask zone values to allow separate conductance parameterisation in MODFLOW 6:
#
#   DRN_Hindon  / DRN_other  – drains inside / outside the Hindon sub-catchment
#   RIV_Hindon  / RIV_other  – rivers inside / outside the Hindon sub-catchment
#   CHD_Ganga               – Ganges boundary (mask == MASK_GANGES)
#   CHD_Yamuna              – Yamuna boundary (mask == MASK_YAMUNA)
#   CHD_southern            – Southern no-natural boundary (mask == MASK_SOUTH);
#                             head set to minimum DEM elevation

# Mask cell values identifying lateral boundary zones:
#   mask == 1  → model interior (domain)
#   mask == 2  → Yamuna river boundary
#   mask == 3  → Ganges river boundary
#   mask == 0  → southern no-natural boundary
MASK_YAMUNA = 2
MASK_GANGES = 3
MASK_SOUTH  = 0

DRN_RIV_CHD_ds = xr.Dataset()

DRN_RIV_CHD_ds['DRN_Hindon_area']  = xr.where(Hindon == 1, DRN_area,  np.nan)
DRN_RIV_CHD_ds['DRN_Hindon_level'] = xr.where(Hindon == 1, DRN_level, np.nan)

DRN_RIV_CHD_ds['DRN_other_area']   = xr.where(Hindon == 0, DRN_area,  np.nan)
DRN_RIV_CHD_ds['DRN_other_level']  = xr.where(Hindon == 0, DRN_level, np.nan)

DRN_RIV_CHD_ds['RIV_Hindon_area']  = xr.where(Hindon == 1, RIV_area,  np.nan)
DRN_RIV_CHD_ds['RIV_Hindon_level'] = xr.where(Hindon == 1, RIV_level, np.nan)

DRN_RIV_CHD_ds['RIV_other_area']   = xr.where(Hindon == 0, RIV_area,  np.nan)
DRN_RIV_CHD_ds['RIV_other_level']  = xr.where(Hindon == 0, RIV_level, np.nan)

DRN_RIV_CHD_ds['CHD_Ganga_level']    = xr.where(mask == MASK_GANGES, CHD_level, np.nan)
DRN_RIV_CHD_ds['CHD_Yamuna_level']   = xr.where(mask == MASK_YAMUNA, CHD_level, np.nan)
DRN_RIV_CHD_ds['CHD_southern_level'] = xr.where(mask == MASK_SOUTH,  dem_resampled_min, np.nan)


#%% =============================================================================
# Diagnostic plots
# =============================================================================

DRN_RIV_CHD_ds['DRN_Hindon_level'].plot(robust=True)
DRN_RIV_CHD_ds['DRN_other_level'].plot(robust=True)
DRN_RIV_CHD_ds['RIV_Hindon_level'].plot(robust=True)
DRN_RIV_CHD_ds['RIV_other_level'].plot(robust=True)
DRN_RIV_CHD_ds['CHD_Ganga_level'].plot(robust=True)
DRN_RIV_CHD_ds['CHD_Yamuna_level'].plot(robust=True)
DRN_RIV_CHD_ds['CHD_southern_level'].plot(robust=True)


#%% =============================================================================
# Export to NetCDF
# =============================================================================

DRN_RIV_CHD_ds.to_netcdf(outpath_ds)
print(f"DRN / RIV / CHD dataset saved to: {outpath_ds}")

