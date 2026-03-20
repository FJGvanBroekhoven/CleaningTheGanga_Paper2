# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
"""
Author : Frank van Broekhoven
=======================
Generate figures for the MODFLOW model of the Ganges–Yamuna interfluve.

Figures produced
----------------
Figure 1A – Study area map
    Map showing the model domain with DEM, river network, irrigation canals,
    industrial locations and city labels.

Figure 1B – MODFLOW cell type categories
    Categorical map showing the spatial distribution of boundary condition
    types (rivers, drains, constant-head cells) and the active model domain.

Figure 1C – Location of study area within the Indo-Gangetic Basin
    Locator map showing the study area bounding box within the IGB and
    broader national context.

Figure 4 – Recharge and abstraction budget over time
    Stacked bar chart of yearly recharge (+) and abstraction (−) components
    [mm/year] with a net recharge line overlay. A secondary y-axis shows
    equivalent values in m³/s.

Figure 6 – Spatial variation of recharge and abstraction components
    Grid of spatial maps showing the time-averaged recharge and abstraction
    [mm/year] for each component across five historical time periods.

"""

#%% =============================================================================
# Imports
# =============================================================================
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
import xarray as xr
import rioxarray
import geopandas as gpd
import os
from shapely.geometry import box


#%% =============================================================================
# Paths
# =============================================================================

main_dir = r'C:\GitHub\Temp_CleaningTheGanga_Paper2'

# --- Input ---
mask_path                = os.path.join(main_dir, '1_model_input', '1_original_data', 'Model_Mask.tif')
DRN_RIV_CHD_path         = os.path.join(main_dir, '1_model_input', '3_model_input_data', 'DRN_RIV_CHD.nc')
recharge_components_path = os.path.join(main_dir, '1_model_input', '3_model_input_data', 'recharge_and_abstraction_components_yearly.nc')
rivers_path              = os.path.join(main_dir, '1_model_input', '1_original_data', 'HydroSHEDS', 'HydroRIVERS_crop.geojson')
canals_path              = os.path.join(main_dir, '1_model_input', '1_original_data', 'irrigation_canals.geojson')
DEM_path                 = os.path.join(main_dir, '1_model_input', '3_model_input_data', 'DIS_TOP.tif')
hindon_catchment_path    = os.path.join(main_dir, '1_model_input', '1_original_data', 'Hindon_Subbasin_polygon.geojson')
industry_path            = os.path.join(main_dir, '1_model_input', '1_original_data', 'Industry.geojson')
label_path               = os.path.join(main_dir, '1_model_input', '1_original_data', 'labels_points.geojson')
countries_path           = os.path.join(main_dir, '1_model_input', '1_original_data', 'World_Countries_polygon.geojson')
IGB_path                 = os.path.join(main_dir, '1_model_input', '1_original_data', 'Indo_Gangetic_Basin_polygon.geojson')


#%% =============================================================================
# Load datasets
# =============================================================================

mask_da        = xr.open_dataarray(mask_path).squeeze(drop='band')
DRN_RIV_CHD_ds = xr.open_dataset(DRN_RIV_CHD_path).squeeze(drop='band')
components     = xr.open_dataset(recharge_components_path)
dem              = xr.open_dataarray(DEM_path).squeeze('band')
rivers           = gpd.read_file(rivers_path).to_crs("EPSG:32643")
canals           = gpd.read_file(canals_path).to_crs("EPSG:32643")
hindon_catchment = gpd.read_file(hindon_catchment_path).to_crs("EPSG:32643")
industry         = gpd.read_file(industry_path).to_crs("EPSG:32643")
labels           = gpd.read_file(label_path).to_crs("EPSG:32643")
IGB             = gpd.read_file(IGB_path).to_crs("EPSG:32643")
countries       = gpd.read_file(countries_path).to_crs("EPSG:32643")

#%% =============================================================================
# Figure 1A – Study area map
# =============================================================================

# --- River line width scaled by Strahler order ---
min_ord = rivers['ORD_STRA'].min()
max_ord = rivers['ORD_STRA'].max()
rivers['width'] = 0.5 + (rivers['ORD_STRA'] - min_ord) / (max_ord - min_ord) * 3

# --- Canal line width scaled by canal type ---
def canal_type_to_width(order):
    """Return assumed canal width [m] for a given canal type code."""
    return {1: 100, 2: 50, 3: 25, 4: 5, 5: 5, 6: 5, 8: 50, 11: 25}.get(order, 2)

canals['width'] = canals['can_type'].apply(canal_type_to_width)

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 10))
# DEM
dem.plot(ax=ax, cmap='YlOrBr', alpha=0.6, vmin=200, vmax=700,
         add_colorbar=True, cbar_kwargs={'label': 'Elevation (m)'})

# Plot cities as large grey squares
labels[labels['Type'] == 'city'].plot(
    ax=ax, color='grey', edgecolor='lightgrey',
    markersize=200, marker='s', linewidth=1.5,
    )

# Rivers
rivers.plot(ax=ax, color='#1f78b4', linewidth=rivers['width'], label='Rivers')

# Canals
canals.plot(ax=ax, color='#33a02c', linewidth=canals['width'] / 25,
            label='Irrigation canals')

# Hindon Catchment boundary
hindon_catchment.boundary.plot(
    ax=ax, color='dimgray', linewidth=2, linestyle=':', label='Hindon Catchment'
)

# Industry
industry.plot(ax=ax, color='#e66101', markersize=20, edgecolor='#fdb863',
              linewidth=0.6, alpha=0.9, label='Industry')

# Scale bar and Layout
ax.add_artist(ScaleBar(dx=1, units='m', location='lower left', scale_loc='bottom'))
ax.set_axis_off()
plt.tight_layout()
plt.show()


#%% =============================================================================
# Figure 1B – MODFLOW cell type categories
# =============================================================================

# Each boundary condition variable is mapped to an integer category value.
# The resulting categorical raster is plotted with a discrete colormap.
cat_values = {
    'RIV_Hindon_level':  1,
    'DRN_Hindon_level':  2,
    'RIV_other_level':   3,
    'DRN_other_level':   4,
    'CHD_Ganga_level':   5,
    'CHD_Yamuna_level':  6,
    'CHD_southern_level': 7,
}

cat_colors = {
    -1: 'lightgrey',   # non-active cell
     0: 'white',       # active cell (no boundary condition)
     1: '#1F78B4',     # Hindon river # 'blue'
     2: 'lightblue',   # Hindon drains
     3: 'green',       # other rivers
     4: 'lightgreen',  # other drains
     5: 'Magenta',     # Ganga constant-head boundary
     6: 'purple',      # Yamuna constant-head boundary
     7: 'gray',        # southern boundary
}

cat_names = {
    -1: 'non active cell',
     0: 'active cell',
     1: 'Hindon river',
     2: 'Hindon drains',
     3: 'other rivers',
     4: 'other drains',
     5: 'Ganga river boundary',
     6: 'Yamuna river boundary',
     7: 'southern boundary',
}

# Build categorical raster: assign integer value per boundary type,
# then overwrite cells outside the model domain with -1.
cat = xr.zeros_like(DRN_RIV_CHD_ds[list(cat_values.keys())[0]])
for var, value in cat_values.items():
    if var in DRN_RIV_CHD_ds:
        cat = xr.where(DRN_RIV_CHD_ds[var].notnull(), value, cat)
cat = xr.where(mask_da == -1, -1, cat)
cat.name = 'boundary_type'

vals     = sorted(cat_colors.keys())
cmap_cat = ListedColormap([cat_colors[v] for v in vals])
norm_cat = BoundaryNorm([v - 0.5 for v in vals] + [vals[-1] + 0.5], len(vals))

fig, ax = plt.subplots(figsize=(8, 10))
cat.plot(ax=ax, cmap=cmap_cat, norm=norm_cat, add_colorbar=False)
handles = [mpatches.Patch(color=cat_colors[v], label=cat_names[v]) for v in vals]
ax.legend(handles=handles, title='Boundary types', loc='upper left',
          bbox_to_anchor=(1.02, 1), borderaxespad=0)
ax.set_title('MODFLOW cell type categories')
labels.plot(ax=ax, color=None, edgecolor=None, markersize=0, alpha=0) # don't show but use geopandas to make sure that proportions are fixed
plt.subplots_adjust(right=0.75)
ax.set_axis_off()
plt.tight_layout()
plt.show()


#%% =============================================================================
# Figure 1C – Location of study area within the Indo-Gangetic Basin
# =============================================================================

countries = countries[countries.is_valid].dissolve()   # merge into single polygon

# Derive study area bounding box from model mask extent
study_bbox = gpd.GeoDataFrame(
    geometry=[box(float(mask_da.x.min()), float(mask_da.y.min()),
                  float(mask_da.x.max()), float(mask_da.y.max()))],
    crs="EPSG:32643",
)

# Plot locator map
fig, ax = plt.subplots(figsize=(6, 6))
countries.plot(ax=ax, color='lightgrey')
# Highlight Indo-Gangetic Basin
IGB.plot(ax=ax, color='red', edgecolor=None, alpha=0.3, label='Indo-Gangetic Basin')
# Plot study area bounding box
study_bbox.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2,
                label='Study area')
ax.set_title('Study Area in Indo-Gangetic Basin', fontsize=12)
ax.set_axis_off()
plt.tight_layout()
plt.show()


#%% =============================================================================
# Figure 4 – Recharge and abstraction budget over time
# =============================================================================
# Stacked bar chart of yearly recharge (+) and abstraction (−) components
# [mm/year]. Net recharge (sum of all components) is overlaid as a black line.
# A secondary y-axis shows the equivalent flux in m³/s.
# Unit conversion: 1 mm/year over model domain → multiply m³/year by
# (1 mm / 1000 mm per m) / domain_area_m2, where domain_area = 8 143 km².

# --- Component ordering, labels and colours (used in Figures 4 and 6) ---
component_order = [
    'municipal_return_rch',
    'canal_rch',
    'irrigation_return_rch',
    'rainfall_rch',
    'industrial_demand_abs',
    'municipal_demand_abs',
    'irrigation_demand_abs',
]

label_names = {
    'municipal_return_rch':  'Municipal Return Flow (RCH)',
    'canal_rch':             'Canal Leakage (RCH)',
    'irrigation_return_rch': 'Irrigation Return Flow (RCH)',
    'rainfall_rch':          'Rainfall Recharge (RCH)',
    'industrial_demand_abs': 'Industrial Demand (ABS)',
    'municipal_demand_abs':  'Municipal Demand (ABS)',
    'irrigation_demand_abs': 'Irrigation Demand (ABS)',
}

color_dict = {
    'municipal_return_rch':  '#08306B',   # deep navy
    'canal_rch':             '#1F78B4',   # medium blue
    'irrigation_return_rch': '#6BAED6',   # cornflower blue
    'rainfall_rch':          '#A6CEE3',   # light sky blue
    'industrial_demand_abs': '#E31A1C',   # strong red
    'municipal_demand_abs':  '#FF7F00',   # orange
    'irrigation_demand_abs': '#FDBF6F',   # light orange
}

# --- Resample components to yearly and compute spatial sum [m³/year] ---
components_annual = components.resample(time='YE').sum(dim='time', skipna=False)
net_recharge_annual = sum(
    components_annual[v].fillna(0) for v in components_annual.data_vars
)
 
per_ts        = components_annual.sum(dim=['x', 'y'], skipna=True)
components_df = per_ts.to_dataframe().drop(columns=['spatial_ref'], errors='ignore')
components_df = components_df[component_order]
net_rch_series = net_recharge_annual.sum(dim=['x', 'y'], skipna=True).to_series()
 
# Unit conversions
conversion_to_mm  = 1000 / 8_143_000_000       # m³/year → mm/year over domain
conversion_to_m3s = 1 / (365 * 24 * 60 * 60)   # m³/year → m³/s
 
components_df_mm  = components_df * conversion_to_mm
components_df_m3s = components_df * conversion_to_m3s
net_rch_mm        = net_rch_series * conversion_to_mm
 
colors = [color_dict[col] for col in components_df_mm.columns]
x      = np.arange(len(components_df_mm))
 
fig, ax1 = plt.subplots(figsize=(14, 7))
 
# Stacked bars (primary axis, mm/year)
components_df_mm.plot(kind='bar', stacked=True, width=1, ax=ax1,
                      zorder=2, color=colors, legend=False)
 
# Invisible stacked bars on secondary axis (m³/s) for tick alignment
ax2 = ax1.twinx()
components_df_m3s.plot(kind='bar', stacked=True, width=1, ax=ax2,
                       zorder=1, alpha=0.0, legend=False)
 
# Net recharge line
line = ax1.plot(x, net_rch_mm, color='black', linewidth=1.5, label='Net Recharge')
line[0].set_zorder(100)
 
ax1.set_xticks(x[::10])
ax1.set_xticklabels(components_df_mm.index.year[::10])
ax1.set_xlabel('Year')
ax1.set_ylabel('mm/year')
ax2.set_ylabel('m³/s')
 
bar_handles, _ = ax1.get_legend_handles_labels()
labels_to_show  = ['Net Recharge'] + [label_names[col] for col in components_df_mm.columns]
handles = bar_handles + [line[0]]
ax1.legend(handles, labels_to_show, loc='upper left')
 
plt.tight_layout()
plt.show()
 
 
#%% =============================================================================
# Figure 6 – Spatial variation of recharge and abstraction components
# =============================================================================
# Grid of maps (rows = components + net recharge, columns = time periods)
# showing time-averaged annual flux [mm/year] per cell.
# Red = abstraction (negative), blue = recharge (positive).
# Cells outside the model domain are shown in light grey.
 
# Convert yearly totals from m³/cell/year to mm/year (cell = 1 km²)
components_mm = components_annual / 1000

# Use the original variable order as they appear in the dataset
vars_to_plot = list(components.data_vars)

time_slices = {
    '1800–1830': (0,   29),
    '1830–1900': (30,  99),
    '1900–1970': (100, 169),
    '1970–2000': (170, 199),
    '2000–2016': (200, 216),
}

vmin, vmax = -1000, 1000
levels     = np.linspace(vmin, vmax, 22)
cmap_div   = plt.get_cmap('RdBu')
norm_div   = mcolors.BoundaryNorm(levels, ncolors=cmap_div.N, clip=True)


nrows = len(time_slices)
ncols = len(vars_to_plot) + 1   # one extra column for net recharge

fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                         figsize=(4 * ncols, 3 * nrows),
                         constrained_layout=True)

time_items = list(time_slices.items()) 

for r, (period_label, (t0, t1)) in enumerate(time_items):
    for c in range(ncols):
        ax = axes[r, c]

        if c < len(vars_to_plot):   
            da = (components_mm[vars_to_plot[c]]
                  .isel(time=slice(t0, t1 + 1))
                  .mean(dim='time', skipna=True))
        else:
            da = sum(
                components_mm[v].isel(time=slice(t0, t1 + 1)).mean(dim='time', skipna=True)
                for v in vars_to_plot
            )

        bg = da.copy()
        bg[:] = 1
        bg.plot(ax=ax, color='lightgrey', add_colorbar=False)
        im = da.plot(ax=ax, cmap=cmap_div, norm=norm_div, add_colorbar=False)
        labels.plot(ax=ax, color=None, edgecolor=None, markersize=0, alpha=0) # don't show but use geopandas to make sure that proportions are fixed

        # Column header (variables) on first row
        if r == 0:
            col_label = (
                label_names.get(vars_to_plot[c], vars_to_plot[c])
                if c < len(vars_to_plot) else 'Net Recharge'
            )
            ax.set_title(col_label, fontsize=10, fontweight='bold')
        else:
            ax.set_title('')
        ax.set_axis_off()

    # Row label (periods) on the left
    axes[r, 0].text(-0.05, 0.5, period_label,
                    transform=axes[r, 0].transAxes,
                    ha='right', va='center', fontsize=9, fontweight='bold')

fig.colorbar(im, ax=axes, orientation='vertical',
             boundaries=levels, ticks=np.linspace(vmin, vmax, 5),
             spacing='proportional', fraction=0.02, pad=0.02,
             label='mm/year')

plt.show()