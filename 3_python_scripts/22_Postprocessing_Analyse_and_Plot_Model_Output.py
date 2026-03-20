# -*- coding: utf-8 -*-
"""
Author : Frank van Broekhoven
=======================
Generate figures from the MODFLOW Monte Carlo model output for the MODFLOW model 
of the Ganges–Yamuna interfluve.
 
Figures produced
----------------
Figure 3  – Parameter distributions
    Histograms comparing prior (grey) and posterior valid-run (blue)
    distributions for all 12 Monte Carlo parameters.
 
Figure 5  – Net recharge time series
    Ensemble min–mean–max of net recharge [m³/s] across valid model runs.
 
Figure 7  – Median groundwater depth time series
    Ensemble min–mean–max of spatially-median groundwater table depth
    below ground level [m bGL] across valid model runs.
 
Figure 8  – Spatial groundwater depth maps
    Part 1: Mean head depth below ground level [m bGL] per historical period.
    Part 2: Change in head depth between consecutive periods [m bGL].
 
Figure 9  – GW–SW exchange time series
    Ensemble min–mean–max of groundwater–surface water exfiltration fluxes
    [m³/s] for the Hindon River, other rivers, the Ganga, and the Yamuna.
 
Figure 10 – Spatial GW–SW interaction maps
    Part 1: Mean GW–SW exchange [m³/year per cell] per historical period.
    Part 2: Change in GW–SW exchange between consecutive periods [m³/year].
"""
 
#%% =============================================================================
# Imports
# =============================================================================
 
import json
import os
from datetime import datetime
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from matplotlib.colors import BoundaryNorm, ListedColormap
 
 
#%% =============================================================================
# Paths and settings
# =============================================================================
 
main_dir = r'C:\GitHub\Temp_CleaningTheGanga_Paper2'
 
#outdir = r'C:\temp_model_output\1500_runs_FINAL_v04'
outdir    = os.path.join(main_dir, '2_model_output', 'model_output_data', '1500_MC_Runs_UGYI')
mask_path = os.path.join(main_dir, '1_model_input', '1_original_data', 'Model_Mask.tif')
label_path = os.path.join(main_dir, '1_model_input', '1_original_data', 'labels_points.geojson')
 
mask_da = xr.open_dataarray(mask_path).squeeze(drop='band')
labels = gpd.read_file(label_path).to_crs("EPSG:32643")
 
DAYS_PER_STRESS_PERIOD = 365.25
PERIOD_LABELS          = ["1800-1830", "1830-1900", "1900-1970", "1970-2000", "2000-2016"]
 
# Unit conversions applied to budget xarray (m³/stress period → m³/s)
CONVERSION_TO_M3S = 1 / (DAYS_PER_STRESS_PERIOD * 24 * 60 * 60)
 
# Monte Carlo parameter space (used for Figure 3 histogram bins and log-scaling)
MC_PARAMS = {
    'k':                            {'dist': 'uniform',     'min': 5,     'max': 100},
    'd':                            {'dist': 'uniform',     'min': 50,    'max': 250},
    'RIV_cond':                     {'dist': 'log-uniform', 'min': 0.001, 'max': 10},
    'DRN_cond':                     {'dist': 'log-uniform', 'min': 0.001, 'max': 10},
    'sy':                           {'dist': 'uniform',     'min': 0.05,  'max': 0.35},
    'rainfall_rch_factor':          {'dist': 'log-uniform', 'min': 0.5,   'max': 2},
    'canal_rch_factor':             {'dist': 'log-uniform', 'min': 0.25,  'max': 4},
    'irrigation_return_rch_factor': {'dist': 'log-uniform', 'min': 0.25,  'max': 4},
    'municipal_return_rch_factor':  {'dist': 'log-uniform', 'min': 0.25,  'max': 4},
    'irrigation_demand_abs_factor': {'dist': 'log-uniform', 'min': 0.25,  'max': 4},
    'municipal_demand_abs_factor':  {'dist': 'log-uniform', 'min': 0.25,  'max': 4},
    'industrial_demand_abs_factor': {'dist': 'log-uniform', 'min': 0.25,  'max': 4},
}
 
PARAM_LABELS = {
    'k':                            'Aquifer conductivity [m/d]',
    'd':                            'Aquifer thickness [m]',
    'RIV_cond':                     'Riverbed conductivity [d⁻¹]',
    'DRN_cond':                     'Drainbed conductivity [d⁻¹]',
    'sy':                           'Specific yield [-]',
    'rainfall_rch_factor':          'Rainfall recharge factor [-]',
    'canal_rch_factor':             'Canal recharge factor [-]',
    'irrigation_return_rch_factor': 'Irrigation return flow factor [-]',
    'municipal_return_rch_factor':  'Municipal return flow factor [-]',
    'irrigation_demand_abs_factor': 'Irrigation abstraction factor [-]',
    'municipal_demand_abs_factor':  'Municipal abstraction factor [-]',
    'industrial_demand_abs_factor': 'Industrial abstraction factor [-]',
}
 
# Validity criteria (applied to the overviews DataFrame)
VALIDITY_CRITERIA = {
    'pct_above_gl':     ('<=', 2.5),
    'pct_below_50mbgl': ('<=', 5.0),
    'MAE':              ('<=', 7.5),
    'mean_error':       ('between', -5, 5),
    'Q_avg_1997_1999':  ('<',  100),
    'Q_avg_1800_1825':  ('>=', 0),
}
 
 
#%% =============================================================================
# Load model output
# =============================================================================
 
def merge_overviews(outdir):
    """
    Load all per-run overview JSONs and return them as a single DataFrame.
 
    Parameters
    ----------
    outdir : str
        Root output directory containing an 'overview' subdirectory.
 
    Returns
    -------
    pd.DataFrame
        One row per model run, indexed by model_id.
    """
    overview_dir = os.path.join(outdir, 'overview')
    records = []
    for fname in os.listdir(overview_dir):
        if fname.endswith('.json'):
            with open(os.path.join(overview_dir, fname), 'r') as f:
                records.append(json.load(f))
    return pd.DataFrame(records).set_index('model_id')
 
 
def merge_budgets_to_xarray(outdir):
    """
    Load all per-run budget CSVs and return them as an xarray Dataset.
 
    Each budget variable (Hindon river, Ganga, net recharge, etc.) becomes a
    Dataset variable with dimensions (time, model_id).
 
    Parameters
    ----------
    outdir : str
 
    Returns
    -------
    xr.Dataset
        Dimensions: time, model_id.
    """
    budget_dir = os.path.join(outdir, 'budget')
    dfs = {}
    for fname in os.listdir(budget_dir):
        if fname.endswith('.csv'):
            model_id = '_'.join(fname.split('_')[:2])
            dfs[model_id] = pd.read_csv(
                os.path.join(budget_dir, fname), index_col=0, parse_dates=True
            )
 
    time_index = next(iter(dfs.values())).index
    variables  = next(iter(dfs.values())).columns
    model_ids  = list(dfs.keys())
 
    ds = xr.Dataset(coords={'time': time_index, 'model_id': model_ids})
    for var in variables:
        data = np.stack([dfs[mid][var].values for mid in model_ids], axis=1)
        ds[var] = (('time', 'model_id'), data)
 
    return ds.transpose('time', 'model_id')
 
 
def merge_median_heads(outdir):
    """
    Load all per-run median-head .npy files and return them as a DataFrame.
 
    Parameters
    ----------
    outdir : str
 
    Returns
    -------
    pd.DataFrame
        Columns = model_id, rows = stress periods.
    """
    folder = os.path.join(outdir, 'median_heads')
    merged = {}
    for fname in os.listdir(folder):
        if fname.endswith('.npy'):
            model_id = '_'.join(fname.split('_')[:2])
            merged[model_id] = np.load(os.path.join(folder, fname))
    return pd.DataFrame(merged)
 
 
def load_spatial_per_period(outdir, subfolder):
    """
    Load all per-run spatial period .npy files from a subfolder.
 
    Stacks arrays into a single (model_id, period, y, x) array.
 
    Parameters
    ----------
    outdir : str
    subfolder : str
        'heads_per_period' or 'GW_SW_per_period'.
 
    Returns
    -------
    data : np.ndarray, shape (n_models, n_periods, nrow, ncol)
    model_ids : list of str
    """
    folder    = os.path.join(outdir, subfolder)
    model_ids = []
    arrays    = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith('.npy'):
            model_ids.append('_'.join(fname.split('_')[:2]))
            arrays.append(np.squeeze(np.load(os.path.join(folder, fname))))
    return np.stack(arrays, axis=0), model_ids
 
 
def build_spatial_xarray(data, model_ids, period_labels, mask_da):
    """
    Wrap a (model_id, period, y, x) numpy array in an xarray DataArray.
 
    Parameters
    ----------
    data : np.ndarray, shape (n_models, n_periods, nrow, ncol)
    model_ids : list of str
    period_labels : list of str
    mask_da : xr.DataArray
        Used to supply y, x, and spatial_ref coordinates.
 
    Returns
    -------
    xr.DataArray
        Dimensions: model_id, period, y, x.
    """
    return xr.DataArray(
        data,
        dims=("model_id", "period", "y", "x"),
        coords={
            "model_id":    model_ids,
            "period":      period_labels,
            "y":           mask_da.y,
            "x":           mask_da.x,
            "spatial_ref": mask_da.spatial_ref,
        },
    )
 
 
#%% =============================================================================
# Assemble combined dataset
# =============================================================================
 
# --- Overviews ---
overviews = merge_overviews(outdir)
overviews['mean_error'] = overviews['Simulated Mean'] - overviews['Observed Mean']
 
# --- Budgets (converted to m³/s; sign flipped so positive = discharge) ---
budget_xarray      = merge_budgets_to_xarray(outdir)
budget_xarray_m3s  = budget_xarray * CONVERSION_TO_M3S * -1
 
# --- Median heads ---
median_heads = merge_median_heads(outdir)
median_heads_da = xr.DataArray(
    data=median_heads.values,
    dims=("time", "model_id"),
    coords={
        "time":     budget_xarray.time,
        "model_id": median_heads.columns.tolist(),
    },
)
 
# --- Combined time-series dataset ---
total_xarray = budget_xarray_m3s.copy()
total_xarray["median_heads"] = median_heads_da
 
# --- Discharge averages used in validity filtering ---
Hindon_discharge = total_xarray["Hindon river"]
 
avg_97_99 = (Hindon_discharge
             .sel(time=slice("1997-01-01", "1999-12-31"))
             .mean(dim="time")
             .to_pandas())
avg_97_99.name = "Q_avg_1997_1999"
 
avg_1800_1825 = (Hindon_discharge
                 .sel(time=slice("1800-01-01", "1825-12-31"))
                 .mean(dim="time")
                 .to_pandas())
avg_1800_1825.name = "Q_avg_1800_1825"
 
overviews = overviews.join(avg_97_99).join(avg_1800_1825)
 
# --- Spatial maps ---
heads_per_period_data, model_ids = load_spatial_per_period(outdir, 'heads_per_period')
heads_da = build_spatial_xarray(heads_per_period_data, model_ids, PERIOD_LABELS, mask_da)
 
gw_sw_per_period_data, model_ids = load_spatial_per_period(outdir, 'GW_SW_per_period')
GW_SW_da = build_spatial_xarray(gw_sw_per_period_data, model_ids, PERIOD_LABELS, mask_da)
 
 
#%% =============================================================================
# Select valid model runs
# =============================================================================
 
overviews_valid = overviews[
    (overviews['pct_above_gl']     <= 2.5) &
    (overviews['pct_below_50mbgl'] <= 5.0) &
    (overviews['MAE']              <= 7.5) &
    (overviews['mean_error']       <= 5.0) &
    (overviews['mean_error']       >= -5.0) &
    (overviews['Q_avg_1997_1999']  <  65) &
    (overviews['Q_avg_1800_1825']  >= 0)
]
print(f"Valid runs: {len(overviews_valid)} / {len(overviews)}")
 
total_valid    = total_xarray.sel(model_id=overviews_valid.index)
heads_da_valid = heads_da.sel(model_id=overviews_valid.index)
GW_SW_da_valid = GW_SW_da.sel(model_id=overviews_valid.index)
 
 
#%% =============================================================================
# Shared plot helpers
# =============================================================================
 
def plot_envelope(ax, da, color, label):
    """
    Plot a mean line with a shaded min–max envelope from a (time, model_id) DataArray.
 
    Parameters
    ----------
    ax : matplotlib.axes.Axes
    da : xr.DataArray
        Must have dimensions (time, model_id).
    color : str
    label : str
        Used as the legend label prefix for mean and envelope.
    """
    mean    = da.mean(dim='model_id')
    minimum = da.min(dim='model_id')
    maximum = da.max(dim='model_id')
    mean.plot(ax=ax, color=color, linewidth=2, label=f'{label} (mean)')
    ax.fill_between(mean.time, minimum, maximum,
                    color=color, alpha=0.35, label=f'{label} (min–max)')
 
 
def make_spatial_figure(da, cmap, norm, bounds, cbar_label, title_fn):
    """
    Plot one spatial panel per period with a shared horizontal colorbar.
 
    Parameters
    ----------
    da : xr.DataArray
        Dimensions (model_id, period, y, x). The ensemble mean is plotted.
    cmap : ListedColormap
    norm : BoundaryNorm
    bounds : list of float
        Colorbar tick positions.
    cbar_label : str
    title_fn : callable
        Receives the period coordinate value and returns the panel title string.
 
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_panels = len(da.period)
    fig = plt.figure(figsize=(5 * n_panels, 5))
    gs  = fig.add_gridspec(2, n_panels, height_ratios=[1, 0.05],
                           wspace=0.05, hspace=0.3)
 
    for i, period in enumerate(da.period):
        ax = fig.add_subplot(gs[0, i])
        da.sel(period=period).mean("model_id").plot(
            ax=ax, x='x', y='y', cmap=cmap, norm=norm, add_colorbar=False
        )
        labels.plot(ax=ax, color=None, edgecolor=None, markersize=0, alpha=0) # don't show but use geopandas to make sure that proportions are fixed
        ax.set_title(title_fn(period))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
 
    cbar_ax = fig.add_subplot(gs[1, :])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax, orientation='horizontal'
    )
    cbar.set_ticks(bounds)
    cbar.set_ticklabels([str(b) for b in bounds])
    cbar.set_label(cbar_label)
    return fig
 
 
def make_difference_figure(da, cmap, norm, bounds, cbar_label):
    """
    Plot the difference between consecutive periods with a shared colorbar.
 
    Parameters
    ----------
    da : xr.DataArray
        Dimensions (model_id, period, y, x). The ensemble mean is differenced.
    cmap : ListedColormap
    norm : BoundaryNorm
    bounds : list of float
    cbar_label : str
 
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_diff = len(da.period) - 1
    fig = plt.figure(figsize=(5 * n_diff, 5))
    gs  = fig.add_gridspec(2, n_diff, height_ratios=[1, 0.05],
                           wspace=0.05, hspace=0.3)
 
    for i in range(n_diff):
        ax   = fig.add_subplot(gs[0, i])
        curr = da.period[i + 1]
        prev = da.period[i]
        diff = (da.sel(period=curr).mean("model_id")
                - da.sel(period=prev).mean("model_id"))
        diff.plot(ax=ax, x='x', y='y', cmap=cmap, norm=norm, add_colorbar=False)
        labels.plot(ax=ax, color=None, edgecolor=None, markersize=0, alpha=0) # don't show but use geopandas to make sure that proportions are fixed
        ax.set_title(f"{curr.values} − {prev.values}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
 
    cbar_ax = fig.add_subplot(gs[1, :])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax, orientation='horizontal'
    )
    cbar.set_ticks(bounds)
    cbar.set_ticklabels([str(b) for b in bounds])
    cbar.set_label(cbar_label)
    return fig
 
 
#%% =============================================================================
# Figure 3 – Parameter distributions (prior vs posterior)
# =============================================================================
 
def plot_parameter_histograms(params, all_df, valid_df, param_labels, ncols=3):
    """
    Compare prior (grey) and posterior valid-run (blue) parameter distributions.
 
    Log-uniform parameters are plotted in log space; tick labels are converted
    back to original scale for readability.
 
    Parameters
    ----------
    params : dict
        MC_PARAMS specification dict.
    all_df, valid_df : pd.DataFrame
        Full and filtered overviews DataFrames.
    param_labels : dict
        Human-readable axis labels keyed by parameter name.
    ncols : int, optional
        Number of subplot columns (default 3).
    """
    n     = len(params)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
 
    for i, (param, cfg) in enumerate(params.items()):
        ax       = axes[i]
        min_val  = cfg['min']
        max_val  = cfg['max']
 
        if cfg['dist'] == 'uniform':
            bins         = np.linspace(min_val, max_val, 10)
            all_vals     = all_df[param].dropna()
            valid_vals   = valid_df[param].dropna()
        else:  # log-uniform: work in log space
            bins         = np.linspace(np.log(min_val), np.log(max_val), 10)
            all_vals     = np.log(all_df[param].dropna())
            valid_vals   = np.log(valid_df[param].dropna())
 
        ax.hist(all_vals,   bins=bins, color='lightgrey', alpha=0.5,
                label='all', edgecolor='white', linewidth=1)
        ax.hist(valid_vals, bins=bins, color='#1F78B4',   alpha=0.5,
                label='valid', edgecolor='white', linewidth=1)
 
        if cfg['dist'] == 'log-uniform':
            ticks = ax.get_xticks()
            ax.set_xticklabels([f"{np.exp(t):.2g}" for t in ticks])
 
        ax.set_xlabel(param_labels.get(param, param))
 
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
 
    plt.tight_layout()
    plt.show()
 
 
plot_parameter_histograms(MC_PARAMS, overviews, overviews_valid, PARAM_LABELS)
 
 
#%% =============================================================================
# Figure 5 – Net recharge time series
# =============================================================================
 
fig, ax = plt.subplots(figsize=(12, 5))
 
# Sign convention: positive = net recharge into aquifer
da = total_valid['net recharge'].transpose('time', 'model_id') * -1
plot_envelope(ax, da, color='grey', label='Net recharge')
 
ax.axhline(y=0, color='black', linewidth=1)
ax.axvline(x=datetime(1900, 12, 31), color='gray', linestyle='--', linewidth=2,
           label='Start CRU TS meteorological data')
ax.set_ylabel('Net recharge [m³/s]')
ax.set_xlabel('Year')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend()
plt.tight_layout()
plt.show()
 
 
#%% =============================================================================
# Figure 7 – Median groundwater depth time series
# =============================================================================
 
fig, ax = plt.subplots(figsize=(12, 5))
 
# Multiply by -1 so that depth below ground level plots as positive downward
da = total_valid['median_heads'].transpose('time', 'model_id') * -1
plot_envelope(ax, da, color='#1F78B4', label='Median depth bGL')
 
ax.axhline(y=0, color='black', linewidth=2)
ax.axvline(x=datetime(1900, 12, 31), color='gray', linestyle='--', linewidth=2,
           label='Start CRU TS meteorological data')
ax.invert_yaxis()   # deeper values plotted downward
ax.set_ylabel('Median depth below ground level [m]')
ax.set_xlabel('Year')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend()
plt.tight_layout()
plt.show()
 
 
#%% =============================================================================
# Figure 9 – GW–SW exchange time series
# =============================================================================
# Four stacked subplots sharing the x-axis. Y-axis limits are fixed per panel
# to ensure consistent visual weight. A positive flux indicates groundwater
# exfiltration to the river; negative indicates river infiltration to aquifer.
 
RIVERS = [
    ('Hindon river', 'blue',    (300, -200)),
    ('other rivers', 'green',   (300, -200)),
    ('Ganga',        'magenta', (150, -100)),
    ('Yamuna',       'purple',  (150, -100)),
]
height_ratios = [maxv - minv for _, _, (maxv, minv) in RIVERS]
 
fig, axes = plt.subplots(
    nrows=len(RIVERS), ncols=1,
    figsize=(12, 10),
    sharex=True,
    gridspec_kw={'height_ratios': height_ratios},
)
 
for ax, (river_name, color, (maxv, minv)) in zip(axes, RIVERS):
    da = total_valid[river_name].transpose('time', 'model_id')
    plot_envelope(ax, da, color=color, label=river_name)
 
    ax.axhline(0, color='black', linewidth=2)
    ax.axvline(datetime(1900, 12, 31), color='gray', linestyle='--', linewidth=1)
    ax.set_ylim(minv, maxv)
    ax.set_ylabel('Exfiltration [m³/s]')
    ax.set_xlabel('')
    ax.set_title('')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.text(0.01, 0.05, river_name, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=11, fontweight='bold')
 
for ax in axes[:-1]:
    ax.tick_params(labelbottom=False)
 
fig.supxlabel('Year')
plt.subplots_adjust(hspace=0.05)
plt.tight_layout()
plt.show()
 
 
#%% =============================================================================
# Figure 8 – Spatial groundwater depth maps
# =============================================================================
 
# --- Colormap for absolute head depth ---
HEADS_COLORS = [
    "#cc4c02", "#ec7014", "#fe9929", "#fec44f", "#fee391",
    "#fff7bc", "#dadaeb", "#bcbddc", "#9e9ac8", "#756bb1", "#542788",
]
HEADS_BOUNDS = [-100, -50, -25, -15, -10, -7.5, -5, -2.5, -1, 0]
heads_norm   = BoundaryNorm(HEADS_BOUNDS, ncolors=len(HEADS_COLORS), extend="both")
heads_cmap   = ListedColormap(HEADS_COLORS)
 
# --- Colormap for head depth difference ---
HEADS_DIFF_COLORS = [
    "#99000d", "#cc2a2f", "#e6615f", "#fcae91", "#fee5d9",
    "#d9d9d9", "#c6dbef", "#6baed6", "#3182bd", "#08519c", "#041f78",
]
HEADS_DIFF_BOUNDS = [-5, -2.5, -1, -0.5, -0.1, 0.1, 0.5, 1, 2.5, 5]
heads_diff_norm   = BoundaryNorm(HEADS_DIFF_BOUNDS, ncolors=len(HEADS_DIFF_COLORS), extend="both")
heads_diff_cmap   = ListedColormap(HEADS_DIFF_COLORS)
 
# Part 1: absolute depth per period
fig = make_spatial_figure(
    heads_da_valid, heads_cmap, heads_norm, HEADS_BOUNDS,
    cbar_label='Metres below ground level',
    title_fn=lambda p: str(p.values),
)
plt.show()
 
# Part 2: change between consecutive periods
fig = make_difference_figure(
    heads_da_valid, heads_diff_cmap, heads_diff_norm, HEADS_DIFF_BOUNDS,
    cbar_label='Difference in metres below ground level',
)
plt.show()
 
 
#%% =============================================================================
# Figure 10 – Spatial GW–SW interaction maps
# =============================================================================
 
# --- Colormap for absolute GW–SW exchange ---
GWSW_COLORS = [
    "#cc4c02", "#ec7014", "#fe9929", "#fec44f", "#fee391",
    "#d9d9d9",
    "#dadaeb", "#bcbddc", "#9e9ac8", "#756bb1", "#542788",
]
GWSW_BOUNDS = [-150000, -100000, -50000, -10000, -1, 1, 10000, 50000, 100000, 150000]
gwsw_norm   = BoundaryNorm(GWSW_BOUNDS, ncolors=len(GWSW_COLORS), extend="both")
gwsw_cmap   = ListedColormap(GWSW_COLORS)
 
# --- Colormap for GW–SW exchange difference ---
GWSW_DIFF_COLORS = [
    "#99000d", "#cc2a2f", "#e6615f", "#fcae91", "#fee5d9",
    "#d9d9d9", "#c6dbef", "#6baed6", "#3182bd", "#08519c", "#041f78",
]
GWSW_DIFF_BOUNDS = [-100000, -10000, -1000, -100, -10, 10, 100, 1000, 10000, 100000]
gwsw_diff_norm   = BoundaryNorm(GWSW_DIFF_BOUNDS, ncolors=len(GWSW_DIFF_COLORS), extend="both")
gwsw_diff_cmap   = ListedColormap(GWSW_DIFF_COLORS)
 
# Part 1: absolute exchange per period
fig = make_spatial_figure(
    GW_SW_da_valid, gwsw_cmap, gwsw_norm, GWSW_BOUNDS,
    cbar_label='m³/year per cell',
    title_fn=lambda p: str(p.values),
)
plt.show()
 
# Part 2: change between consecutive periods
fig = make_difference_figure(
    GW_SW_da_valid, gwsw_diff_cmap, gwsw_diff_norm, GWSW_DIFF_BOUNDS,
    cbar_label='Difference in m³/year per cell',
)
plt.show()
