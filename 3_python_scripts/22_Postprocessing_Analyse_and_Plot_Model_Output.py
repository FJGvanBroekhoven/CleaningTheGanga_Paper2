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
    part 3: Change between periods 1800-1830 and 2000-2016.
 
Figure 9  – GW–SW exchange time series
    Ensemble min–mean–max of groundwater–surface water exfiltration fluxes
    [m³/s] for the Hindon River, other rivers, the Ganga, and the Yamuna.
 
Figure 10 – Spatial GW–SW interaction maps
    Part 1: Mean GW–SW exchange [m³/year per cell] per historical period.
    Part 2: Change in GW–SW exchange between consecutive periods [m³/year].
    part 3: Change between periods 1800-1830 and 2000-2016.
    
Figures appendix B – comparison obs and sim heads and trends.
    B.1 Scatter plot, observed vs simulated heads
    B.3 Box plot, observed vs simulated trends
    B.3 Plot wells obs vs sim with the most observations
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
from scipy.stats import theilslopes
from matplotlib.colors import BoundaryNorm, ListedColormap

# Set pdf.fonttype to make sure that the figure labels are 'text' in the pdf exports and not 'outlines'
plt.rcParams['pdf.fonttype'] = 42 
 
 
#%% =============================================================================
# Paths and settings
# =============================================================================
 
main_dir = r'C:\GitHub\CleaningTheGanga_Paper2_revisions'
 
outdir    = os.path.join(main_dir, '2_model_output', 'model_output_data', '1500_Runs_UGYI_obs')
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
#VALIDITY_CRITERIA = {
#    'pct_above_gl':     ('<=', 2.5),
#    'pct_below_50mbgl': ('<=', 5.0),
#    'MAE':              ('<=', 7.5),
#    'mean_error':       ('between', -5, 5),
#    'Q_avg_1997_1999':  ('<',  100),
#    'Q_avg_1800_1825':  ('>=', 0),
#}

# Revisions, comparison wells obs vs sim
# Observation well metadata (GeoJSON) and head time series (CSV)
OBS_META_PATH =         os.path.join(main_dir, '1_model_input', '3_model_input_data', 'metadata_selected_observation_wells_yearly.geojson')
OBS_TS_PATH =           os.path.join(main_dir, '1_model_input', '3_model_input_data', 'timeseries_data_selected_observation_wells_yearly.csv') 
obs_wells_gdf = gpd.read_file(OBS_META_PATH)
obs_wells_gdf.set_index('well_id', inplace=True)
timeseries_df = pd.read_csv(OBS_TS_PATH, parse_dates=['date'], index_col='date')
timeseries_df.columns = timeseries_df.columns.str.upper()

 
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


# Revisions, added the simulated heads at obs locations    
def merge_obs_to_xarray(outdir):
    """
    Load all per-run obs CSVs and return them as an xarray Dataset.

    Each observation well becomes a Dataset variable with dimensions
    (time, model_id).

    Parameters
    ----------
    outdir : str
        Root output directory containing an 'obs' subdirectory.

    Returns
    -------
    xr.Dataset
        Dimensions: time, model_id.
    """
    obs_dir = os.path.join(outdir, 'obs')
    dfs = {}
    for fname in os.listdir(obs_dir):
        if fname.endswith('.csv'):
            model_id = '_'.join(fname.split('_')[:2])
            dfs[model_id] = pd.read_csv(
                os.path.join(obs_dir, fname), index_col=0, parse_dates=True
            )

    time_index = next(iter(dfs.values())).index
    variables  = next(iter(dfs.values())).columns
    model_ids  = list(dfs.keys())

    ds = xr.Dataset(coords={'time': time_index, 'model_id': model_ids})
    for var in variables:
        data = np.stack([dfs[mid][var].values for mid in model_ids], axis=1)
        ds[var] = (('time', 'model_id'), data)

    return ds.transpose('time', 'model_id')

 
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
 
#%% === Area study area
#mask_da.plot()
#count = (mask_da != -1).sum().item() # all non -1 values    # 16759 km2
#ount = (mask_da == 1).sum().item() # all 1 values    # 16135 km2
## so roughly 16500 km2
## widest point roughtly 100 km
## longest roughly 220 km

#%% =============================================================================
# Assemble combined dataset
# =============================================================================
 
# --- Overviews ---
overviews = merge_overviews(outdir)
overviews['mean_error'] = overviews['Simulated Mean'] - overviews['Observed Mean']

#NEW CODE EXPORT PARAM DF LIST FOR REVISIONS
#overviews_clean = overviews[list(MC_PARAMS)].copy()
#overviews_clean_output_path = os.path.join(main_dir, '1_model_input', '1_original_data', 'overview_original_params_for_revisions.csv')
#overviews_clean.to_csv(overviews_clean_output_path)

# --- Obs ---
obs_xarray = merge_obs_to_xarray(outdir)

### ADDED REVISIONS DOWN ###
# =============================================================================
# Calculate Theil-Sen slopes
# =============================================================================

# Calculate observed Theil-Sen slopes for the obersations
obs_slopes = {}
for well in timeseries_df.columns:
    obs = timeseries_df[well]
    mask = obs.notna()
    if mask.sum() < 3:
        obs_slopes[well] = np.nan
        continue
    x = obs.index.year.values[mask]
    y = obs.values[mask]
    obs_slopes[well] = theilslopes(y, x).slope
obs_slopes = pd.Series(obs_slopes, name="obs_slope")

# Calculate simulated slopes using the observation years
sim_slopes = {}
for well in timeseries_df.columns:
    obs = timeseries_df[well]
    mask = obs.notna()
    if mask.sum() < 3:
        continue
    obs_times = obs.index[mask]
    x = obs_times.year.values
    sim = obs_xarray[well].sel(time=obs_times)
    slopes = []
    for model in sim.model_id.values:
        y = sim.sel(model_id=model).values
        slopes.append(theilslopes(y, x).slope)
    sim_slopes[well] = pd.Series(slopes, index=sim.model_id.values)
sim_slopes = pd.DataFrame(sim_slopes)

# add mean slopes to overviews df
overviews['obs_mean_slope'] = obs_slopes.mean()
overviews['sim_mean_slope'] = sim_slopes.mean(axis=1) # Row-wise mean (across wells)
overviews['slope_error'] = abs(overviews['obs_mean_slope'] - overviews['sim_mean_slope'])
### ADDED REVISIONS UP ###

 
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



# Revisions, count for each criteria how many realisations are rejected
cols = ['pct_above_gl', 'pct_below_50mbgl', 'MAE', 'mean_error', 'Q_avg_1997_1999', 'Q_avg_1800_1825']
overviews_clean = overviews.dropna(subset=cols)

total = len(overviews_clean)

criteria = [
    ('pct_above_gl > 2.5',     overviews_clean['pct_above_gl'] > 2.5),
    ('pct_below_50mbgl > 5.0', overviews_clean['pct_below_50mbgl'] > 5.0),
    ('MAE > 7.5',              overviews_clean['MAE'] > 7.5),
    ('mean_error > 5.0',       overviews_clean['mean_error'] > 5.0),
    ('mean_error < -5.0',      overviews_clean['mean_error'] < -5.0),
    ('Q_avg_1997_1999 >= 65',  overviews_clean['Q_avg_1997_1999'] >= 65),
    ('Q_avg_1800_1825 < 0',    overviews_clean['Q_avg_1800_1825'] < 0)
]

for name, mask in criteria:
    count = mask.sum()
    print(f"{name}: {count} / {total} ({100*count/total:.1f}%)")
 
 
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
    #fig = plt.figure(figsize=(5 * n_panels, 5))
    fig = plt.figure(figsize=(5 * n_panels + 1, 5))  # revised
    #gs  = fig.add_gridspec(2, n_panels, height_ratios=[1, 0.05],
    #                       wspace=0.05, hspace=0.3)
    gs  = fig.add_gridspec(1, n_panels + 1, 
                           width_ratios=[1] * n_panels + [0.05],
                           wspace=0.05)  # revised 
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
 
    #cbar_ax = fig.add_subplot(gs[1, :])
    cbar_ax = fig.add_subplot(gs[0, -1])  # revised
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        #cax=cbar_ax, orientation='horizontal'
        cax=cbar_ax, orientation='vertical'  # revised
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
    #fig = plt.figure(figsize=(5 * n_diff, 5))
    fig = plt.figure(figsize=(5 * n_diff + 1, 5))  # revised
    #gs  = fig.add_gridspec(2, n_diff, height_ratios=[1, 0.05],
    #                       wspace=0.05, hspace=0.3)
    gs  = fig.add_gridspec(1, n_diff + 1,
                           width_ratios=[1] * n_diff + [0.05],
                           wspace=0.05)  # revised 
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
 
    #cbar_ax = fig.add_subplot(gs[1, :])
    cbar_ax = fig.add_subplot(gs[0, -1])  # revised
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        #cax=cbar_ax, orientation='horizontal'
        cax=cbar_ax, orientation='vertical'  # revised
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
    
    return fig, axes
 
 
fig, axex = plot_parameter_histograms(MC_PARAMS, overviews, overviews_valid, PARAM_LABELS)

# save as PDF file
# outfile
fig_3_path = os.path.join(main_dir, '2_model_output', 'figures', 'fig_3.pdf')
fig.savefig(fig_3_path, format="pdf", bbox_inches="tight") 
 
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

# save as PDF file
# outfile
fig_5_path = os.path.join(main_dir, '2_model_output', 'figures', 'fig_5.pdf')
fig.savefig(fig_5_path, format="pdf", bbox_inches="tight") 
 
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

# save as PDF file
# outfile
fig_7_path = os.path.join(main_dir, '2_model_output', 'figures', 'fig_7.pdf')
fig.savefig(fig_7_path, format="pdf", bbox_inches="tight")  
 
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

# save as PDF file
# outfile
fig_9_path = os.path.join(main_dir, '2_model_output', 'figures', 'fig_9.pdf')
fig.savefig(fig_9_path, format="pdf", bbox_inches="tight") 
 
 
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

# save as 600 dpi raster png
# outfile
fig_8B_path = os.path.join(main_dir, '2_model_output', 'figures', 'fig_8B.png')
fig.savefig(fig_8B_path, dpi=600,bbox_inches="tight")
 
# Part 2: change between consecutive periods
fig = make_difference_figure(
    heads_da_valid, heads_diff_cmap, heads_diff_norm, HEADS_DIFF_BOUNDS,
    cbar_label='Difference in metres below ground level',
)
plt.show()

# save as 600 dpi raster png
# outfile
fig_8C_path = os.path.join(main_dir, '2_model_output', 'figures', 'fig_8C.png')
fig.savefig(fig_8C_path, dpi=600,bbox_inches="tight")

# Revisions
# Part 3: change between first and last period
fig, ax = plt.subplots()
last = heads_da_valid.period[-1]
first = heads_da_valid.period[0]
diff = (heads_da_valid.sel(period=last).mean("model_id")
        - heads_da_valid.sel(period=first).mean("model_id"))
diff.plot(ax=ax, x='x', y='y', cmap=heads_diff_cmap, norm=heads_diff_norm, add_colorbar=False)
labels.plot(ax=ax, color=None, edgecolor=None, markersize=0, alpha=0) # don't show but use geopandas to make sure that proportions are fixed
ax.set_title(f"{last.values} − {first.values}")
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')
plt.show()

# save as 600 dpi raster png
# outfile
fig_8A_path = os.path.join(main_dir, '2_model_output', 'figures', 'fig_8A.png')
fig.savefig(fig_8A_path, dpi=600,bbox_inches="tight")

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

# save as 600 dpi raster png
# outfile
fig_10B_path = os.path.join(main_dir, '2_model_output', 'figures', 'fig_10B.png')
fig.savefig(fig_10B_path, dpi=600,bbox_inches="tight")
 
# Part 2: change between consecutive periods
fig = make_difference_figure(
    GW_SW_da_valid, gwsw_diff_cmap, gwsw_diff_norm, GWSW_DIFF_BOUNDS,
    cbar_label='Difference in m³/year per cell',
)
plt.show()

# save as 600 dpi raster png
# outfile
fig_10C_path = os.path.join(main_dir, '2_model_output', 'figures', 'fig_10C.png')
fig.savefig(fig_10C_path, dpi=600,bbox_inches="tight")

# Revisions
# Part 3: change between first and last period
fig, ax = plt.subplots()
last = GW_SW_da_valid.period[-1]
first = GW_SW_da_valid.period[0]
diff = (GW_SW_da_valid.sel(period=last).mean("model_id")
        - GW_SW_da_valid.sel(period=first).mean("model_id"))
diff.plot(ax=ax, x='x', y='y', cmap=gwsw_diff_cmap, norm=gwsw_diff_norm, add_colorbar=False)
labels.plot(ax=ax, color=None, edgecolor=None, markersize=0, alpha=0) # don't show but use geopandas to make sure that proportions are fixed
ax.set_title(f"{last.values} − {first.values}")
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')
plt.show()

# save as 600 dpi raster png
# outfile
fig_10A_path = os.path.join(main_dir, '2_model_output', 'figures', 'fig_10A.png')
fig.savefig(fig_10A_path, dpi=600,bbox_inches="tight")

#%% ### ADDED REVISIONS DOWN ###
# =============================================================================
# Make plots comparisons observed and simulated heads and trends
# =============================================================================

# get overview stats trends (theil-sen slope) per well

# Select valid models
sim_slopes_valid = sim_slopes.loc[overviews_valid.index]

# get df with stats per well
trend_summary = pd.DataFrame({
    "obs_slope": obs_slopes,
    "sim_min": sim_slopes_valid.min(),
    "sim_median": sim_slopes_valid.median(),
    "sim_max": sim_slopes_valid.max(),
    "n_obs": timeseries_df.notna().sum()
    })

# =============================================================================
# Ensemble statistics for simulated heads
# =============================================================================

# Select valid models
obs_xarray_valid = obs_xarray.sel(model_id=overviews_valid.index)

# get stats per well
sim_median = obs_xarray_valid.median("model_id")
sim_min = obs_xarray_valid.min("model_id")
sim_max = obs_xarray_valid.max("model_id")

# Build dataframe with observed and simulated heads
rows = []
for well in timeseries_df.columns:
    obs = timeseries_df[well].dropna()
    for t, value in obs.items():
        rows.append({
            "well": well,
            "time": t,
            "obs": value,
            "sim_min": sim_min[well].sel(time=t).item(),
            "sim_median": sim_median[well].sel(time=t).item(),
            "sim_max": sim_max[well].sel(time=t).item()
        })
compare_df = pd.DataFrame(rows)

# calc R2 score
obs, sim = compare_df['obs'], compare_df['sim_median']
ss_res = np.sum((obs - sim) ** 2)
ss_tot = np.sum((obs - obs.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"R² (obs vs sim_median): {r2:.3f}")

# =============================================================================
# Scatter plot: observed vs simulated heads
# =============================================================================
### FIGURE B.1 ###
fig, ax = plt.subplots(figsize=(7, 7))

# Calculate error bars
yerr = np.vstack([
    (compare_df["sim_median"] - compare_df["sim_min"]).values,
    (compare_df["sim_max"] - compare_df["sim_median"]).values,
])

ax.errorbar(
    compare_df["obs"],
    compare_df["sim_median"],
    yerr=yerr,
    fmt="o",
    alpha=0.5,
    ecolor="gray",
    capsize=0,
    ms=3,
)

lims = [
    min(compare_df[["obs", "sim_min"]].min()),
    max(compare_df[["obs", "sim_max"]].max()),
]
ax.plot(lims, lims, "k--", lw=1, label="1:1")

ax.set_xlabel("Observed head (m)")
ax.set_ylabel("Simulated head (median)\n(error bars = min-max)")
ax.set_title("Observed vs simulated groundwater heads")
ax.legend()
fig.tight_layout()

# save as PDF file
# outfile
fig_B1_path = os.path.join(main_dir, '2_model_output', 'figures', 'fig_B1.pdf')
fig.savefig(fig_B1_path, format="pdf", bbox_inches="tight")  

# =============================================================================
# Box plot: observed vs simulated trends
# =============================================================================
### FIGURE B.2 ###
# box plots trends
# Data and labels
data = [
    trend_summary['obs_slope'],
    trend_summary['sim_min'],
    trend_summary['sim_median'],
    trend_summary['sim_max']
]
labels = ['Observed', 'Simulated min', 'Simulated median', 'Simulated max']

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot(data, patch_artist=True)

# Color: obs in blue, all sims in orange
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor('#1F78B4' if i == 0 else '#cc4c02')

ax.set_xticklabels(labels)
ax.set_ylabel('Theil-Sen slope (m/year)')
ax.grid(axis='y', alpha=0.3)
plt.show()

# save as PDF file
# outfile
fig_B2_path = os.path.join(main_dir, '2_model_output', 'figures', 'fig_B2.pdf')
fig.savefig(fig_B2_path, format="pdf", bbox_inches="tight")  


# =============================================================================
# Plot wells obs vs sim with the most observations
# =============================================================================
### FIGURE B.3 ###
nplots = 12
nrows, ncols = 4, 3

best_wells = (
    timeseries_df.notna()
    .sum()
    .sort_values(ascending=False)
    .head(nplots)
    .index
)

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(15, 16),
    sharex=True,
    constrained_layout=True,
)

axes = axes.ravel()

for ax, well in zip(axes, best_wells):

    # Ensemble envelope
    ax.fill_between(
        sim_median.time.values,
        sim_min[well].values,
        sim_max[well].values,
        color="C0",
        alpha=0.25,
        label="Model range",
    )

    # Ensemble median
    ax.plot(
        sim_median.time.values,
        sim_median[well].values,
        color="C0",
        lw=2,
        label="Model median",
    )

    # Observations
    ax.plot(
        timeseries_df.index,
        timeseries_df[well],
        "ko",
        ms=3.5,
        label="Observations",
    )

    ax.set_title(f"{well} (N={timeseries_df[well].count()})", fontsize=10)
    ax.grid(alpha=0.3)

    # Fix x-limits
    ax.set_xlim(pd.Timestamp("1980-01-01"), sim_median.time.values[-1])

# Remove unused axes
for ax in axes[len(best_wells):]:
    fig.delaxes(ax)

for ax in axes:
    ax.tick_params(axis="x", labelrotation=45)

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    frameon=False,
)

fig.suptitle("Observed and simulated groundwater heads", fontsize=16)

# save as PDF file
# outfile
fig_B3_path = os.path.join(main_dir, '2_model_output', 'figures', 'fig_B3.pdf')
fig.savefig(fig_B3_path, format="pdf", bbox_inches="tight")  
