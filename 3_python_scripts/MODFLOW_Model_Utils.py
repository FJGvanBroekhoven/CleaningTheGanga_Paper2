# -*- coding: utf-8 -*-
"""
Author : Frank van Broekhoven
=======================
Utility functions for the MODFLOW model of the Ganges–Yamuna interfluve and Monte Carlo analysis.
 
Contents
--------
1.  MODFLOW package input construction  (CHD, RIV, DRN, RCHA)
2.  Model performance metrics
3.  Post-processing  (heads, budget, GW-SW exchange)
 
"""
 
import numpy as np
import pandas as pd
 

# =============================================================================
# 1.  MODFLOW PACKAGE INPUT CONSTRUCTION
# =============================================================================
 
def dataarray_to_stress_period_dict(da):
    """
    Convert a 3-D xarray DataArray to a FloPy stress-period dictionary.
 
    Parameters
    ----------
    da : xr.DataArray
        Array with dimensions (time, y, x).
 
    Returns
    -------
    dict
        Keys are zero-based stress-period integers; values are 2-D numpy
        arrays (y, x) for the corresponding time slice.
    """
    return {i: da.sel(time=t).values for i, t in enumerate(da.time)}
 
 
def create_chd_input(level_data):
    """
    Build a CHD (Constant Head) stress-period record list from a 2-D level array.
 
    Parameters
    ----------
    level_data : np.ndarray, shape (nrow, ncol)
        Head values (m). NaN cells are excluded.
 
    Returns
    -------
    list of tuple
        Each entry: ((layer, row, col), head).
    """
    rows, cols = np.where(~np.isnan(level_data))
    return [((0, r, c), level_data[r, c]) for r, c in zip(rows, cols)]
 
 
def create_riv_input(level_data, area_data, conductance_factor=1, bottom_offset=5):
    """
    Build a RIV (River) stress-period record list.
 
    Conductance = area_data * conductance_factor  [m2/day per m head difference].
    Riverbed bottom = stage - bottom_offset.
 
    Parameters
    ----------
    level_data : np.ndarray, shape (nrow, ncol)
        River stage (m). NaN cells are excluded.
    area_data : np.ndarray, shape (nrow, ncol)
        River area per cell (m2).
    conductance_factor : float, optional
        Scaling factor for conductance (default 1).
    bottom_offset : float, optional
        Depth of riverbed below river stage (m, default 5).
 
    Returns
    -------
    list of tuple
        Each entry: ((layer, row, col), stage, conductance, rbot).
    """
    rows, cols  = np.where(~np.isnan(level_data))
    conductance = area_data * conductance_factor
    rbot        = level_data - bottom_offset
    return [
        ((0, r, c), level_data[r, c], conductance[r, c], rbot[r, c])
        for r, c in zip(rows, cols)
    ]
 
 
def create_drn_input(level_data, area_data, conductance_factor=1):
    """
    Build a DRN (Drain) stress-period record list.
 
    The drain only removes water when the head exceeds the drain elevation.
 
    Parameters
    ----------
    level_data : np.ndarray, shape (nrow, ncol)
        Drain elevation (m). NaN cells are excluded.
    area_data : np.ndarray, shape (nrow, ncol)
        Drain area per cell (m2).
    conductance_factor : float, optional
        Scaling factor for conductance (default 1).
 
    Returns
    -------
    list of tuple
        Each entry: ((layer, row, col), elevation, conductance).
    """
    rows, cols  = np.where(~np.isnan(level_data))
    conductance = area_data * conductance_factor
    return [
        ((0, r, c), level_data[r, c], conductance[r, c])
        for r, c in zip(rows, cols)
    ]
 

# =============================================================================
# 2.  MODEL PERFORMANCE METRICS
# =============================================================================

def mean_error(sim, obs):
    return np.mean(sim - obs)
 
def mean_absolute_error(sim, obs):
    return np.mean(np.abs(sim - obs))
 
def mean_squared_error(sim, obs):
    return np.mean((sim - obs) ** 2)
 
def root_mean_squared_error(sim, obs):
    return np.sqrt(mean_squared_error(sim, obs))
 
def nash_sutcliffe_efficiency(sim, obs):
    return 1 - (np.sum((sim - obs) ** 2) / np.sum((obs - np.mean(obs)) ** 2))
 
def log_nash_sutcliffe_efficiency(sim, obs):
    log_obs = np.log(obs + 1e-6)
    log_sim = np.log(sim + 1e-6)
    return 1 - (np.sum((log_sim - log_obs) ** 2) / np.sum((log_obs - np.mean(log_obs)) ** 2))
 
def kling_gupta_efficiency(sim, obs):
    r     = np.corrcoef(obs, sim)[0, 1] # correlation coefficient
    alpha = np.std(sim) / np.std(obs) # variability ratio
    beta  = np.mean(sim) / np.mean(obs)  # bias ratio
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
 
def calculate_metrics(sim, obs):
    return {
        'Observed Mean':  np.mean(obs),
        'Simulated Mean': np.mean(sim),
        'ME':             mean_error(sim, obs),
        'MAE':            mean_absolute_error(sim, obs),
        'MSE':            mean_squared_error(sim, obs),
        'RMSE':           root_mean_squared_error(sim, obs),
        'NSE':            nash_sutcliffe_efficiency(sim, obs),
        'LNSE':           log_nash_sutcliffe_efficiency(sim, obs),
        'KGE':            kling_gupta_efficiency(sim, obs),
    }
 
def model_performance(simulated_df, observed_df):
    """
    Compute performance metrics for each observation well and print a summary.
 
    For each well, only time steps present in both the simulated and observed
    DataFrames are used (inner join on the time index).
 
    Parameters
    ----------
    simulated_df : pd.DataFrame
        Simulated head time series. Columns = well IDs (uppercase),
        index = datetime.
    observed_df : pd.DataFrame
        Observed head time series. Same column/index convention.
 
    Returns
    -------
    metrics_df : pd.DataFrame
        One row per well, one column per metric.
    """
    metrics_df = pd.DataFrame(index=observed_df.columns)
 
    for well in observed_df.columns:
        observed  = observed_df[well].dropna()
        simulated = simulated_df[well].dropna()
        common_idx = observed.index.intersection(simulated.index)
        metrics = calculate_metrics(simulated.loc[common_idx], observed.loc[common_idx])
        for metric_name, metric_value in metrics.items():
            metrics_df.loc[well, metric_name] = metric_value
 
    # Print summary across all wells
    print("Model Performance Metrics:")
    print("=" * 50)
    for metric in metrics_df.columns:
        mean = metrics_df[metric].mean()
        if metric in ('Observed Mean', 'Simulated Mean'):
            print(f"  {metric}: {mean:.3f}")
        else:
            print(f"  {metric}: {mean:.3f} ± {metrics_df[metric].std():.3f}")
 
    return metrics_df


# =============================================================================
# 3.  POST-PROCESSING
# =============================================================================
 
def get_model_obs(gwf, time_coords):
    """
    Extract simulated head time series at observation well locations.
 
    Parameters
    ----------
    gwf : flopy.mf6.ModflowGwf
    time_coords : array-like
        Datetime coordinates to assign as the DataFrame index.
 
    Returns
    -------
    pd.DataFrame
        Columns = well IDs (uppercase), index = time_coords.
    """
    csv    = gwf.head_obs.output.obs(f="head_obs.csv").get_data()
    obs_df = pd.DataFrame(csv).drop(columns=['totim'])
    obs_df.index = time_coords
    return obs_df
 
 
def get_heads_mbgl(gwf):
    """
    Return simulated heads expressed as depth below ground level.
 
    Positive values = above ground level; negative values = below ground level.
    MODFLOW's no-data sentinel (1e30) is replaced with NaN.
 
    Parameters
    ----------
    gwf : flopy.mf6.ModflowGwf
 
    Returns
    -------
    np.ndarray, shape (ntime, nlay, nrow, ncol)
        head - DEM top.
    """
    heads = gwf.output.head().get_alldata()
    heads[heads == 1e30] = np.nan
    top = gwf.dis.top.get_data()   # shape (nrow, ncol)
    return heads - top[np.newaxis, np.newaxis, :, :]
 
 
def heads_median_mbgl(heads_mbgl):
    """
    Compute the spatial median depth below ground level at each time step.
 
    Parameters
    ----------
    heads_mbgl : np.ndarray, shape (ntime, nlay, nrow, ncol)
 
    Returns
    -------
    np.ndarray, shape (ntime,)
    """
    return np.nanmedian(heads_mbgl, axis=(1, 2, 3))
 
 
def heads_realism_checks(heads_mbgl):
    """
    Evaluate spatial realism criteria on the simulated head depth distribution.
 
    Criteria (thresholds checked against model-acceptance rules):
    - Waterlogging  : fraction of cells with head above ground level (> 0).
    - Shallow depth : fraction of cells shallower than 8 m bGL.
    - Very deep     : fraction of cells deeper than 50 m bGL.
 
    Parameters
    ----------
    heads_mbgl : np.ndarray, shape (ntime, nlay, nrow, ncol)
 
    Returns
    -------
    dict
        Keys: 'pct_above_gl', 'pct_below_8mbgl', 'pct_below_50mbgl'  (%).
    """
    n_total = np.sum(~np.isnan(heads_mbgl))
    return {
        'pct_above_gl':     np.sum(heads_mbgl > 0)   / n_total * 100,
        'pct_below_8mbgl':  np.sum(heads_mbgl < -8)  / n_total * 100,
        'pct_below_50mbgl': np.sum(heads_mbgl < -50) / n_total * 100,
    }
 
 
def get_budget(gwf, days_per_stress_period, time_coords):
    """
    Extract water-balance fluxes per stress period from the MODFLOW budget CSV.
 
    Rates (m3/day) are multiplied by days_per_stress_period to give volumes
    (m3/stress period). Sign convention: positive = flux into aquifer.
 
    Parameters
    ----------
    gwf : flopy.mf6.ModflowGwf
    days_per_stress_period : float
    time_coords : array-like
 
    Returns
    -------
    pd.DataFrame
        Columns: 'Hindon river', 'other rivers', 'Ganga', 'Yamuna','southern boundary', 'net recharge', 'storage change'.
        Index = time_coords.
    """
    budget_df = pd.DataFrame(gwf.output.budgetcsv().data) * days_per_stress_period
    budget_df.index = time_coords
 
    results = pd.DataFrame(index=time_coords)
    results['Hindon river']      = budget_df['RIV(RIV_HINDON)_IN'] - budget_df['RIV(RIV_HINDON)_OUT'] - budget_df['DRN(DRN_HINDON)_OUT']
    results['other rivers']      = budget_df['RIV(RIV_OTHER)_IN'] - budget_df['RIV(RIV_OTHER)_OUT'] - budget_df['DRN(DRN_OTHER)_OUT']
    results['Ganga']             = budget_df['CHD(CHD_GANGA)_IN'] - budget_df['CHD(CHD_GANGA)_OUT']
    results['Yamuna']            = budget_df['CHD(CHD_YAMUNA)_IN'] - budget_df['CHD(CHD_YAMUNA)_OUT']
    results['southern boundary'] = budget_df['CHD(CHD_SOUTHERN)_IN'] - budget_df['CHD(CHD_SOUTHERN)_OUT']
    results['net recharge']      = budget_df['RCHA(NET_RECHARGE)_IN'] - budget_df['RCHA(NET_RECHARGE)_OUT']
    results['storage change']    = budget_df['STO-SY(STORAGE)_IN']- budget_df['STO-SY(STORAGE)_OUT']
    return results
 
 
def means_per_period(arr3d):
    """
    Compute spatially averaged 2-D fields for five historical sub-periods.
 
    Sub-periods correspond to major phases in canal development and land-use
    change in the Hindon Basin.
 
    Parameters
    ----------
    arr3d : np.ndarray, shape (ntime, nrow, ncol)
 
    Returns
    -------
    np.ndarray, shape (5, nrow, ncol)
        Period-averaged arrays stacked in chronological order:
        1800-1830, 1830-1900, 1900-1970, 1970-2000, 2000-2016.
    """
    periods = {
        "1800-1830": (0,   29),
        "1830-1900": (30,  99),
        "1900-1970": (100, 169),
        "1970-2000": (170, 199),
        "2000-2016": (200, 216),
    }
    return np.stack([
        np.nanmean(arr3d[s:e + 1], axis=0)
        for s, e in periods.values()
    ])
 
 
def build_flow_array(records_list, nrow, ncol):
    """
    Convert a list of MODFLOW budget recarrays to a 3-D flow array.
 
    Sign convention: positive = exfiltration (aquifer to surface water);
                     negative = infiltration (surface water to aquifer).
 
    Parameters
    ----------
    records_list : list of np.recarray
        One recarray per stress period, as returned by budget().get_data().
    nrow, ncol : int
        Grid dimensions.
 
    Returns
    -------
    np.ndarray, shape (ntime, nrow, ncol)
    """
    ncell = nrow * ncol
    nt    = len(records_list)
    arr3d = np.full((nt, ncell), np.nan)
    for t, records in enumerate(records_list):
        nodes           = records["node"] % ncell
        arr3d[t, nodes] = records["q"] * -1
    return arr3d.reshape((nt, nrow, ncol))
 
 
def get_sw_gw_interaction(gwf):
    """
    Aggregate groundwater-surface water exchange fluxes across all packages.
 
    Sums RIV, DRN, and CHD budget fluxes cell-by-cell into a single array
    representing total GW-SW exchange per cell per stress period.
 
    Parameters
    ----------
    gwf : flopy.mf6.ModflowGwf
 
    Returns
    -------
    np.ndarray, shape (ntime, nrow, ncol)
        Positive = net exfiltration; negative = net infiltration.
        NaN where no surface-water package is active.
    """
    nrow, ncol = gwf.modelgrid.nrow, gwf.modelgrid.ncol
    paknames   = [
        "chd_Ganga", "chd_Yamuna", "chd_southern",
        "riv_Hindon", "riv_other",
        "drn_Hindon", "drn_other",
    ]
 
    combined = None
    nan_mask = None
    for pname in paknames:
        records_list = gwf.output.budget().get_data(paknam2=pname)
        arr3d        = build_flow_array(records_list, nrow, ncol)
        if combined is None:
            combined = np.zeros_like(arr3d)
            nan_mask = np.isnan(arr3d)
        else:
            nan_mask &= np.isnan(arr3d)
        combined += np.nan_to_num(arr3d)
 
    combined[nan_mask] = np.nan
    return combined


