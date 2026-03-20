# -*- coding: utf-8 -*-
"""
Author : Frank van Broekhoven
=======================================================================
This script sets up, runs, and evaluates an ensemble of transient MODFLOW 6
groundwater models of the Ganges–Yamuna interfluve.

Each realisation randomly samples hydraulic conductivity, aquifer thickness,
specific yield, river/drain conductance, and multiplication factors for each
recharge and abstraction component. Results are evaluated against 94
observation wells and screened with realism criteria before acceptance.
 
Simulation period : 1800–2016 (yearly stress periods)
Grid              : 231 rows × 108 columns, 1 km × 1 km cells, single layer
Model framework   : MODFLOW 6 via FloPy
 
Workflow
--------
1. Load spatial and temporal input data (masks, DEM, boundary conditions,
   recharge sources, observation wells).
2. Define stress-period discretisation.
3. Run N model realisations (default: 1500) using random sampling of
   parameter space.
4. Post-process each run: heads, water-balance budgets, GW–SW exchange,
   and goodness-of-fit metrics against observation wells.
5. Save results to structured output directories.
 
See modflow_utils.py for all helper functions.

Dependencies
------------
numpy, pandas, geopandas, xarray, rioxarray, flopy, obs_utils_v01 (local)
 
"""
 
#%% =============================================================================
# IMPORTS
# =============================================================================
 
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import flopy
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
 
# import own utils functions
sys.path.append(r'C:\GitHub\Temp_CleaningTheGanga_Paper2\3_python_scripts')
import MODFLOW_Model_Utils as utils
 

#%% =============================================================================
# PATHS
# =============================================================================

main_dir = r'C:\GitHub\Temp_CleaningTheGanga_Paper2'
 
# --- Input ---
# Active-cell mask (value >= 0 = boundary cell, value = 1 = active interior cell)
MASK_PATH =             os.path.join(main_dir, '1_model_input', '1_original_data', 'Model_Mask.tif')
# Mean DEM from HydroSHEDS, used as the model top surface
DEM_MEAN_PATH =         os.path.join(main_dir, '1_model_input', '3_model_input_data', 'DIS_TOP.tif')
# Starting heads for the IC package; also defines aquifer bottom (IC head - d)
INITIAL_CONDITIONS_PATH = os.path.join(main_dir, '1_model_input', '1_original_data','Initial_Conditions', 'starting_heads.nc')
# Boundary condition levels and areas for RIV, DRN, and CHD packages
DRN_RIV_CHD_PATH =      os.path.join(main_dir, '1_model_input', '3_model_input_data', 'DRN_RIV_CHD.nc')
# Annual recharge and abstraction components (m3/year per cell)
RECHARGE_SOURCES_PATH = os.path.join(main_dir, '1_model_input', '3_model_input_data', 'recharge_and_abstraction_components_yearly.nc')
# Observation well metadata (GeoJSON) and head time series (CSV)
OBS_META_PATH =         os.path.join(main_dir, '1_model_input', '3_model_input_data', 'metadata_selected_observation_wells_yearly.geojson')
OBS_TS_PATH =           os.path.join(main_dir, '1_model_input', '3_model_input_data', 'timeseries_data_selected_observation_wells_yearly.csv')

# --- Output ---
BASE_WS = os.path.join(main_dir, '2_model_output', 'temp_MODFLOW_workspaces', '1500_MC_Runs_UGYI')   # Temporary MODFLOW workspaces
OUTDIR  = os.path.join(main_dir, '2_model_output', 'model_output_data', '1500_MC_Runs_UGYI')   # Saved post-processed results


#%% =============================================================================
# SETTINGS
# =============================================================================
 
START_DATE = datetime(1800, 1, 1)    # Pre-canal baseline start
END_DATE   = datetime(2016, 12, 31)  # Last day of final simulated year
TIMESTEP   = 'year'                  # 'year' or 'month'
N_RUNS     = 1500                    # Number of Monte Carlo realisations
 

# Monte Carlo parameter space.
# 'uniform'     : sampled uniformly between min and max.
# 'log-uniform' : sampled uniformly in log space, giving equal probability to factors above and below the central estimate.
MC_PARAMS = {
    'k':                            {'dist': 'uniform',     'min': 5,     'max': 100},   # Hydraulic conductivity (m/day)
    'd':                            {'dist': 'uniform',     'min': 50,    'max': 250},   # Aquifer thickness (m)
    'RIV_cond':                     {'dist': 'log-uniform', 'min': 0.001, 'max': 10},    # River conductance factor (-)
    'DRN_cond':                     {'dist': 'log-uniform', 'min': 0.001, 'max': 10},    # Drain conductance factor (-)
    'sy':                           {'dist': 'uniform',     'min': 0.05,  'max': 0.35},  # Specific yield (-)
    'rainfall_rch_factor':          {'dist': 'log-uniform', 'min': 0.5,   'max': 2},     
    'canal_rch_factor':             {'dist': 'log-uniform', 'min': 0.25,  'max': 4},
    'irrigation_return_rch_factor': {'dist': 'log-uniform', 'min': 0.25,  'max': 4},
    'municipal_return_rch_factor':  {'dist': 'log-uniform', 'min': 0.25,  'max': 4},
    'irrigation_demand_abs_factor': {'dist': 'log-uniform', 'min': 0.25,  'max': 4},
    'municipal_demand_abs_factor':  {'dist': 'log-uniform', 'min': 0.25,  'max': 4},
    'industrial_demand_abs_factor': {'dist': 'log-uniform', 'min': 0.25,  'max': 4},
}


#%% =============================================================================
# READ INPUT DATA
# =============================================================================
 
# --- Spatial input arrays (shared across all runs) ---
mask_da            = xr.open_dataarray(MASK_PATH).squeeze(drop='band')
dem_mean           = rioxarray.open_rasterio(DEM_MEAN_PATH)
initial_conditions = xr.open_dataarray(INITIAL_CONDITIONS_PATH)
 
drn_riv_chd_ds = xr.open_dataset(DRN_RIV_CHD_PATH).squeeze(drop='band')
CHD_ds         = xr.where(mask_da > -1, drn_riv_chd_ds, np.nan)   # boundary cells
DRN_RIV_ds     = xr.where(mask_da == 1, drn_riv_chd_ds, np.nan)   # active interior cells
 
recharge_sources = xr.open_dataset(RECHARGE_SOURCES_PATH).sel(time=slice(START_DATE, END_DATE))
time_coords = recharge_sources.coords['time'].values
 
input_arrays = {
    'mask_da':            mask_da,
    'dem_mean':           dem_mean,
    'initial_conditions': initial_conditions,
    'CHD_ds':             CHD_ds,
    'DRN_RIV_ds':         DRN_RIV_ds,
    'recharge_sources':   recharge_sources,
}
 
# --- Observation wells ---
obs_wells_gdf = gpd.read_file(OBS_META_PATH)
obs_wells_gdf.set_index('well_id', inplace=True)
for well_id, row in obs_wells_gdf.iterrows():
    x, y = row.geometry.x, row.geometry.y
    obs_wells_gdf.loc[well_id, 'col'] = int(np.argmin(np.abs(mask_da.x - x).values))
    obs_wells_gdf.loc[well_id, 'row'] = int(np.argmin(np.abs(mask_da.y - y).values))
 
obs_input = {
    "head_obs.csv": [
        (well_id, 'HEAD', (0, int(r['row']), int(r['col'])))
        for well_id, r in obs_wells_gdf.iterrows()
    ]
}
 
timeseries_df = pd.read_csv(OBS_TS_PATH, parse_dates=['date'], index_col='date')
timeseries_df.columns = timeseries_df.columns.str.upper()

 
#%% =============================================================================
# STRESS-PERIOD DISCRETISATION
# =============================================================================
 
def build_stress_period_data(start_date, end_date, timestep):
    """
    Build the TDIS stress-period list for MODFLOW 6.
 
    One stress period per year (365.25 days). The first period is replaced
    by a 1000-year warm-up (365 250 days, 1 step) to achieve stable initial
    conditions before the transient simulation begins in 1800.
 
    Parameters
    ----------
    start_date, end_date : datetime
    timestep : str
        'year' or 'month'.
 
    Returns
    -------
    perioddata : list of tuple
        (period_length_days, n_steps, step_multiplier) per stress period.
    days_per_stress_period : float
    """
    if timestep == 'year':
        days_per_stress_period = 365.25
    elif timestep == 'month':
        days_per_stress_period = 365.25 / 12
    else:
        raise ValueError(f"Unsupported timestep '{timestep}'. Use 'year' or 'month'.")
 
    n_periods  = round((end_date - start_date).days / days_per_stress_period)
    perioddata = [(days_per_stress_period, 1, 1.0)] * n_periods
    perioddata[0] = (365_250, 1, 1.0)  # warm-up period
 
    print(f"Simulation  : {start_date.date()} to {end_date.date()}")
    print(f"Timestep    : {timestep}  ({days_per_stress_period:.2f} days)")
    print(f"Stress periods: {n_periods}")
    return perioddata, days_per_stress_period
 

# --- Stress-period discretisation ---
perioddata, days_per_stress_period = build_stress_period_data(START_DATE, END_DATE, TIMESTEP)

assert len(time_coords) == len(perioddata), (
    f"Recharge time steps ({len(time_coords)}) do not match "
    f"stress periods ({len(perioddata)}). "
    "Check that END_DATE is the last day of the final simulation year.")

 
#%% =============================================================================
# MODEL CONSTRUCTION
# =============================================================================
 
def create_model(
    model_id,
    mask_da,
    dem_mean,
    initial_conditions,
    CHD_ds,
    DRN_RIV_ds,
    recharge_sources,
    obs_input,
    perioddata,
    days_per_stress_period,
    k=30,
    d=150,
    RIV_cond=1,
    DRN_cond=1,
    sy=0.2,
    rainfall_rch_factor=1,
    canal_rch_factor=1,
    irrigation_return_rch_factor=1,
    municipal_return_rch_factor=1,
    irrigation_demand_abs_factor=1,
    municipal_demand_abs_factor=1,
    industrial_demand_abs_factor=1,
    workdir=None,
):
    """
    Construct and return a transient MODFLOW 6 groundwater flow simulation.
 
    The model represents the Hindon River Basin as a single-layer unconfined
    aquifer (1 km x 1 km cells). Recharge and abstraction components are each
    scaled by their Monte Carlo multiplication factor, then summed into a net
    recharge array (m/day) applied via the RCHA package. This preserves the
    spatial and temporal structure of individual components while allowing
    total magnitudes to vary across realisations.
 
    Boundary conditions
    -------------------
    CHD  : Ganga, Yamuna, and southern model boundaries (constant head).
    RIV  : Hindon River and other interior rivers (bidirectional GW-SW).
    DRN  : Hindon River drains and other drains (one-way: head > drain elev.).
    RCHA : Net recharge = sum of (rainfall + canal leakage + return flows
                                  - irrigation - municipal - industrial abs.).
 
    Parameters
    ----------
    model_id : str
        Unique model identifier (max 16 characters).
    mask_da : xr.DataArray
        Active-cell mask.
    dem_mean : xr.DataArray
        Mean DEM used as model top surface.
    initial_conditions : xr.DataArray
        Starting heads; aquifer bottom is set to these values minus d.
    CHD_ds : xr.Dataset
        Constant-head boundary levels (Ganga, Yamuna, southern).
    DRN_RIV_ds : xr.Dataset
        River and drain levels and areas for interior active cells.
    recharge_sources : xr.Dataset
        Annual recharge and abstraction components (m3/year per cell).
    obs_input : dict
        FloPy observation package input (CSV filename -> list of well records).
    perioddata : list of tuple
        TDIS stress-period data from build_stress_period_data().
    days_per_stress_period : float
        Used to convert m3/year recharge to m/day.
    k : float
        Horizontal hydraulic conductivity (m/day).
    d : float
        Saturated aquifer thickness (m).
    RIV_cond : float
        Conductance scaling factor for all RIV packages.
    DRN_cond : float
        Conductance scaling factor for all DRN packages.
    sy : float
        Specific yield (-).
    *_factor : float
        Multiplication factors for each recharge/abstraction component.
    workdir : str or None
        Parent directory for the model workspace. None = temporary directory.
 
    Returns
    -------
    sim : flopy.mf6.MFSimulation
        Fully configured (not yet written or run) MODFLOW 6 simulation.
    """
    workspace = os.path.join(workdir, model_id) if workdir else Path(TemporaryDirectory().name)
 
    # --- Simulation and solver ---
    sim = flopy.mf6.MFSimulation(
        sim_name=model_id, exe_name="mf6", version="mf6", sim_ws=workspace
    )
    flopy.mf6.ModflowTdis(sim, pname="tdis", time_units="DAYS",
                          nper=len(perioddata), perioddata=perioddata)
    flopy.mf6.ModflowIms(sim, pname="ims", complexity="SIMPLE",
                         linear_acceleration="BICGSTAB")
 
    # --- Groundwater flow model ---
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=model_id, model_nam_file=f"{model_id}.nam",
        save_flows=True,
        newtonoptions="NEWTON UNDER_RELAXATION",  # Improves convergence for dry cells
    )
 
    # DIS - spatial discretisation
    idomain = mask_da.values.copy()
    idomain[idomain > -1] = 1  # Convert boundary type codes to active flag (1)
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=1, nrow=231, ncol=108, delr=1000, delc=1000,
        top=dem_mean.values[0, :, :],
        botm=initial_conditions.values - d,  # aquifer bottom = IC head - thickness
        idomain=idomain,
    )
 
    # IC, NPF, STO, OC
    flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=initial_conditions.values)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=k)   # icelltype=1: unconfined
    flopy.mf6.ModflowGwfsto(gwf, iconvert=1, ss=1e-5, sy=sy, steady_state=False)
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{gwf.name}.hds",
        budget_filerecord=f"{gwf.name}.cbc",
        budgetcsv_filerecord=f"{gwf.name}.bud.csv",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
 
    # --- Boundary conditions ---
 
    # CHD: constant-head cells on Ganga, Yamuna, and southern model boundary
    for bname, level_var in [
        ("chd_Ganga",    'CHD_Ganga_level'),
        ("chd_Yamuna",   'CHD_Yamuna_level'),
        ("chd_southern", 'CHD_southern_level'),
    ]:
        flopy.mf6.ModflowGwfchd(
            gwf, pname=bname, filename=f'{bname}.chd',
            stress_period_data=utils.create_chd_input(CHD_ds[level_var].values),
        )
 
    # RIV: bidirectional groundwater-surface water exchange
    for rname, lvl_var, area_var in [
        ("riv_Hindon", 'RIV_Hindon_level', 'RIV_Hindon_area'),
        ("riv_other",  'RIV_other_level',  'RIV_other_area'),
    ]:
        flopy.mf6.ModflowGwfriv(
            gwf, pname=rname, filename=f'{rname}.riv',
            stress_period_data=utils.create_riv_input(
                DRN_RIV_ds[lvl_var].values, DRN_RIV_ds[area_var].values,
                conductance_factor=RIV_cond,
            ),
        )
 
    # DRN: one-way drainage (removes water when head > drain elevation)
    for dname, lvl_var, area_var in [
        ("drn_Hindon", 'DRN_Hindon_level', 'DRN_Hindon_area'),
        ("drn_other",  'DRN_other_level',  'DRN_other_area'),
    ]:
        flopy.mf6.ModflowGwfdrn(
            gwf, pname=dname, filename=f'{dname}.drn',
            stress_period_data=utils.create_drn_input(
                DRN_RIV_ds[lvl_var].values, DRN_RIV_ds[area_var].values,
                conductance_factor=DRN_cond,
            ),
        )
 
    # RCHA: net recharge (m3/year per cell -> m/day)
    # Each component is multiplied by its uncertainty factor before summation,
    # preserving spatial/temporal patterns while varying total magnitudes.
    rch_factors = {
        'rainfall_rch':          rainfall_rch_factor,
        'canal_rch':             canal_rch_factor,
        'irrigation_return_rch': irrigation_return_rch_factor,
        'municipal_return_rch':  municipal_return_rch_factor,
        'irrigation_demand_abs': irrigation_demand_abs_factor,
        'municipal_demand_abs':  municipal_demand_abs_factor,
        'industrial_demand_abs': industrial_demand_abs_factor,
    }
    sources = recharge_sources.copy()
    for var, factor in rch_factors.items():
        sources[var] *= factor
 
    net_recharge      = sum(sources[v].fillna(0) for v in sources.data_vars)
    net_recharge_mday = net_recharge / 1_000_000 / days_per_stress_period
    flopy.mf6.ModflowGwfrcha(
        gwf, pname="net_recharge", filename='net_recharge.rch',
        recharge=utils.dataarray_to_stress_period_dict(net_recharge_mday),
    )
 
    # OBS: head observations at monitoring wells
    flopy.mf6.ModflowUtlobs(
        gwf, pname="head_obs", filename=f"{gwf.name}.obs",
        print_input=True, continuous=obs_input,
    )
 
    return sim
 
 
# =============================================================================
# SINGLE-RUN EXECUTION
# =============================================================================
 
def run_model(model_id, model_params, input_arrays, obs_input, time_coords,
              timeseries_df, perioddata, days_per_stress_period, base_ws):
    """
    Build, run, and post-process a single MODFLOW 6 realisation.
 
    Parameters
    ----------
    model_id : str
        Unique identifier (e.g. 'model_0042').
    model_params : dict
        Sampled parameter values: k, d, sy, RIV_cond, DRN_cond, and the
        seven recharge/abstraction multiplication factors.
    input_arrays : dict
        Spatial arrays shared across all runs: mask_da, dem_mean,
        initial_conditions, CHD_ds, DRN_RIV_ds, recharge_sources.
    obs_input : dict
        FloPy observation package input dictionary.
    time_coords : array-like
        Datetime coordinates for the simulation stress periods.
    timeseries_df : pd.DataFrame
        Observed head time series for goodness-of-fit evaluation.
    perioddata : list of tuple
    days_per_stress_period : float
    base_ws : str
        Parent directory for temporary MODFLOW workspaces.
 
    Returns
    -------
    dict with keys:
        'overview'         : dict            Parameters, metrics, realism checks.
        'budget'           : pd.DataFrame    Annual water-balance fluxes (or None).
        'median_heads'     : np.ndarray      Spatial-median depth time series (or None).
        'heads_per_period' : np.ndarray      Mean head depth per sub-period (or None).
        'GW_SW_per_period' : np.ndarray      Mean GW-SW exchange per sub-period (or None).
    """
    try:
        sim = create_model(
            model_id,
            **input_arrays,
            obs_input=obs_input,
            perioddata=perioddata,
            days_per_stress_period=days_per_stress_period,
            **model_params,
            workdir=base_ws,
        )
        sim.write_simulation()
        success, _ = sim.run_simulation(silent=True)
        if not success:
            raise RuntimeError("MODFLOW 6 did not terminate normally.")
 
        gwf = sim.get_model()
 
        obs_df      = utils.get_model_obs(gwf, time_coords)
        metrics     = utils.model_performance(simulated_df=obs_df,
                                              observed_df=timeseries_df).mean()
 
        heads_mbgl  = utils.get_heads_mbgl(gwf)
        overview = {
            'model_id': model_id,
            'success':  True,
            **model_params,
            **metrics.to_dict(),
            **utils.heads_realism_checks(heads_mbgl),
        }
 
        return {
            'overview':         overview,
            'budget':           utils.get_budget(gwf, days_per_stress_period, time_coords),
            'median_heads':     utils.heads_median_mbgl(heads_mbgl),
            'heads_per_period': utils.means_per_period(heads_mbgl[:, 0, :, :]),
            'GW_SW_per_period': utils.means_per_period(utils.get_sw_gw_interaction(gwf)),
        }
 
    except Exception as e:
        print(f"  Model {model_id} failed: {e}")
        return {
            'overview':         {'model_id': model_id, 'success': False,
                                 'error': str(e), **model_params},
            'budget':           None,
            'median_heads':     None,
            'heads_per_period': None,
            'GW_SW_per_period': None,
        }
 
    finally:
        try:
            shutil.rmtree(sim.sim_path)  # Remove workspace to save disk space
        except Exception:
            pass
 
 
def save_result(model_id, result, outdir):
    """
    Write a single run's results to structured subdirectories.
 
    Output structure::
 
        outdir/
        +-- overview/          {model_id}_overview.json
        +-- budget/            {model_id}_budget.csv
        +-- median_heads/      {model_id}_median_heads.npy
        +-- heads_per_period/  {model_id}_heads_per_period.npy
        +-- GW_SW_per_period/  {model_id}_GW_SW_per_period.npy
 
    Parameters
    ----------
    model_id : str
    result : dict
        As returned by run_model().
    outdir : str
        Root output directory.
    """
    subdirs = ['overview', 'budget', 'median_heads', 'heads_per_period', 'GW_SW_per_period']
    paths   = {s: os.path.join(outdir, s) for s in subdirs}
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
 
    with open(os.path.join(paths['overview'], f'{model_id}_overview.json'), 'w') as f:
        json.dump(result['overview'], f, indent=4)
 
    if result['budget'] is not None:
        result['budget'].to_csv(os.path.join(paths['budget'], f'{model_id}_budget.csv'))
 
    for key in ('median_heads', 'heads_per_period', 'GW_SW_per_period'):
        if result[key] is not None:
            np.save(os.path.join(paths[key], f'{model_id}_{key}.npy'), result[key])
 
 
# =============================================================================
# MONTE CARLO PARAMETER SAMPLING
# =============================================================================
 
def generate_parameter_combinations(params, num_samples):
    """
    Draw random parameter combinations for Monte Carlo simulation.
 
    Parameters
    ----------
    params : dict
        Parameter specification dict (see MC_PARAMS).
        Each entry: {'dist': 'uniform'|'log-uniform', 'min': float, 'max': float}.
    num_samples : int
 
    Returns
    -------
    list of dict
        Each dict maps parameter names to sampled np.float64 values.
    """
    samples = []
    for _ in range(num_samples):
        sample = {}
        for param_name, cfg in params.items():
            if cfg['dist'] == 'uniform':
                value = np.random.uniform(cfg['min'], cfg['max'])
            elif cfg['dist'] == 'log-uniform':
                value = np.exp(np.random.uniform(np.log(cfg['min']), np.log(cfg['max'])))
            else:
                raise ValueError(f"Unknown distribution: '{cfg['dist']}'")
            sample[param_name] = np.float64(value)
        samples.append(sample)
    return samples
 
 
# =============================================================================
# ENSEMBLE RUNNER
# =============================================================================
 
def run_ensemble(model_configs, input_arrays, obs_input, time_coords,
                 timeseries_df, perioddata, days_per_stress_period, base_ws, outdir):
    """
    Run the full Monte Carlo ensemble serially, saving outputs after each run.
 
    Parameters
    ----------
    model_configs : list of dict
        Parameter combinations from generate_parameter_combinations().
    input_arrays : dict
        Shared spatial arrays (mask_da, dem_mean, initial_conditions,
        CHD_ds, DRN_RIV_ds, recharge_sources).
    obs_input : dict
    time_coords : array-like
    timeseries_df : pd.DataFrame
    perioddata : list of tuple
    days_per_stress_period : float
    base_ws : str
    outdir : str
 
    Returns
    -------
    dict
        Maps 'model_N' to the result dict for each realisation.
    """
    results = {}
    for i, config in enumerate(model_configs):
        model_id = f'model_{i}'
        print(f"Running {model_id} ({i + 1}/{len(model_configs)})")
        result = run_model(
            model_id, config, input_arrays, obs_input, time_coords,
            timeseries_df, perioddata, days_per_stress_period, base_ws,
        )
        save_result(model_id, result, outdir)
        results[model_id] = result
    return results
 
 
#%% =============================================================================
# SETUP MODELS AND PERFORM MONTE CARLO ANALYSIS
# =============================================================================

# --- Monte Carlo parameter sampling ---
print(f"\nGenerating {N_RUNS} parameter combinations...")
model_configs = generate_parameter_combinations(MC_PARAMS, N_RUNS)
 
# --- Run ensemble ---
os.makedirs(BASE_WS, exist_ok=True)
os.makedirs(OUTDIR,  exist_ok=True)
 
print(f"Starting Monte Carlo ensemble ({N_RUNS} runs)...\n")
t0 = time.time()
 
results = run_ensemble(
    model_configs, 
    input_arrays, 
    obs_input, 
    time_coords,
    timeseries_df, 
    perioddata, 
    days_per_stress_period, 
    BASE_WS, 
    OUTDIR,
    )
 
elapsed = time.time() - t0
print(f"\nEnsemble complete.")
print(f"Total time   : {elapsed:.1f} s")
print(f"Avg per run  : {elapsed / N_RUNS:.1f} s")

