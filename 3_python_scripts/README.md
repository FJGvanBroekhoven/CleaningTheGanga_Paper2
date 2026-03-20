# Python scripts

Python scripts for the Upper Ganges–Yamuna interfluve MODFLOW 6 model and Monte Carlo analysis, organised in processing order.

---

## Script overview

### Preprocessing

**`01_Preprocessing_TOP_CHD_RIV_DRN_Schematisation.py`**  
Generates static spatial input for the MODFLOW boundary condition packages (DRN, RIV, CHD) and the DIS TOP surface. Starting from HydroSHEDS DEM tiles and the HydroRIVERS network, the script merges and reprojects the DEM, computes river length and area per model cell, assigns boundary condition types by Strahler order and river width, and exports a single NetCDF dataset containing levels and areas for all packages.

---

**`02_Preprocessing_OBS_Observations_wells.py`**  
Processes raw groundwater level measurements (State Ground Water Department) into clean time series and metadata files for model calibration. Converts depths below ground level (mbgl) to metres above mean sea level (m MSL) using HydroSHEDS surface elevations. Exports monthly and yearly filtered time series (minimum record length thresholds applied).

---

**`03_Preprocessing_RCH_Rainfall_Recharge.py`**  
Derives spatially and temporally explicit rainfall recharge (1800–2024, monthly) from CRU TS v4.09 precipitation and PET. A simple water balance partitions precipitation into surface runoff, actual evapotranspiration, and groundwater recharge. The period 1800–1900 is extended using long-term monthly climatological averages.

---

**`04_Preprocessing_RCH_Canal_Leakage_Recharge.py`**  
Estimates canal leakage recharge (1800–2024, monthly) by combining canal geometry (India-WRIS) with a literature-based seepage rate (Raza et al., 2013). Canal area per cell is computed from canal length × width; leakage is activated from each canal's construction year.

---

**`05_Preprocessing_RCH_Irrigation_Demand_and_Returnflow.py`**  
Computes irrigation groundwater demand and return flow recharge (1800–present, monthly). Groundwater-irrigated area is derived from Copernicus land cover, Census 2011 irrigation fractions, and a logistic hindcast curve fitted to national GW withdrawal data (UN Water, 2022). Crop water demand is calculated from CRU TS PET and district-level monthly crop coefficients (IWMI).

---

**`06_Preprocessing_RCH_Municipal_Demand_and_Returnflow.py`**  
Computes municipal groundwater demand and return flow recharge (1800–2020, monthly). Population distribution is derived from WorldPop (2017 baseline) downscaled with HYDE 3.2 hindcast indices. Per capita water use follows Joseph et al. (2021), extrapolated to 1800. Return flow is split between river-adjacent cells and pond-infiltration cells based on proximity to the HydroSHEDS river network.

---

**`07_Preprocessing_RCH_Industrial_Demand.py`**  
Computes industrial groundwater demand (1800–2020, monthly). Current abstraction rates (UP Pollution Control Board) are rasterized onto the model grid and hindcast using a logistic growth curve fitted to national Indian GW withdrawal data (UN Water, 2022).

---

**`08_Preprocessing_RCH_Combine_Recharge_and_Abstraction_Components.py`**  
Reads all individual monthly recharge and abstraction NetCDF files, masks them to the model domain, clips to the simulation period (1800–2016), and resamples to yearly totals [m³/year]. Abstraction components are stored as negative values. Exports a single combined NetCDF file used as MODFLOW forcing.

---

### MODFLOW model

**`11_MODFLOW_Model_Setup_and_Monte_Carlo_Analysis.py`**  
Sets up and runs the Monte Carlo ensemble of 1500 transient MODFLOW 6 models. Each realisation randomly samples 12 parameters (hydraulic conductivity, aquifer thickness, specific yield, river/drain conductance, and seven recharge/abstraction multiplication factors) from uniform or log-uniform prior distributions. Post-processing extracts heads, water-balance budgets, and GW–SW exchange per run; results are saved to structured output subdirectories.

Depends on: `MODFLOW_Model_Utils.py`  

---

**`MODFLOW_Model_Utils.py`**  
Utility module imported by the model script. Contains three sections:
1. MODFLOW package input constructors (CHD, RIV, DRN, RCHA)
2. Model performance metrics (ME, MAE, etc.)
3. Post-processing functions (heads, budget, GW–SW exchange)

---

### Postprocessing

**`21_Postprocessing_Analyse_and_Plot_Model_Input.py`**  
Generates figures describing the model input data (Figures 1, 4, and 6 in the manuscript): study area map, MODFLOW boundary condition categories, basin location within the Indo-Gangetic Plain, recharge/abstraction budget time series, and spatial maps of recharge components per historical period.

---

**`22_Postprocessing_Analyse_and_Plot_Model_Output.py`**  
Loads and filters Monte Carlo results, applies validity criteria, and generates output figures (Figures 3, 5, 7, 8, 9, 10 in the manuscript): parameter distributions, net recharge time series, median groundwater depth, spatial head maps, GW–SW exchange time series, and spatial GW–SW interaction maps.

---

## Requirements

    Package                        Version
    ------------------------------ -----------
    python                         3.12.7
    numpy                          2.0.2
    pandas                         2.2.3
    geopandas                      1.0.1
    matplotlib                     3.9.2
    matplotlib-scalebar            0.9.0
    scipy                          1.14.1
    shapely                        2.0.6
    rasterio                       1.4.2
    rioxarray                      0.17.0
    xarray                         2024.11.0
    flopy                          3.8.2
    openpyxl                       3.1.5

MODFLOW 6 must be installed separately and the `mf6` executable must be on
the system path. Download from [USGS](https://www.usgs.gov/software/modflow-6-usgs-modular-hydrologic-model)
