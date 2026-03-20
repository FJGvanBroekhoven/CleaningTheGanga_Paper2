# Model input

All input data  for the Upper Ganges–Yamuna interfluve MODFLOW 6 model, organised into three subdirectories reflecting the processing stage.

```
1_model_input/
├── 1_original_data/      # Raw downloaded datasets (read-only)
├── 2_temp_data/          # Intermediate files produced by preprocessing scripts
└── 3_model_input_data/   # Final model-ready input files consumed by the MODFLOW script
```

---

## 1_original_data

Raw input datasets downloaded from external sources. These files are not modified by any script; copies or derived products are written to `2_temp_data/` or `3_model_input_data/`.

| File / folder | Description | Source |
|---|---|---|
| `HydroSHEDS/n20e070_con.tif` | HydroSHEDS DEM tile (20°N–30°N, 70°E–80°E) | Lehner et al. (2008) |
| `HydroSHEDS/n30e070_con.tif` | HydroSHEDS DEM tile (30°N–40°N, 70°E–80°E) | Lehner et al. (2008) |
| `HydroSHEDS/HydroRIVERS_crop.geojson` | River network clipped to model domain, with Strahler orders | Lehner et al. (2008) |
| `HydroSHEDS/HydroSHEDS_DEM_merged_cropped.tif` | Merged full-resolution HydroSHEDS DEM clipped to model domain | Lehner et al. (2008) |
| `CRUTS_v4_09/cru_ts4.09.1901.2024.pre.dat.nc` | CRU TS v4.09 monthly precipitation [mm/month], 1901–2024 | Harris et al. (2020) |
| `CRUTS_v4_09/cru_ts4.09.1901.2024.pet.dat.nc` | CRU TS v4.09 monthly PET [mm/day], 1901–2024 | Harris et al. (2020) |
| `HYDE3_2/HYDE3_2_population_data_1800-2017.nc` | HYDE 3.2 population count per cell, 1800–2017 | Klein Goldewijk et al. (2017) |
| `HYDE3_2/HYDE3_2_total_irrigated_data_1800-2017.nc` | HYDE 3.2 total irrigated area per cell, 1800–2017 | Klein Goldewijk et al. (2017) |
| `Initial_Conditions/starting_heads.nc` | Initial groundwater heads for the IC package [m a.s.l.] | Script 11 (warm-up run) |
| `WorldPop/ind_ppp_20[15-20]_1km_Aggregated_UNadj.tif` | WorldPop gridded population, India 2015–2020, 1 km | Bondarenko et al. (2025) |
| `Land_Cover_Copernicus_2015.tif` | Copernicus Global Land Cover 2015 (class 40 = cropland) | Buchhorn et al. (2020) |
| `Census_2011/Canal_irrigated.tif` | Fraction of agricultural area irrigated by canal water | Government of India (2011) |
| `Census_2011/GW_irrigated.tif` | Fraction of agricultural area irrigated by groundwater | Government of India (2011) |
| `Districts_IND_adm2.geojson` | District polygons, India (administrative level 2) | Government of India |
| `monthly_crop_coefficients_per_district.csv` | Monthly crop coefficients per district | Cai et al. (2010) / INTACH (2017) |
| `irrigation_canals.geojson` | Irrigation canal network with type and construction year | India-WRIS (n.d.) |
| `Observation_wells/State_GWD_Hindon_WL.xlsx` | Raw groundwater level measurements (mbgl), all monitored wells | State GWD (n.d.) |
| `domestic_water_use_per_capita.csv` | Per capita domestic water use [m³/person/year], 1975–2015 | Joseph et al. (2021) |
| `Industry.geojson` | Industrial facility locations with reported discharge rates [m³/day] | UP PCB (n.d.) |
| `Model_Mask.tif` | Model active-cell mask (1 = active, 0 = south boundary, 2 = Yamuna, 3 = Ganga, −1 = inactive) | Derived |
| `Hindon_Subbasin_polygon.geojson` | Hindon River sub-catchment polygon | HydroBASINS |
| `Hindon_catchment_mask.tif` | Binary raster mask of the Hindon sub-catchment | HydroBASINS |
| `World_Countries_polygon.geojson` | Country polygons for locator map | Natural Earth |
| `Indo_Gangetic_Basin_polygon.geojson` | Indo-Gangetic Basin outline | Bonsor et al. (2017) |
| `labels_points.geojson` | City label locations for study area map | Derived |


---

## 2_temp_data

Intermediate NetCDF/GeoTIFF files produced by the preprocessing scripts (`03`–`07`). These are the inputs to `08_Preprocessing_RCH_Combine_Recharge_and_Abstraction_Components.py` and are not directly consumed by MODFLOW.

| File | Description | Produced by |
|---|---|---|
| `CRUTS_derived_rainfall_recharge.nc` | Monthly rainfall, PET, runoff, recharge, AET [m³/month], 1800–2024 | Script 03 |
| `irrigation_canal_leakage.nc` | Monthly canal leakage recharge [m³/month], 1800–2024 | Script 04 |
| `irrigation_demand.nc` | Monthly irrigation groundwater demand [m³/month], 1800–present | Script 05 |
| `irrigation_returnflow.nc` | Monthly irrigation return flow recharge [m³/month], 1800–present | Script 05 |
| `Municipal_demand.nc` | Monthly municipal groundwater demand [m³/month], 1800–2020 | Script 06 |
| `Municipal_return_flow.nc` | Monthly municipal return flow components [m³/month], 1800–2020 | Script 06 |
| `industrial_demand.nc` | Monthly industrial groundwater demand [m³/month], 1800–2020 | Script 07 |

---

## 3_model_input_data

Final model-ready input files consumed directly by the MODFLOW model script (`11`) and postprocessing scripts (`21`, `22`).

| File | Description | Produced by |
|---|---|---|
| `DIS_TOP.tif` | Model top surface (mean-resampled HydroSHEDS DEM) [m a.s.l.] | Script 01 |
| `DRN_RIV_CHD.nc` | Boundary condition levels and areas for DRN, RIV, and CHD packages | Script 01 |
| `metadata_selected_observation_wells.geojson` | Observation well metadata (monthly, ≥ 50 observations) | Script 02 |
| `timeseries_data_selected_observation_wells.csv` | Monthly head time series [m MSL] for selected wells | Script 02 |
| `metadata_selected_observation_wells_yearly.geojson` | Observation well metadata (yearly, ≥ 10 years) | Script 02 |
| `timeseries_data_selected_observation_wells_yearly.csv` | Yearly head time series [m MSL] for selected wells | Script 02 |
| `recharge_and_abstraction_components_yearly.nc` | Yearly recharge and abstraction components [m³/year], 1800–2016 | Script 08 |


### `recharge_and_abstraction_components_yearly.nc` – variable reference

| Variable | Sign | Units | Description |
|---|---|---|---|
| `rainfall_rch` | + | m³/year | Rainfall recharge |
| `canal_rch` | + | m³/year | Canal leakage recharge |
| `irrigation_return_rch` | + | m³/year | Irrigation return flow recharge |
| `municipal_return_rch` | + | m³/year | Municipal return flow recharge |
| `irrigation_demand_abs` | − | m³/year | Irrigation groundwater abstraction |
| `municipal_demand_abs` | − | m³/year | Municipal groundwater abstraction |
| `industrial_demand_abs` | − | m³/year | Industrial groundwater abstraction |

### `DRN_RIV_CHD.nc` – variable reference

| Variable | Package | Description |
|---|---|---|
| `DRN_Hindon_area` / `DRN_Hindon_level` | DRN | Drain area [m²] and stage [m a.s.l.] within Hindon sub-catchment |
| `DRN_other_area` / `DRN_other_level` | DRN | Drain area and stage outside Hindon sub-catchment |
| `RIV_Hindon_area` / `RIV_Hindon_level` | RIV | River area and stage within Hindon sub-catchment |
| `RIV_other_area` / `RIV_other_level` | RIV | River area and stage outside Hindon sub-catchment |
| `CHD_Ganga_level` | CHD | Ganges boundary head [m a.s.l.] |
| `CHD_Yamuna_level` | CHD | Yamuna boundary head [m a.s.l.] |
| `CHD_soutern_level` | CHD | Southern boundary head [m a.s.l.] |
