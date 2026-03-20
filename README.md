# Cleaning The Ganga and Agri-Water: Paper 2

This repository contains data and code for the article:

## Changes in groundwater-surface water interactions following two centuries of irrigation practices and groundwater use in the Upper Ganges-Yamuna interfluve, North India.

* **Authors**: F.J.G. van Broekhoven\*, S.C. Dekker,  J. Griffioen, A. Bhagwat, and P.P. Schot
* **Published in**: ... , DOI: ...
* **Corresponding Author**: Frank van Broekhoven (f.j.g.vanbroekhoven@uu.nl), [ORCiD](https://orcid.org/0009-0008-1593-9842)


### MODFLOW 6 Monte Carlo Analysis

This repository contains all Python scripts used to set up, run, and analyse a transient MODFLOW 6 groundwater model of the Upper Ganges–Yamuna Interfluve, India. A Monte Carlo approach is used to quantify uncertainty in subsurface properties and recharge/abstraction components, covering the period 1800–2016.

---

## Study area

The model domain covers the **Hindon River Basin** within the Upper Ganges–Yamuna interfluve, Uttar Pradesh, India. The basin is characterised by intensive irrigated agriculture and rapidly growing urban centres (including Saharanpur, Muzaffarnagar, Meerut, and Ghaziabad), making it one of the most groundwater-stressed regions in the Indo-Gangetic Plain.

---

## Model overview

| Property | Value |
|---|---|
| Model code | MODFLOW 6 (Langevin et al., 2017) |
| Python framework | FloPy (Bakker et al., 2016) |
| Grid | 231 rows × 108 columns, 1 km × 1 km cells |
| Layers | 1 (unconfined) |
| Simulation period | 1800–2016 (yearly stress periods) |
| Monte Carlo runs | 1500 |


---

## Repository structure

```
├── 1_model_input/          # Raw data, intermediate files, and final model input
├── 2_model_output/         # Model run results (overviews, budgets, spatial maps)
├── 3_python_scripts/       # All Python scripts (preprocessing, model, postprocessing)
└── README.md
```

See the README in each subdirectory for details.

---

## Workflow

The scripts are numbered to reflect the processing order:

```
01  Boundary condition schematisation
02  Observation well preparation
03  Rainfall recharge
04  Canal leakage recharge
05  Irrigation demand & return flow
06  Municipal demand & return flow
07  Industrial demand
08  Combine recharge & abstraction components
    ↓
11  MODFLOW model setup & Monte Carlo runs
    ↓
21  Postprocessing: model input figures
22  Postprocessing: model output figures
```

---

## Citation

When using any material from this repository for your projects, please cite the article as specified: .....

---

## Contact

Frank van Broekhoven (f.j.g.vanbroekhoven@uu.nl), [ORCiD](https://orcid.org/0009-0008-1593-9842), 
Utrecht University, Copernicus Institute of Sustainable Development  
