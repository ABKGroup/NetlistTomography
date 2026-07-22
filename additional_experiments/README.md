# Additional Experiments and Results

This directory contains additional experiments and results that supplement
the paper. The underlying numbers for each study are provided as CSV files
under [`data/`](data/).

## Contents

| Study | Description | Data |
|---|---|---|
| [noise_sensitivity.md](noise_sensitivity.md) | Sensitivity of TomoGNN to noise in the placeOpt→synthesis netlist mapping, measured by cluster-stability NMI under injected feature perturbations. | [`data/noise_summary.csv`](data/noise_summary.csv) |
| [tomognn_ablation.md](tomognn_ablation.md) | One-component-removal ablations of TomoGNN (FiLM, dual-graph message passing, calibration loss) with post-routeOpt PPA. | [`data/tomognn_ablation.csv`](data/tomognn_ablation.csv) |
| [radius_M_sensitivity.md](radius_M_sensitivity.md) | Stability of the adaptive co-location radius estimate with respect to the query-sample size `M`. | [`data/radius_M_summary.csv`](data/radius_M_summary.csv) |

## Data files

- **`data/noise_summary.csv`** — cluster-stability NMI between clean-mapping and
  noisy-mapping TomoGNN results, per design and perturbation fraction (2/5/10/20%).
- **`data/tomognn_ablation.csv`** — post-routeOpt PPA (rWL, power, WNS, TNS, DRV,
  runtime) for the full model and the three one-component-removal variants.
- **`data/radius_M_summary.csv`** — estimated adaptive co-location radius
  `r_adapt` as a function of the query-sample size `M`, per design.
