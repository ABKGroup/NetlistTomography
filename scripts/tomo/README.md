# Netlist Tomography Script

This directory contains scripts for generating netlist tomography features by
running multiple placement variations and mapping timing information across
different placements.

## Overview

The netlist tomography pipeline performs the following steps:
1. Generates multiple perturbed floorplans with different perturbations
   (site, row, and row orientation)
2. Runs placement and timing analysis for each perturbed floorplan
3. Maps synthesis netlist to each post-place-opt netlist's timing and
   location data
4. Merges results into unified CSV files for downstream PPA-based ML-driven
   clustering flow

## Quick Start

### Basic Usage

```bash
# <reference_dir>/end/<design_name>_floorplan.enc should exist.
# Based on this floorplan, the script perturbs and generates 30
# perturbed floorplans and runs place-opt for each floorplan.
./gen_netlist_tomo.sh <design_name> <reference_dir> <output_base_dir>
```


### Example

```bash
# Example from run_test.sh
ref_dir="xxx"
base_dir="yyy"
./gen_netlist_tomo.sh ariane $ref_dir $base_dir
```

## Parameters

- `design_name`: Name of the design (e.g., `ariane`, `jpeg_encoder`)
- `reference_dir`: Path to reference directory containing the initial
  floorplan (`<reference_dir>/end/<design_name>_floorplan.enc`)
- `output_base_dir`: Base directory where all perturbed floorplan runs and
  merged data will be stored

## Pipeline Details

### Step 1: Generate Perturbed Floorplans

The script creates 30 perturbed floorplans using:
- **Horizontal site perturbation**: -2, -1, 0, 1, 2 (5 values)
- **Vertical row perturbation**: -1, 0, 1 (3 values)
- **Row orientation**: f (flip), s (standard) (2 values)
- Total combinations: 5 × 3 × 2 = 30 runs

Each perturbed floorplan run is stored in:
```
<output_base_dir>/<design>_<site_v>_<site_h>_<flip>/
```

### Step 2: Run Place-Opt Jobs

The pipeline uses GNU parallel to run place-opt jobs across multiple
compute nodes. Jobs are distributed using a node file with hostnames.

Each run executes:
- Innovus placement and optimization (place-opt)
- Timing analysis
- DEF and timing report generation

### Step 3: Synthesis Netlist to Post-Place-Opt Mapping

After all perturbed floorplan runs complete, `batch_synth_mapping.py`
maps the synthesis netlist to each post-place-opt netlist, extracting
timing and location data:

```bash
python3 batch_synth_mapping.py \
  --base_dir <output_base_dir> \
  --pattern "<design>_*_*_*" \
  --design <design_name> \
  --dir_b <output_base_dir>/<design>_0_0_f/post_synth \
  --output_dir <output_base_dir>/merged_mapped_data
```

### Step 4: Output Data

The final merged data is saved in:
```
<output_base_dir>/merged_mapped_data/
├── <design>_complete_nodes.csv  # Node timing and locations
└── <design>_complete_edges.csv  # Edge timing and lengths
```

## Output Data Format

### Nodes CSV (`<design>_complete_nodes.csv`)

Columns:
- `Instance`: Cell instance name
- `Cell`: Cell type
- `ClockPeriod`: Clock period used
- `Width`, `Height`: Cell dimensions
- `Slack_1` to `Slack_30`: Slack values from each run
- `PT_X_1` to `PT_X_30`: X coordinates from each run
- `PT_Y_1` to `PT_Y_30`: Y coordinates from each run

### Edges CSV (`<design>_complete_edges.csv`)

Columns:
- `Net`: Net name
- `Source`: Source instance
- `Sink`: Sink instance
- `ClockPeriod`: Clock period used
- `Length_1` to `Length_30`: Wire lengths from each run
- `Slack_1` to `Slack_30`: Slack values from each run

## Data Location

After successful execution, merged data will be available at:
```
<output_base_dir>/merged_mapped_data/
```

This merged data serves as input to the PPA-based ML-driven clustering
pipeline described in the main ML code (see `../../ml/CLAUDE.md`).

## Requirements

- Innovus 21.1 or later
- Python 3.6+
- Required Python packages:
  - pandas
  - pathlib
  - concurrent.futures

## Component Scripts

- `gen_netlist_tomo.sh`: Main orchestrator script for generating perturbed
  floorplans
- `run.sh`: Single perturbed floorplan place-opt run script
- `run_invs.tcl`: Innovus TCL commands for place-opt
- `netlist_tomo_utils.tcl`: Utility functions for netlist processing
- `batch_synth_mapping.py`: Parallel synthesis netlist to post-place-opt
  mapping coordinator
- `syn_map_complete.py`: Single mapping job executor
- `syn_map_loc_generation.py`: Location data extraction
- `syn_map_edge_generation.py`: Edge data extraction

## Notes

- The pipeline requires access to compute cluster nodes (configured via
  node file)
- Place-opt jobs run in parallel across available nodes for all 30
  perturbed floorplans
- Typical runtime: 1-3 hours depending on design size and node
  availability
- Each run directory contains complete place-opt data including DEF,
  timing reports, and logs
