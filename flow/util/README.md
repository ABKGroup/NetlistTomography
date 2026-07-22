# Utility Scripts

This directory contains shared utility scripts used across all design flows
for netlist tomography, clustering, and P&R automation.

## Python Scripts

### adjust_sdc_clock.py

**Purpose**: Adjust clock periods in SDC timing constraint files

**Description**: Modifies clock period constraints for bp_clk, bp_clock,
core_clock, or clk signals. Automatically detects units (picoseconds or
nanoseconds) and applies specified adjustments.

**Usage**:
```bash
python3 adjust_sdc_clock.py <input.sdc> <adjustment>
```

**Parameters**:
- `input.sdc`: Input SDC file path
- `adjustment`: 0, 1, 2, -1, or -2
  - 0: No change (copy only)
  - 1: +0.1ps (+0.0001ns)
  - 2: +0.2ps (+0.0002ns)
  - -1: -0.1ps (-0.0001ns)
  - -2: -0.2ps (-0.0002ns)

**Output**: `<basename>_updated.sdc`

---

### analyze_path_clustering.py

**Purpose**: Analyze timing path clustering quality metrics

**Description**: Computes statistics on how instances along critical timing
paths are distributed across clusters. Calculates cluster cuts (transitions
between different clusters) and unique clusters per path.

**Usage**:
```bash
python3 analyze_path_clustering.py \
  --cluster-csv <cluster_map.csv> \
  --paths-rpt <timing_paths.rpt> \
  --output-csv <analysis.csv>
```

**Inputs**:
- Cluster mapping CSV (Instance, Cluster_ID columns)
- Timing paths report from Innovus

**Outputs**:
- Per-path metrics: cuts, unique clusters
- Statistics: average, max, median, standard deviation

---

### generate_cluster_def.py

**Purpose**: Convert cluster mapping CSV to DEF format

**Description**: Generates a DEF (Design Exchange Format) file with GROUPS
statements that can be read by Cadence Innovus to apply soft placement
guides.

**Usage**:
```bash
python3 generate_cluster_def.py \
  --cluster-map <cluster_map.csv> \
  --design <design_name> \
  --output <output.def>
```

**Inputs**:
- Cluster mapping CSV with Instance and Cluster_ID columns

**Outputs**:
- DEF file with GROUPS section for soft guides

**DEF Format**:
- VERSION 5.8
- DESIGN statement
- GROUPS with cluster assignments
- Each group lists member instances

---

### optimize_clustering.py

**Purpose**: Refine clustering assignments for improved PPA

**Description**: Location-aware cluster optimization that minimizes timing
path cuts while preserving spatial locality and timing criticality. Uses a
three-phase approach:

1. Assign unmapped instances using BFS topology + physical location
2. Refine boundary instances with distance-aware scoring
3. Merge strategically close clusters with high interaction

**Usage**:
```bash
python3 optimize_clustering.py \
  --cluster-csv <cluster_map.csv> \
  --nodes-csv <nodes.csv> \
  --edges-csv <edges.csv> \
  --paths-rpt <timing_paths.rpt> \
  --output-csv <optimized_cluster_map.csv> \
  --summary-file <summary.txt> \
  [--phases <1|2|3>]
```

**Inputs**:
- Initial cluster mapping CSV
- Node locations CSV (x, y coordinates, slack)
- Graph edges CSV (connectivity)
- Timing paths report

**Outputs**:
- Optimized cluster mapping CSV
- Optimization summary with statistics

**Features**:
- BFS-based topological distance calculation
- Path criticality based on slack/period ratio
- Maximum cluster size constraint (1.1x growth limit)
- Physical distance awareness

---

## TCL Scripts

### extract_report.tcl

**Purpose**: Extract key metrics from EDA tool reports

**Description**: Parses timing, area, and power reports from Cadence tools
and extracts structured data for analysis.

**Usage**: Source in Innovus or Genus
```tcl
source extract_report.tcl
extract_from_timing_rpt <report_file>
```

---

### invs_scripts_scratch.tcl

**Purpose**: Innovus helper procedures for netlist tomography

**Description**: Collection of TCL procedures for:
- Writing cluster mappings
- Generating timing path reports
- Applying soft guides from DEF files
- Running optimization workflows
- Extracting design data (nodes, edges, locations)

**Key Procedures**:
- `write_current_cluster_map`: Export instance clustering
- `write_top_paths_rpt_start_end_pairs`: Generate timing paths
- `reassign_optimize_cluster`: Run full optimization flow
- `write_details`: Export node and edge data

**Usage**: Source in Innovus
```tcl
source invs_scripts_scratch.tcl
reassign_optimize_cluster <output_dir>
```

---

### mmmc_setup.tcl

**Purpose**: Multi-mode multi-corner (MMMC) setup configuration

**Description**: Configures timing analysis corners for different
technologies. Sets library units and creates analysis views.

**Supported Technologies**:
- asap7 (7nm)
- nangate45 (45nm)

**Usage**: Source in design_setup.tcl or run scripts

---

### pdn_config.tcl

**Purpose**: Power delivery network (PDN) configuration

**Description**: Defines power grid specifications including:
- Power/ground net names
- Stripe widths and spacing
- Metal layer assignments
- Via definitions

**Usage**: Source before PDN generation

---

### pdn_flow.tcl

**Purpose**: PDN generation flow script

**Description**: Automated flow for creating power delivery networks in
Innovus. Handles:
- Power ring generation
- Power stripe creation
- Via insertion
- DRC checking

**Usage**: Source in Innovus after floorplan
```tcl
source pdn_flow.tcl
```

---

### write_required_def.tcl

**Purpose**: Export placement DEF files

**Description**: Writes out DEF files containing placement information for
specific analysis or transfer between tools.

**Usage**: Source in Innovus
```tcl
source write_required_def.tcl
```

---

## Typical Workflow

1. **Run P&R flow** (Innovus with run_invs.tcl)
2. **Extract design data**:
   ```tcl
   write_details <output_dir>
   write_current_cluster_map <output_dir>
   write_top_paths_rpt_start_end_pairs 50000 <output_dir> 20
   ```

3. **Optimize clustering**:
   ```bash
   python3 optimize_clustering.py \
     --cluster-csv cluster_map.csv \
     --nodes-csv nodes.csv \
     --edges-csv edges.csv \
     --paths-rpt paths.rpt \
     --output-csv optimized_cluster_map.csv
   ```

4. **Generate DEF**:
   ```bash
   python3 generate_cluster_def.py \
     --cluster-map optimized_cluster_map.csv \
     --design ariane \
     --output cluster.def
   ```

5. **Analyze results**:
   ```bash
   python3 analyze_path_clustering.py \
     --cluster-csv optimized_cluster_map.csv \
     --paths-rpt paths.rpt \
     --output-csv analysis.csv
   ```

6. **Apply soft guides**: Load DEF in Innovus and re-run P&R

---

## Dependencies

### Python Scripts
- Python 3.7+
- Standard library modules: csv, argparse, math, statistics, re,
  collections

### TCL Scripts
- Cadence Innovus (for invs_scripts_scratch.tcl)
- Cadence Genus or Innovus (for mmmc_setup.tcl)

---

## Notes

- All Python scripts include detailed docstrings with usage examples
- TCL procedures assume Cadence tool context (Innovus/Genus)
- Paths in scripts are relative to project root (${proj_dir})
- Cluster IDs use -1 for unmapped/unassigned instances
