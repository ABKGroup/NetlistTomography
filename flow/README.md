# Flow Directory

This directory contains complete synthesis and place-and-route (P&R)
flows for benchmark designs using Cadence EDA tools.

## Directory Structure

```
flow/
├── ariane/              # Ariane RISC-V 64-bit CPU core
├── bp_quad/             # BlackParrot quad-core processor
├── jpeg_encoder/        # JPEG encoder accelerator
├── swerv_wrapper/       # SweRV EH1 RISC-V core wrapper
├── pdk_ng45/            # NanGate45 PDK files
└── util/                # Shared utility scripts
```

## Design Testcases

### 1. Ariane (ariane/)
- **Description**: 64-bit RISC-V CPU core implementing RV64IMC ISA
- **Source**: OpenHW Group CVA6 core
- **Complexity**: ~150K instances after synthesis

### 2. BlackParrot Quad (bp_quad/)
- **Description**: Quad-core BlackParrot RISC-V processor
- **Source**: BlackParrot project
- **Complexity**: ~450K instances, includes SRAM macros

### 3. JPEG Encoder (jpeg_encoder/)
- **Description**: JPEG image compression accelerator
- **Source**: OpenCores
- **Complexity**: ~80K instances

### 4. SweRV Wrapper (swerv_wrapper/)
- **Description**: Western Digital SweRV EH1 RISC-V core
- **Source**: CHIPS Alliance
- **Complexity**: ~120K instances

## Running a Flow

### Prerequisites

1. **Load EDA tool modules**:
   ```bash
   module load genus      # For synthesis
   module load innovus    # For place and route
   ```

2. **Verify PDK setup**: Ensure `pdk_ng45/` contains:
   - LEF files in `lef/`
   - Liberty files in `lib/`
   - QRC files in `qrc/`

### Execution Steps

1. **Navigate to design directory**:
   ```bash
   cd ariane/  # or bp_quad/, jpeg_encoder/, swerv_wrapper/
   ```

2. **Run the flow**:
   ```bash
   bash run.sh
   ```

### What Happens During Execution

The `run.sh` script orchestrates the complete flow:

1. **Synthesis (Genus)**:
   - Reads RTL from `rtl/` directory
   - Applies timing constraints from SDC files
   - Runs synthesis with `run_genus_hybrid.tcl`
   - Outputs: Gate-level netlist, timing reports

2. **Place and Route (Innovus)**:
   - Imports synthesized netlist
   - Reads floorplan and macro placements
   - Executes P&R flow via `run_invs.tcl`:
     - Initial placement
     - Clock tree synthesis (CTS)
     - Routing
     - Optimization
   - Post-placement scripts (`post_place.tcl`) for analysis

3. **Outputs Generated**:
   - `logs/`: Tool logs and reports
   - `reports/`: Timing, area, power reports
   - `results/`: Final DEF, GDS (if enabled)
   - `genus_*/`: Synthesis database
   - `innovus_*/`: P&R database

## Configuration Files

Each design directory contains:

- **design_setup.tcl**: Design-specific parameters (top module, files)
- **mmmc_setup.tcl**: Multi-mode multi-corner setup
- **run_genus_hybrid.tcl**: Synthesis script
- **run_invs.tcl**: Place and route script
- **post_place.tcl**: Post-placement analysis
- **rtl/*.sdc**: Timing constraints

## PDK Directory (pdk_ng45/)

Contains NanGate45 open-source 45nm PDK files:

- **lef/**: Technology and macro LEF files
- **lib/**: Liberty timing libraries (.lib)
- **qrc/**: QRC technology files for parasitic extraction

The PDK includes SRAM compiler macros (fakeram45_*) for memory
instances.

## Utility Scripts (util/)

Shared scripts used across multiple designs:

- **generate_cluster_def.py**: Convert clustering results to DEF
- **analyze_path_clustering.py**: Analyze critical path clusters
- **extract_report.tcl**: Extract metrics from tool reports
- **optimize_clustering.py**: Refine clustering assignments
- **pdn_config.tcl**: Power delivery network configuration
- **adjust_sdc_clock.py**: Modify clock constraints
- **write_required_def.tcl**: Export placement DEF files

See `util/README.md` for detailed documentation on each utility script.

## Data Generation and Denoising

The netlist tomography flow creates multiple placement variations to
enable robust ML model training. This denoising process ensures the
learned placement-to-synthesis mappings generalize across different
placement solutions rather than overfitting to a single placement.

### Why Denoising?

Machine learning models trained on a single placement solution may learn
noise and placement-specific artifacts instead of fundamental synthesis
patterns. By training on multiple valid placements of the same design, the
model learns robust features that transfer across different timing targets
and placement configurations.

### Denoising Methodology

The denoising process creates variations through two complementary
mechanisms:

#### 1. Timing Variations (SDC Clock Period Adjustment)

**Script**: `util/adjust_sdc_clock.py`

Creates different timing constraints to force the placer to generate
diverse placement solutions:

- **Adjustments**: ±0.1ps or ±0.2ps clock period modifications
- **Effect**: Different timing targets lead to different placement
  trade-offs
- **Result**: Placer optimizes for varying critical paths, producing
  different but timing-valid placements

**Example**:
```bash
# Create a tighter timing constraint (faster clock)
python3 util/adjust_sdc_clock.py design.sdc -1
# Output: design_updated.sdc with 0.1ps faster clock

# Create a relaxed timing constraint (slower clock)
python3 util/adjust_sdc_clock.py design.sdc 1
# Output: design_updated.sdc with 0.1ps slower clock
```

The script automatically detects units (picoseconds or nanoseconds) and
applies appropriate adjustments. This generates placement variations that
satisfy different timing requirements while maintaining design
functionality.

#### 2. Placement Variations (Site Shifts and Flips)

**Script**: `scripts/tomo/gen_netlist_tomo.sh`

Creates additional variations through placement perturbations:

- **Horizontal site shifts**: -2, -1, 0, +1, +2 sites
- **Vertical site shifts**: -1, 0, +1 sites
- **Cell orientations**: Normal (N) and Flipped (FN/FS)

These shifts create ~30 placement variations per design, each representing
a valid placement solution with slightly different cell positions and
orientations.

**Combination**:
- 5 horizontal shifts × 3 vertical shifts × 2 orientations = 30 variations
- Plus timing-adjusted variants = comprehensive training dataset

### Benefits

1. **Robustness**: Model learns placement patterns that generalize across
   different solutions
2. **Data Augmentation**: Increases training data without requiring
   additional designs
3. **Noise Reduction**: Filters out placement-specific artifacts
4. **Better Generalization**: Learned features transfer to new designs
   with different timing constraints

The denoised training data enables the GNN to learn fundamental
placement-to-synthesis relationships that improve PPA across diverse
design scenarios.

## Customization

To run with a different design:

1. Create a new directory following the existing structure
2. Add RTL files to `rtl/`
3. Create timing constraints (SDC)
4. Modify `design_setup.tcl` with your design parameters
5. Update `mmmc_setup.tcl` for your timing corners
6. Run `bash run.sh`

## Troubleshooting

**Issue**: Module load commands fail
- **Solution**: Ensure you're on a system with environment modules
  configured for Cadence tools

**Issue**: License errors
- **Solution**: Verify Cadence license server is accessible

**Issue**: PDK files not found
- **Solution**: Check that `pdk_ng45/` is properly populated

**Issue**: Synthesis/P&R crashes
- **Solution**: Check `logs/` directory for error messages

## Notes

- Flows are configured for NanGate45 PDK (45nm open-source)
- Timing targets vary by design (see SDC files)
- Some designs include pre-placed macros (see `*_fp_placed_macros.def`)
- All paths in scripts are relative to the design directory
