# This script was written and developed by ABKGroup students at UCSD VLSI CAD Lab. However, the underlying commands and reports are copyrighted by Cadence.
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
#!/bin/bash
module unload genus
module load genus/21.1
module unload innovus
module load innovus/21.1

#
# To run the Physical Synthesis (iSpatial) flow - flow2
export PHY_SYNTH=1
export clk_period=1300

# Automatically detect project root directory (2 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJ_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
export DESIGN="bp_quad"
export TOPLAY="8"
export TECH="ng45"

# Default values
CLUSTER_DEF=""
NOISE=0
EDGE_PERCENTILE=95
SEED=111

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --cluster_def)
      CLUSTER_DEF="$2"
      shift 2
      ;;
    --noise)
      NOISE="$2"
      shift 2
      ;;
    --edge_percentile)
      EDGE_PERCENTILE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Export environment variables
export CLUSTER_DEF
export NOISE
export EDGE_PERCENTILE
export SEED

mkdir log -p
genus -overwrite -log log/genus.log -no_gui -files run_genus_hybrid.tcl
innovus -64 -overwrite -log log/innovus.log -files run_invs.tcl
