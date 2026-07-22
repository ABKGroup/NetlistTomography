# This script was written and developed by ABKGroup students at UCSD VLSI CAD Lab. However, the underlying commands and reports are copyrighted by Cadence. 
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.

set rtldir ${proj_dir}/rtl/${DESIGN}
set sdc  ${rtldir}/${DESIGN}.sdc
set searchdir "$rtldir"

# DEF file for floorplan initialization
if {[info exist ::env(PHY_SYNTH)] && $::env(PHY_SYNTH) == 1} {
    set floorplan_def "${proj_dir}/inputs/def/${DESIGN}_${TECH}.def"
} else {
    set floorplan_def "${proj_dir}/inputs/def/${DESIGN}_${TECH}.def"
}

# Effort level during optimization in syn_generic -physical (or called generic) stage
# possible values are : high, medium or low
set GEN_EFF medium

# Effort level during optimization in syn_map -physical (or called mapping) stage
# possible values are : high, medium or low
set MAP_EFF high
