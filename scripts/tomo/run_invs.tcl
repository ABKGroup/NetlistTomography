# This script was written and developed by ABKGroup students at UCSD VLSI CAD Lab. However, the underlying commands and reports are copyrighted by Cadence.
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.

## Load the floorplan design
setMultiCpuUsage -localCpu 16
set design $::env(DESIGN)
set site_count_h [expr {[info exists ::env(SITE_COUNT_H)] ? $::env(SITE_COUNT_H) : 0}]
set site_count_v [expr {[info exists ::env(SITE_COUNT_V)] ? $::env(SITE_COUNT_V) : 0}]
set flip [expr {[info exists ::env(FLIP)] ? $::env(FLIP) : "s"}]
set ref_dir $::env(REF_DIR)
set floorplan_enc "${ref_dir}/enc/${design}_floorplan.enc"

if { ![file exists ${floorplan_enc}] } {
  puts "ERROR: Enc file $floorplan_enc does not exist"
  exit 1
}
set script_dir $::env(SCRIPT_DIR)
## Load the helper functions
source ${script_dir}/netlist_tomo_utils.tcl

source $floorplan_enc
if { [file exists ${ref_dir}/place_pins.tcl] } {
    source ${ref_dir}/place_pins.tcl
    dbset [dbget top.terms.pStatus placed -p ].pStatus fixed
}

## Check if this is the baseline run (0, 0, "f")
## If so, write out post_synth baseline data before any modifications
if { $site_count_h == 0 && $site_count_v == 0 && $flip == "f" } {
  puts "INFO: Baseline run detected (0, 0, f) - writing post_synth data"

  ## Run pre-placement timing analysis to get accurate slack values
  timeDesign -prePlace -outDir timingReports

  ## Write baseline data to post_synth directory
  set baseline_step "post_synth"
  write_details $baseline_step
  write_top_paths_rpt_start_end_pairs 50000 $baseline_step 20

  puts "INFO: Baseline post_synth data written to ${baseline_step}/"
}

## Remove the powerplan from the floorplan
editDelete -net [dbget [dbget top.nets.isPwrOrGnd 1 -p -e ].name]

## Remove all placement and routing blockages
deletePlaceBlockage -all
deleteRouteBlk -all

## Set dont touch for instance and nets
# dbset top.insts.dontTouch sizeOk
# dbset top.nets.dontTouch true

## Write out IO and macros placement information
fp_io_macro_details

## Update the floorplan
expand_fp_by_site $site_count_h $site_count_v $flip

## Update the IO and macros placement information
if { [file exists macro_details.tcl] } {
  source macro_details.tcl
}

if { [file exists io_details.tcl] } {
  source io_details.tcl
}

## Place the design
setPlaceMode -place_global_align_macro true
setPlaceMode -place_global_place_io_pins true

## If macros are there then run refine_macro_place
if { [dbget top.insts.cell.subClass block -p -e] != "" && ($site_count_h < 0 || $site_count_v < 0) } {
  dbset [dbget top.insts.cell.subClass block -p2 ].pStatus placed
  refine_macro_place
}

if { [dbget top.insts.cell.subClass block -p -e] != "" } {
  dbset [dbget top.insts.cell.subClass block -p2 ].pStatus fixed
} 

## Set dont touch for instance and nets
# dbset top.insts.dontTouch sizeok

## Place design
place_opt_design

exec mkdir -p enc
saveDesign ./enc/${design}.enc

## Write down the graph information and features
set step post_place_opt
write_details $step
write_top_paths_rpt_start_end_pairs 50000 $step 20

exit
