# This script was written and developed by ABKGroup students at UCSD VLSI CAD Lab. However,
# the underlying commands and reports are copyrighted by Cadence. 
# We thank Cadence for granting permission to share our research to help
# promote and foster the next generation of innovators.

set proj_dir "$::env(PROJ_DIR)"
set TOPLAY $env(TOPLAY)
set DESIGN $env(DESIGN) 
set TECH $env(TECH)
source ./design_setup.tcl
source ${proj_dir}/flow/pdk_ng45/lib_setup.tcl

setMultiCpuUsage -localCpu 16

set handoff_dir "./syn_handoff"

set netlist ${handoff_dir}/${DESIGN}.v
set ref_sdc ${handoff_dir}/${DESIGN}.sdc
exec cp -rf ${ref_sdc} ${DESIGN}.sdc
exec python3.9 ${proj_dir}/flow/util/adjust_sdc_clock.py ${DESIGN}.sdc $::env(NOISE)
set sdc ./${DESIGN}_updated.sdc
source ${proj_dir}/flow/util/mmmc_setup.tcl

set rptDir summaryReport/ 
set encDir enc/

if {![file exists $rptDir/]} {
  exec mkdir $rptDir/
}

if {![file exists $encDir/]} {
  exec mkdir $encDir/
}

# default settings
set init_pwr_net VDD
set init_gnd_net VSS

# default settings
set init_verilog "$netlist"
set init_design_netlisttype "Verilog"
set init_design_settop 1
set init_top_cell "$DESIGN"
set init_lef_file "$lefs"

# MCMM setup
init_design -setup {WC_VIEW} -hold {BC_VIEW}
set_power_analysis_mode -leakage_power_view WC_VIEW -dynamic_power_view WC_VIEW

set_interactive_constraint_modes {CON}
setAnalysisMode -reset
setAnalysisMode -analysisType onChipVariation -cppr both

clearGlobalNets
globalNetConnect VDD -type pgpin -pin VDD -inst * -override
globalNetConnect VSS -type pgpin -pin VSS -inst * -override
globalNetConnect VDD -type tiehi -inst * -override
globalNetConnect VSS -type tielo -inst * -override

setOptMode -powerEffort low -leakageToDynamicRatio 0.5
setGenerateViaMode -auto true
generateVias

# basic path groups
createBasicPathGroups -expanded

# ------------------------------------------------------------------------------
# Floorplan 
# ------------------------------------------------------------------------------
## Generate the floorplan ##
defIn swerv_wrapper_fp_placed_macros.def

### Write postSynth report ###
echo "Physical Design Stage, Core Area (um^2), Standard Cell Area (um^2), Macro Area (um^2), Total Power (mW), Wirelength(um), WS(ns), TNS(ns), Congestion(H), Congestion(V)" > ${DESIGN}_DETAILS.rpt
source ${proj_dir}/flow/util/extract_report.tcl
set rpt_post_synth [extract_report postSynth]
echo "$rpt_post_synth" >> ${DESIGN}_DETAILS.rpt

### Write out the def files ###
source ${proj_dir}/flow/util/write_required_def.tcl

### Add power plan ###
source ${proj_dir}/flow/util/pdn_config.tcl
source ${proj_dir}/flow/util/pdn_flow.tcl

saveDesign ${encDir}/${DESIGN}_floorplan.enc

# ------------------------------------------------------------------------------
# Placement 
# ------------------------------------------------------------------------------
setPlaceMode -place_detail_legalization_inst_gap 1
setPlaceMode -place_detail_honor_inst_pad true
setFillerMode -fitGap true
setDesignMode -topRoutingLayer $TOP_ROUTING_LAYER
setDesignMode -bottomRoutingLayer 2

## Conditionally run cluster evaluation if CLUSTER_DEF exists
set is_cluster 1
if {[info exists ::env(CLUSTER_DEF)] && $::env(CLUSTER_DEF) ne "" && [file exists $::env(CLUSTER_DEF)]} {
  # Cluster evaluation flow
  puts "Running cluster evaluation flow with CLUSTER_DEF: $::env(CLUSTER_DEF)"
  setPlaceMode -place_opt_post_place_tcl post_place.tcl
} else {
  # Default flow - no post_place.tcl hook
  puts "Running default flow (no cluster evaluation)"
  set is_cluster 0
}

place_opt_design -out_dir $rptDir -prefix place
saveDesign $encDir/${DESIGN}_placed.enc

set rpt_pre_cts [extract_report preCTS]
echo "$rpt_pre_cts" >> ${DESIGN}_DETAILS.rpt

source ${proj_dir}/flow/util/invs_scripts_scratch.tcl
if {$is_cluster == 1} {
  reassign_optimize_cluster pre_ccopt
}

set_ccopt_property post_conditioning_enable_routing_eco 1
set_ccopt_property -cts_def_lock_clock_sinks_after_routing true
setOptMode -unfixClkInstForOpt false

create_ccopt_clock_tree_spec
ccopt_design

set_interactive_constraint_modes [all_constraint_modes -active]
set_propagated_clock [all_clocks]
set_clock_propagation propagated

saveDesign $encDir/${DESIGN}_cts.enc
set rpt_post_cts [extract_report postCTS]
echo "$rpt_post_cts" >> ${DESIGN}_DETAILS.rpt
if {$is_cluster == 1} {
  reassign_optimize_cluster post_ccopt
}

# ------------------------------------------------------------------------------
# Routing
# ------------------------------------------------------------------------------
setNanoRouteMode -drouteVerboseViolationSummary 1
setNanoRouteMode -routeWithSiDriven true
setNanoRouteMode -routeWithTimingDriven true
setNanoRouteMode -routeUseAutoVia true
setNanoRouteMode -drouteEndIteration 10

##Recommended by lib owners
# Prevent router modifying M1 pins shapes
setNanoRouteMode -routeWithViaInPin "1:1"
setNanoRouteMode -routeWithViaOnlyForStandardCellPin "1:1"

## limit VIAs to ongrid only for VIA1 (S1)
setNanoRouteMode -drouteOnGridOnly "via 1:1"
setNanoRouteMode -drouteAutoStop false
setNanoRouteMode -drouteExpAdvancedMarFix true
setNanoRouteMode -routeExpAdvancedTechnology true

#SM suggestion for solving long extraction runtime during GR
setNanoRouteMode -grouteExpWithTimingDriven false

routeDesign
#route_opt_design
saveDesign ${encDir}/${DESIGN}_route.enc

### Run DRC and LVS ###
verify_connectivity -error 0 -geom_connect -no_antenna
verify_drc -limit 0

set rpt_post_route [extract_report postRoute]
echo "$rpt_post_route" >> ${DESIGN}_DETAILS.rpt
defOut -netlist -floorplan -routing ${DESIGN}_route.def

#route_opt_design
optDesign -postRoute
set rpt_post_route [extract_report postRouteOpt]
echo "$rpt_post_route" >> ${DESIGN}_DETAILS.rpt

if { ${TECH} == "asap7" } {
  ### Add V1 vias ###
  setViaGenMode -reset
  editPowerVia -top_layer M2 -bottom_layer M1 -orthogonal_only 0 -add_vias 1
}

summaryReport -noHtml -outfile summaryReport/post_route.sum
saveDesign ${encDir}/${DESIGN}.enc
defOut -netlist -floorplan -routing ${DESIGN}.def

### Run DRC and LVS ###
verify_connectivity -error 0 -geom_connect -no_antenna
verify_drc -limit 0

## Update the sdc ##
update_constraint_mode -ilm_sdc_files $sdc -name CON -sdc_files ${DESIGN}.sdc
set_propagated_clock [all_clocks]
set_clock_propagation propagated

## Rewrite the WNS, TNS with the updated sdc file ##
set rpt_post_route [extract_report postRouteOpt]
echo "$rpt_post_route" >> ${DESIGN}_DETAILS.rpt

## Write Post Route Opt Details ##
write_details post_route_opt
write_top_paths_rpt_start_end_pairs 50000 "post_route_opt" 20

write_details_rpt

exit
