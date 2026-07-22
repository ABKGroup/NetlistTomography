# This script was written and developed by ABKGroup students at UCSD VLSI CAD Lab. However, the underlying commands and reports are copyrighted by Cadence.
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.

################################################################################
# netlist_tomo_utils.tcl
#
# Consolidated utility functions for netlist tomography workflow
#
# This file combines functions from:
#   - gen_graph_invs.tcl
#   - invs_scripts_scratch.tcl
#   - extract_report_netlist_tomography.tcl
#
# Functions are organized by functionality for easier maintenance
################################################################################

################################################################################
# SECTION A: FLOORPLAN UTILITIES
################################################################################

# expand_fp_by_site - Resize floorplan by specified number of sites
# Arguments:
#   site_count_h - Number of sites to add/remove horizontally
#   site_count_v - Number of sites to add/remove vertically
#   row_flip     - Row flip orientation ('f' or 's')
proc expand_fp_by_site { site_count_h site_count_v row_flip} {
  set core_width [dbget top.fplan.coreBox_sizex]
  set core_height [dbget top.fplan.coreBox_sizey]
  set site_width [lindex [lsort [dbget head.sites.size_x]] 0]
  set site_height [lindex [lsort [dbget head.sites.size_y]] 0]
  set core2bot [dbget top.fplan.core2bot]
  set core2top [dbget top.fplan.core2top]
  set core2left [dbget top.fplan.core2left]
  set core2right [dbget top.fplan.core2right]

  set new_core_width [expr {$core_width + $site_count_h * $site_width}]
  set new_core_height [expr {$core_height + $site_count_v * $site_height}]
  floorPlan -s $new_core_width $new_core_height $core2left $core2bot \
               $core2right $core2top -adjustToSite -flip $row_flip
}

# fp_io_macro_details - Save IO and macro placement details to TCL scripts
# Generates macro_details.tcl and io_details.tcl if macros/IOs exist
proc fp_io_macro_details { } {
  if { [dbget top.insts.cell.subClass block -e -u] == "block" } {
    deselectAll
    select_obj [dbget top.insts.cell.subClass block -p2]
    dbset selected.pStatus placed
    writeFPlanScript -fileName macro_details.tcl -selected
    deselectAll
  }

  if { [dbget top.terms.pStatus -e -u] == "fixed" } {
    select_obj [dbget top.terms]
    writeFPlanScript -fileName io_details.tcl -selected
    deselectAll
  }
}

################################################################################
# SECTION B: GRAPH/NET HELPER FUNCTIONS
################################################################################

# get_net_source - Extract the source pin or instance of a net
# Arguments:
#   nPtr  - Net pointer
#   isPin - If 1, return pin name; if 0, return instance name (default: 0)
# Returns: Source pin/instance name or empty string on error
proc get_net_source { nPtr {isPin 0} } {
  set term_source [dbget [dbget ${nPtr}.terms.direction input -p ].name -e]
  if { $isPin } {
    set inst_source [dbget [dbget ${nPtr}.instTerms.isOutput 1 -p ].name -e]
  } else {
    set inst_source [dbget [dbget ${nPtr}.instTerms.isOutput 1 -p \
        ].inst.name -e]
  }

  if { $term_source != "" } {
    return [concat {*}$term_source]
  } elseif { $inst_source != "" } {
    return [concat {*}$inst_source]
  } else {
    set net_name [dbget ${nPtr}.name]
    puts "Error: Check source of net: $net_name"
  }
  return ""
}

# get_net_sinks - Extract all sink pins or instances of a net
# Arguments:
#   nPtr  - Net pointer
#   isPin - If 1, return pin names; if 0, return instance names (default: 0)
# Returns: Sorted unique list of sink pins/instances
proc get_net_sinks { nPtr {isPin 0} } {
  set sinks {}

  ## Add term sinks ##
  foreach outputTermPtr [dbget ${nPtr}.terms.direction output -p -e] {
    set sink_name [dbget ${outputTermPtr}.name]
    lappend sinks [concat {*}$sink_name]
  }

  ## Add inst sinks ##
  foreach inputIntTermPtr [dbget ${nPtr}.instTerms.isInput 1 -p -e] {
    if { $isPin } {
      set sink_name [dbget ${inputIntTermPtr}.name]
    } else {
      set sink_name [dbget ${inputIntTermPtr}.inst.name]
    }
    lappend sinks [concat {*}$sink_name]
  }
  return [lsort -unique $sinks]
}

# get_net_fanout - Calculate the fanout count of a net
# Arguments:
#   nPtr - Net pointer
# Returns: Total fanout count (inst terms + block terms)
proc get_net_fanout { nPtr } {
  set instTerm_count [llength [dbget ${nPtr}.instTerms.isInput 1 -e]]
  set bTerm_count [llength [dbget ${nPtr}.terms.direction output -e]]
  return [expr $instTerm_count + $bTerm_count]
}

################################################################################
# SECTION C: NETLIST EXPORT FUNCTIONS
################################################################################

# get_net_source_map - Build a dictionary mapping net names to their sources
# Returns: Dictionary with net_name -> source_instance_name mappings
# Note: Filters out clock nets, power/ground nets, and nets with term sources
proc get_net_source_map {} {
  # This function returns a dictionary mapping net name to net source
  set net_source_map {}
  foreach netPtr [dbget top.nets {.isClock == 0 && .isPwrOrGnd == 0}] {
    set net_source [get_net_source $netPtr]
    if {$net_source == ""} {
      continue
    }
    ## If net source is term skip it ##
    if { [dbget top.terms.name $net_source -e] != "" } {
      continue
    }
    set net_name [concat {*}[dbget ${netPtr}.name]]
    dict set net_source_map $net_name $net_source
  }
  return $net_source_map
}

# write_slack_pt_info - Export node timing and location data to CSV
# Arguments:
#   file_name - Output CSV file name (default: <design>_slack_pt.csv)
# Output format: Instance,Cell,Slack,ClockPeriod,Width,Height,PT_X,PT_Y
proc write_slack_pt_info { { file_name ""} } {
  ## If file_name is not specified set it to the top cell name ##
  if {$file_name == ""} {
    set design [dbget top.name]
    set file_name "${design}_slack_pt.csv"
  }

  ## First get all the output terms ##
  set terms_ptr [dbget top.insts.instTerms.isoutput 1 -p]
  set insts_name [dbget ${terms_ptr}.inst.name]
  set cells_ptr [dbget ${terms_ptr}.inst.cell]
  set cells_name [dbget ${cells_ptr}.name]
  set cells_width [dbget ${cells_ptr}.size_x]
  set cells_height [dbget ${cells_ptr}.size_y]
  set slacks [get_property [get_pins [dbget ${terms_ptr}.name]] slack_max]
  set i 0
  set fp [open $file_name w]
  puts $fp "Instance,Cell,Slack,ClockPeriod,Width,Height,PT_X,PT_Y"
  foreach term_ptr $terms_ptr {
    set term_name [dbget ${term_ptr}.name]
    set inst_name [lindex $insts_name $i]
    set cell_name [lindex $cells_name $i]
    set cell_width [lindex $cells_width $i]
    set cell_height [lindex $cells_height $i]
    set slack [lindex $slacks $i]
    set clks [get_property [get_pins $term_name] arrival_clocks]
    set clk_period [lindex [get_property $clks period] 0]
    set pt_x [dbget ${term_ptr}.pt_x]
    set pt_y [dbget ${term_ptr}.pt_y]
    puts $fp "$inst_name,$cell_name,$slack,$clk_period,$cell_width,$cell_height,$pt_x,$pt_y"
    incr i
  }
  close $fp
}

# write_edge_slack_info - Export edge timing data to CSV
# Arguments:
#   file_name - Output CSV file name (default: <design>_edge_slack_length.csv)
# Output format: Net,Source,Sink,Slack,ClockPeriod
proc write_edge_slack_info { {file_name "" } } {
  ## If file_name is not specified set it to the top cell name ##
  if {$file_name == ""} {
    set design [dbget top.name]
    set file_name "${design}_edge_slack_length.csv"
  }

  ## First get all the input terms ##
  set terms_ptr [dbget top.insts.instTerms.isinput 1 -p]
  set terms_name [dbget ${terms_ptr}.name]
  set slacks [get_property [get_pins $terms_name] slack_max]
  set net_source_map [get_net_source_map]
  set i 0
  set fp [open $file_name w]
  puts $fp "Net,Source,Sink,Slack,ClockPeriod"

  foreach term_ptr $terms_ptr {
    set term_name [dbget ${term_ptr}.name]
    set net_name [concat {*}[dbget ${term_ptr}.net.name]]
    set net_sink [concat {*}[dbget ${term_ptr}.inst.name]]

    ## If term net then skip ##
    if { ![dict exists $net_source_map $net_name] } {
      incr i
      continue
    }

    set net_source [dict get $net_source_map $net_name]
    set slack [lindex $slacks $i]
    set clks [get_property [get_pins $term_name] arrival_clocks]
    set clk_period [lindex [get_property $clks period] 0]
    puts $fp "$net_name,$net_source,$net_sink,$slack,$clk_period"
    incr i
  }
  close $fp
}

# print_hyperedge - Format and write a hyperedge to file
# Arguments:
#   fp    - File pointer for output
#   nPtr  - Net pointer
#   isPin - If 1, use pin names; if 0, use instance names (default: 0)
# Output format: net_name source_name sink1 sink2 ... sinkN
proc print_hyperedge { fp nPtr {isPin 0}} {
  set net_name [concat {*}[dbget ${nPtr}.name]]
  set source_name [get_net_source $nPtr $isPin]
  set sink_names [get_net_sinks $nPtr $isPin]
  set sinks [join $sink_names " "]
  puts $fp "$net_name $source_name $sinks"
}

# write_hypergraph - Export netlist in hypergraph (.hgr) format
# Arguments:
#   file_name - Output file base name (default: <design>)
#   isPin     - If 1, use pin names; if 0, use instance names (default: 0)
# Output: Creates <file_name>.hgr file
proc write_hypergraph { {file_name ""} {isPin 0}} {
  if {$file_name == ""} {
    set file_name [dbget top.name]
  }

  set hgr_file "${file_name}.hgr"
  set hgr_fp [open $hgr_file w]
  foreach nPtr [dbget top.nets] {
    if { [dbget ${nPtr}.isPwrOrGnd] || [get_net_fanout $nPtr] == 0 } {
      continue
    }
    print_hyperedge $hgr_fp $nPtr $isPin
  }
  close $hgr_fp
}

# write_details - Main coordinator function to export all netlist data
# Arguments:
#   output_dir - Directory for output files (default: "design_dir")
# Creates:
#   - <design>_nodes.csv (node timing/location data)
#   - <design>_edges.csv (edge timing data)
#   - <design>.hgr (hypergraph format)
proc write_details { {output_dir "design_dir"} } {
  exec mkdir -p $output_dir
  set file_name [dbget top.name]

  # Write Slack Point Info "design_hgr/${file_name}_nodes.csv"
  write_slack_pt_info "${output_dir}/${file_name}_nodes.csv"

  # Write Edge Slack Info "design_hgr/${file_name}_edges.csv"
  write_edge_slack_info "${output_dir}/${file_name}_edges.csv"

  # Write Hypergraph "design_hgr/${file_name}.hgr"
  write_hypergraph "${output_dir}/${file_name}"
}

################################################################################
# SECTION D: PATH REPORTING FUNCTIONS
################################################################################

# get_path_insts_helper - Extract instance names from timing path nets
# Arguments:
#   nets - Collection of nets in a timing path
# Returns: List of instance names along the path
proc get_path_insts_helper { nets } {
  set insts {}
  foreach_in_collection net $nets {
    set node_name [get_object_name [get_cells -of_objects \
              [get_property $net driver_pins] -quiet]]
    if { $node_name == "" } {
      continue
    }
    lappend insts $node_name
    set previou_net [get_object_name $net]
    set previous_node $node_name
  }
  return $insts
}

# write_top_paths_rpt_start_end_pairs - Generate timing path report with unique start-end pairs
# Arguments:
#   num_paths - Number of unique start-end path pairs to report
#   dir_path  - Output directory (default: ".")
#   nworst    - Number of worst paths per endpoint to analyze (default: 20)
# Output: Creates top_<num_paths>_paths_start_end_pairs.rpt with path details
proc write_top_paths_rpt_start_end_pairs {num_paths {dir_path "."} {nworst 20}} {
  set fp [open "${dir_path}/top_${num_paths}_paths_start_end_pairs.rpt" w]
  set max_paths [expr ${num_paths}*10]
  set reg2reg_paths [report_timing -from [all_registers] -to [all_registers] -max_paths $max_paths -collection -nworst ${nworst}]

  # Initialize tracking structure for seen pairs
  array set seen_pairs {}
  set unique_count 0

  set i 1
  foreach_in_collection path $reg2reg_paths {
    set start_point [get_object_name [get_cells -of_objects [get_property $path launching_point_name]]]
    set end_point [get_object_name [get_cells -of_objects [get_property $path capturing_point_name]]]

    # Create unique key for this start-end pair
    set pair_key "${start_point}::${end_point}"

    # If this start_point and end_point pair has already been seen, skip it
    if {[info exists seen_pairs($pair_key)]} {
      continue
    }

    # Mark this pair as seen
    set seen_pairs($pair_key) 1
    incr unique_count

    # Write path information
    puts $fp "Path $i:"
    set path_nets [get_property $path nets]
    set path_insts [get_path_insts_helper $path_nets]
    ## Remove the elements until the first element is the start_point
    set start_index [lsearch -exact $path_insts $start_point]
    set path_insts [lrange $path_insts $start_index end]
    set path_insts [lappend path_insts $end_point]
    set period [get_property $path phase_shift]
    puts $fp "  Instances: [join $path_insts " -> "]"
    puts $fp "  Slack: [get_property $path slack] ns"
    puts $fp "  Period: $period ns"
    puts $fp ""
    incr i

    # Stop if we've collected enough unique pairs
    if {$unique_count >= $num_paths} {
      break
    }
  }
  close $fp
}

################################################################################
# END OF netlist_tomo_utils.tcl
################################################################################
