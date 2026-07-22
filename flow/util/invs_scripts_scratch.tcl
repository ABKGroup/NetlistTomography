# This script was written and developed by ABKGroup students at UCSD VLSI CAD Lab. However,
# the underlying commands and reports are copyrighted by Cadence. 
# We thank Cadence for granting permission to share our research to help
# promote and foster the next generation of innovators.

proc get_source_to_sink_distance_details {} {
  set distances {}
  foreach source_ptr [dbget [dbget top.insts.cell.subClass core -p2 ].instTerms.isOutput 1 -p] {
    set sink_name [dbget ${source_ptr}.name]
    set slack [get_property [get_pins ${sink_name}] slack_max]
    if { $slack == "INFINITY" || $slack > 0.0 } {
      continue
    }
    set sink_ptrs [dbget ${source_ptr}.net.instTerms.isInput 1 -p ]
    set source_pt_x [dbget ${source_ptr}.inst.pt_x]
    set source_pt_y [dbget ${source_ptr}.inst.pt_y]
    foreach sink_ptr $sink_ptrs {
      set sink_pt_x [dbget ${sink_ptr}.inst.pt_x]
      set sink_pt_y [dbget ${sink_ptr}.inst.pt_y]
      set distance [expr {sqrt((($source_pt_x - $sink_pt_x) ** 2) + (($source_pt_y - $sink_pt_y) ** 2))}]
      lappend distances [list $source_ptr $sink_ptr $distance]
    }
  }
  ## Report Average Distance, Max Distance, Median Distance, and Standard Deviation
  set num [llength $distances]
  if {$num == 0} {
    puts "No distances found."
    return
  }
  set sum 0
  set max -Inf
  set values {}
  foreach entry $distances {
    set d [lindex $entry 2]
    set sum [expr {$sum + $d}]
    if {$d > $max} { set max $d }
    lappend values $d
  }
  set avg [expr {$sum / $num}]
  set sorted [lsort -real $values]
  if {[expr {$num % 2}] == 1} {
    set median [lindex $sorted [expr {$num / 2}]]
  } else {
    set mid [expr {$num / 2}]
    set median [expr {([lindex $sorted $mid] + [lindex $sorted [expr {$mid - 1}]]) / 2.0}]
  }
  # Standard deviation
  set sumsq 0
  foreach d $values {
    set sumsq [expr {$sumsq + ($d - $avg) * ($d - $avg)}]
  }
  set stddev [expr {sqrt($sumsq / $num)}]

  puts "Average Distance: $avg"
  puts "Max Distance: $max"
  puts "Median Distance: $median"
  puts "Standard Deviation: $stddev"
}

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

proc get_net_fanout { nPtr } {
  set instTerm_count [llength [dbget ${nPtr}.instTerms.isInput 1 -e]]
  set bTerm_count [llength [dbget ${nPtr}.terms.direction output -e]]
  return [expr $instTerm_count + $bTerm_count]
}

proc highlight_critical_cells { {slack_th 0} } {
  set terms_ptr [dbget top.insts.instTerms.isoutput 1 -p]
  set slacks [get_property [get_pins [dbget ${terms_ptr}.name]] slack_max]
  set i 0
  foreach term_ptr $terms_ptr {
    set slack [lindex $slacks $i]
    if { $slack < $slack_th } {
      set inst_name [dbget ${term_ptr}.inst.name]
      puts "Highlighting critical instance: $inst_name with slack: $slack"
      highlight $inst_name
    }
    incr i
  }
}

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

proc print_hyperedge { fp nPtr {isPin 0}} {
  set net_name [concat {*}[dbget ${nPtr}.name]]
  set source_name [get_net_source $nPtr $isPin]
  set sink_names [get_net_sinks $nPtr $isPin]
  set sinks [join $sink_names " "]
  puts $fp "$net_name $source_name $sinks"
}

proc highlight_group {} {
  foreach group [dbget top.fplan.groups.name] {highlight $group -auto_color}
}

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


proc get_cell_delay { {outptu_pin} } {
  set pin_obj [get_pins $outptu_pin]
  set timing_arc [get_arc -to $pin_obj]
  set delays [get_property $timing_arc delay_max]
  set arc_types [get_property $timing_arc arc_type]
  ## Report the highest delay for the rising_edge or falling_edge arc type ##
  ## Ignore NA or INFINITY delays ##
  ## delays and arc_types are lists, so we need to iterate through them ##
  set max_delay -Inf
  foreach delay $delays arc_type $arc_types {
    if { $arc_type == "combinational" || $arc_type == "rising_edge" || $arc_type == "falling_edge"  } {
      if { [string is double $delay] && $delay != "INFINITY" && $delay != "-INFINITY" && $delay > $max_delay } {
        set max_delay $delay
      }
    }
  }
  if { $max_delay == -Inf } {
    puts "Error: No valid delay found for pin: $outptu_pin"
    return "NA"
  }
  return $max_delay
}

proc get_cell_net_delay { } {
  set paths [report_timing -path_group reg2reg -max_paths 10000 -collection]
  set net_delays 
  set cell_delays {}
  
}

proc write_cell_delays { {output_file "cell_delays.csv"} } {
  set fp [open $output_file w]
  puts $fp "Cell,Pin,Delay"
  
  foreach cell_ptr [dbget top.insts.cell.subClass core -p2] {
    set cell_name [dbget ${cell_ptr}.name]
    # set terms_ptr [dbget top.insts.instTerms.isoutput 1 -p]
    foreach pin_ptr [dbget ${cell_ptr}.instTerms.isoutput 1 -p] {
      set pin_name [dbget ${pin_ptr}.name]
      set delay [get_cell_delay $pin_name]
      puts $fp "$cell_name,$pin_name,$delay"
    }
  }
  close $fp
}

proc get_group_slack {} {
  set paths [report_timing -path_group reg2reg -max_paths [sizeof_collection [all_registers ]] -collection]
  set group_tns 0
  set ungroup_tns 0
  foreach_in_collection path $paths {
    set slack [get_property $path slack]
    if { $slack == "INFINITY" || $slack > 0.0 } {
      continue
    }
    set end_point [get_property $path capturing_point_name]
    set cell_name [get_object_name [get_cells -of_object [get_pins $end_point]]]
    set group_name [dbget [dbget top.insts.name $cell_name -p ].group.name -e]
    if { $group_name != "" } {
      set group_tns [expr {$group_tns + $slack}]
    } else {
      set ungroup_tns [expr {$ungroup_tns + $slack}]
    }
  }
  puts "Group TNS: $group_tns Ungroup TNS: $ungroup_tns"
}

proc parse_drc_report {report_file} {
    # Initialize result
    set num_violations -1

    # Check if file exists
    if {![file exists $report_file]} {
        puts stderr "Error: File '$report_file' not found."
        return -1
    }

    # Read the file
    set fp [open $report_file r]
    set content [read $fp]
    close $fp

    # Parse Total Violations
    # Format: "Total Violations : <number> Viols."
    if {[regexp {Total Violations\s*:\s*(\d+)\s+Viols?\.} $content match violations]} {
        set num_violations $violations
    }

    return $num_violations
}

proc parse_power_report {report_file} {
    # Initialize power data dictionary
    set power_data [dict create \
        total_power "" \
        leakage_power "" \
        internal_power "" \
        switching_power "" \
        total_clock_power "" \
        power_unit "mW"]

    # Check if file exists
    if {![file exists $report_file]} {
        puts stderr "Error: File '$report_file' not found."
        return {}
    }

    # Read the file
    set fp [open $report_file r]
    set content [read $fp]
    close $fp

    # Parse Power Units
    if {[regexp {\*\s*Power Units\s*=\s*1(\w+)} $content match power_unit]} {
        dict set power_data power_unit $power_unit
    }

    # Parse Total Internal Power
    if {[regexp {Total Internal Power:\s+([\d.]+)} $content match internal_power]} {
        dict set power_data internal_power $internal_power
    }

    # Parse Total Switching Power
    if {[regexp {Total Switching Power:\s+([\d.]+)} $content match switching_power]} {
        dict set power_data switching_power $switching_power
    }

    # Parse Total Leakage Power
    if {[regexp {Total Leakage Power:\s+([\d.]+)} $content match leakage_power]} {
        dict set power_data leakage_power $leakage_power
    }

    # Parse Total Power
    if {[regexp {Total Power:\s+([\d.]+)} $content match total_power]} {
        dict set power_data total_power $total_power
    }

    # Parse Total Clock Power from Clock section
    # Look for the Total row in the Clock table
    if {[regexp {Clock\s+Internal\s+Switching\s+Leakage\s+Total.*?-+\s*\nTotal\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)} $content match clk_int clk_sw clk_leak clk_total]} {
        dict set power_data total_clock_power $clk_total
    }

    return $power_data
}

proc get_clk_buf_count {} {
    set clk_buf_count [llength [dbget [dbget top.nets.isCTSClock 1 -p ].instTerms.inst.cell.isSequential 0 -p2 -u -e ]]
    return $clk_buf_count
}

proc get_wns_tns {} {
    ## Reg2Reg
    set reg2reg_paths [report_timing -from [all_registers] -to [all_registers] -collection -max_paths [sizeof_collection [all_registers ]] -max_slack 0.0]
    set all_reg2reg_paths [report_timing -from [all_registers] -to [all_registers] -collection -max_paths [sizeof_collection [all_registers ]]]
    set wns 0.0
    set tns 0.0
    set fep [sizeof_collection $reg2reg_paths]
    set tep [sizeof_collection $all_reg2reg_paths]
    if { $fep > 0 } {
        set slack [get_property $reg2reg_paths slack]
        set wns [lindex $slack 0]
        set tns [expr [join $slack +]]
    }
    set timing_data [dict create \
        reg2reg_wns $wns \
        reg2reg_tns $tns \
        reg2reg_fep $fep \
        reg2reg_tep $tep]
    ## All paths report
    set all_paths2reg [report_timing -to [all_registers] -collection -max_paths [sizeof_collection [all_registers ]] -max_slack 0.0]
    set wns 0.0
    set tns 0.0
    if { [sizeof_collection $all_paths2reg] > 0 } {
        set slack [get_property $all_paths2reg slack]
        set wns [lindex $slack 0]
        set tns [expr [join $slack +]]
    }
    set all_paths2output [report_timing -to [all_outputs] -collection -max_paths [sizeof_collection [all_outputs]] -max_slack 0.0]
    if { [sizeof_collection $all_paths2output] > 0 } {
        set slack [get_property $all_paths2output slack]
        set wns_tmp [lindex $slack 0]
        if { $wns_tmp < $wns } {
            set wns $wns_tmp
        }
        set tns_tmp [expr [join $slack +]]
        set tns [expr $tns + $tns_tmp]
    }
    set fep [expr [sizeof_collection $all_paths2reg] + [sizeof_collection $all_paths2output]]
    set all_paths2reg [report_timing -to [all_registers] -collection -max_paths [sizeof_collection [all_registers ]]]
    set all_paths2output [report_timing -to [all_outputs] -collection -max_paths [sizeof_collection [all_outputs]]]
    set tep [expr [sizeof_collection $all_reg2reg_paths] + [sizeof_collection $all_paths2output]]
    dict set timing_data all_wns $wns
    dict set timing_data all_tns $tns
    dict set timing_data all_fep $fep
    dict set timing_data all_tep $tep
    
    return $timing_data
}

proc reg2reg_path_depth_distribution {} {
    set path_sizes {500 1000 5000 10000 50000}
    set path_details {}
    foreach size $path_sizes {
        set reg2reg_paths [report_timing -from [all_registers] -to [all_registers] -collection -max_paths $size]
        set path_depths [get_property $reg2reg_paths num_cell_arcs]
        ## path_depths is a list of number compute mean, median std, max
        set sorted_depths [lsort -integer $path_depths]
        set len [llength $sorted_depths]
        set mean [expr 1.0*([join $path_depths +]) / $len]
        set median [lindex $sorted_depths [expr $len / 2]]
        set max [lindex $sorted_depths end]
        set sum_sq 0.0
        foreach depth $path_depths {
            set sum_sq [expr $sum_sq + ($depth - $mean) * ($depth - $mean)]
        }
        set std [expr sqrt($sum_sq / $len)]
        dict set path_details $size [dict create \
            mean $mean \
            median $median \
            std $std \
            max $max]
    }
    return $path_details
}

proc write_details_rpt {} {
    set clk_buf_count [get_clk_buf_count]
    set timing_data [get_wns_tns]
    set path_depth_data [reg2reg_path_depth_distribution]
    set fp [open "design_details.rpt" w]
    puts $fp "Clock Buffer Count: $clk_buf_count"
    puts $fp ""
    puts $fp "Timing Summary:"
    foreach {key value} $timing_data {
        puts $fp "  $key : $value"
    }
    puts $fp ""
    puts $fp "Reg2Reg Path Depth Distribution:"
    foreach {size details} $path_depth_data {
        puts $fp "  For top $size paths:"
        foreach {dkey dvalue} $details {
            puts $fp "    $dkey : $dvalue"
        }
    }

    ## DRC Report
    verify_drc -limit 0 -report post_route_drc.rpt
    set drc_violations [parse_drc_report "post_route_drc.rpt"]
    puts $fp ""
    puts $fp "DRC Violations: $drc_violations"

    ## Power Report
    set power_data [parse_power_report "power_postRouteOpt.rpt"]
    puts $fp ""
    puts $fp "Power Report Summary ([dict get $power_data power_unit]):"
    puts $fp "  Total Power:        [dict get $power_data total_power] [dict get $power_data power_unit]"
    puts $fp "  Internal Power:     [dict get $power_data internal_power] [dict get $power_data power_unit]"
    puts $fp "  Switching Power:    [dict get $power_data switching_power] [dict get $power_data power_unit]"
    puts $fp "  Leakage Power:      [dict get $power_data leakage_power] [dict get $power_data power_unit]"
    puts $fp "  Total Clock Power:  [dict get $power_data total_clock_power] [dict get $power_data power_unit]"
    close $fp
    ## Generate reg2reg slack histogram
    generate_reg2reg_slack_histogram
}

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

proc write_top_paths_rpt {num_paths {dir_path "."}} {
  set fp [open "${dir_path}/top_${num_paths}_paths.rpt" w]
  set reg2reg_paths [report_timing -from [all_registers] -to [all_registers] -max_paths $num_paths -collection]
  set i 1
  foreach_in_collection path $reg2reg_paths {
    puts $fp "Path $i:"
    set path_nets [get_property $path nets]
    set end_point [get_object_name [get_cells -of_objects [get_property $path capturing_point_name]]]
    set start_point [get_object_name [get_cells -of_objects [get_property $path launching_point_name]]]
    set path_insts [get_path_insts_helper $path_nets]
    set start_index [lsearch -exact $path_insts $start_point]
    set path_insts [lrange $path_insts $start_index end]
    set path_insts [lappend path_insts $end_point]
    puts $fp "  Instances: [join $path_insts " -> "]"
    puts $fp "  Slack: [get_property $path slack] ns"
    puts $fp ""        
    incr i
  }
  close $fp
}

proc write_top_paths_rpt_start_end_pairs {num_paths {dir_path "."} {nworst 20} {max_slack ""}} {
  set fp [open "${dir_path}/top_${num_paths}_paths_start_end_pairs.rpt" w]
  set max_paths [expr ${num_paths}*${nworst}]
  if {$max_slack != ""} {
    set reg2reg_paths [report_timing -from [all_registers] -to [all_registers] -max_paths $max_paths -collection -nworst ${nworst} -max_slack $max_slack]
  } else {
    set reg2reg_paths [report_timing -from [all_registers] -to [all_registers] -max_paths $max_paths -collection -nworst ${nworst}]
  }
  
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

proc print_run_time {start_time} {
  set end_time [clock seconds]
  set run_time [expr $end_time - $start_time]
  puts "Run time: $run_time seconds"
}

proc write_current_cluster_map {step} {
  set design [dbget top.name]
  set i 0
  set fp [open ${step}/${design}_cluster_map.csv "w"]
  puts $fp "Instance,Cluster_ID"
  foreach cluster_ptr [dbget top.fplan.groups -e] {
    foreach inst [dbget ${cluster_ptr}.members.name -e] {
      puts $fp "$inst,$i"
    }
    incr i
  }
  close $fp
}

set proj_dir "$::env(PROJ_DIR)"

proc reassign_optimize_cluster {step {path_count 50000} {max_slack ""}} {
  set design [dbget top.name]
  write_details $step
  write_current_cluster_map $step
  # write_top_paths_rpt 50000 $step
  write_top_paths_rpt_start_end_pairs $path_count $step 20 $max_slack
  set inst_group_count [llength [dbget top.fplan.groups -e]]
  if { $inst_group_count > 0 } {
    deleteAllInstGroups
  }
  
  set python_script_path "${proj_dir}/flow/util/"
  
  exec python3.9 ${python_script_path}/optimize_clustering.py \
   --cluster-csv "${step}/${design}_cluster_map.csv" \
   --nodes-csv "${step}/${design}_nodes.csv" \
   --edges-csv "${step}/${design}_edges.csv" \
   --paths-rpt "${step}/top_${path_count}_paths_start_end_pairs.rpt" \
   --output-csv "${step}/${design}_optimized_cluster_map.csv" \
   --summary-file "${step}/optimization_summary.txt"
  
  exec python3.9 ${python_script_path}/generate_cluster_def.py \
    --cluster-map "${step}/${design}_optimized_cluster_map.csv" \
    --design ${design} \
    --output ${step}/${design}_optimized_cluster.def
  
  ## Analye Path and Clustering ##
  exec python3.9 ${python_script_path}/analyze_path_clustering.py \
    --cluster-csv "${step}/${design}_optimized_cluster_map.csv" \
    --paths-rpt "${step}/top_${path_count}_paths_start_end_pairs.rpt" \
    --output-csv "${step}/${design}_path_clustering_analysis.csv"
  
  defIn ${step}/${design}_optimized_cluster.def
}

proc reassign_optimize_cluster_phase {step {phase 1}} {
  set design [dbget top.name]
  write_details $step
  write_current_cluster_map $step
  # write_top_paths_rpt 50000 $step
  write_top_paths_rpt_start_end_pairs 50000 $step 20
  set inst_group_count [llength [dbget top.fplan.groups -e]]
  if { $inst_group_count > 0 } {
    deleteAllInstGroups
  }
  
  set python_script_path "${proj_dir}/flow/util/"
  
  exec python3.9 ${python_script_path}/optimize_clustering.py \
   --cluster-csv "${step}/${design}_cluster_map.csv" \
   --nodes-csv "${step}/${design}_nodes.csv" \
   --edges-csv "${step}/${design}_edges.csv" \
   --paths-rpt "${step}/top_50000_paths_start_end_pairs.rpt" \
   --output-csv "${step}/${design}_optimized_cluster_map.csv" \
   --summary-file "${step}/optimization_summary.txt" --phases $phase
  
  exec python3.9 ${python_script_path}/generate_cluster_def.py \
    --cluster-map "${step}/${design}_optimized_cluster_map.csv" \
    --design ${design} \
    --output ${step}/${design}_optimized_cluster.def
  
  ## Analye Path and Clustering ##
  exec python3.9 ${python_script_path}/analyze_path_clustering.py \
    --cluster-csv "${step}/${design}_optimized_cluster_map.csv" \
    --paths-rpt "${step}/top_50000_paths_start_end_pairs.rpt" \
    --output-csv "${step}/${design}_path_clustering_analysis.csv"
  
  defIn ${step}/${design}_optimized_cluster.def
}

