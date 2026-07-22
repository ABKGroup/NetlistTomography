# This script was written and developed by ABKGroup students at UCSD VLSI CAD Lab. However,
# the underlying commands and reports are copyrighted by Cadence. 
# We thank Cadence for granting permission to share our research to help
# promote and foster the next generation of innovators.
proc extract_from_timing_rpt {timing_rpt} {
    set wns ""
    set tns ""
    set hc ""
    set vc ""
    set flag "0"
    # puts "File name $timing_rpt"
    set fp [open $timing_rpt r]
    zlib push gunzip $fp
    while { [gets $fp line] >= 0 } {
        # puts "Lins is : $line"
        if { $flag == 0 } {
            set words [split $line "|"]
        } else {
            set words [split $line]
        }
        if {[string map {" " "" } [lindex $words 1]] == "WNS(ns):"} {
            set wns [string map {" " "" } [lindex $words 2]]
        } elseif {[string map {" " "" } [lindex $words 1]] == "TNS(ns):"} {
            set tns [string map {" " "" } [lindex $words 2]]
            set flag 1
        } elseif { [lindex $words 0] == "Routing" && [llength $words] == 7} {
            set hc [lindex $words 2]
            set vc [lindex $words 5]
            break
        }
    }
    close $fp
    set ans [list $wns $tns $hc $vc]
    return $ans
}

proc extract_from_power_rpt {power_rpt} {
  set power ""
  set fp [open $power_rpt r]
  while { [gets $fp line] >= 0 } {
    if { [lindex $line 0] == "Total" && [llength $line] == 3 } {
      set power [lindex $line 2]
      break
    }
  }
  return $power
}

proc extract_cell_area {} {
  set macro_area [expr  [join [dbget [dbget top.insts.cell.subClass block -p2 \
      ].area ] +]]
  set std_cell_area [expr  [join [dbget [dbget top.insts.cell.subClass \
      block -v -p2 ].area ] +]]
  return [list $macro_area $std_cell_area]
}

proc extract_wire_length {} {
  set wire_length [expr [join [dbget top.nets.wires.length ] + ]]
}

proc effective_cp { } {
  set path [report_timing -max_paths 1 -collection]
  set launch_period [get_property [get_property $path launching_clock ] period]
  set capture_period [get_property [get_property $path capturing_clock ] period]
  set period [expr max($launch_period, $capture_period)]
  if { $period == 0.0 } {
    return 1.0e+1000
  }
  set slack [get_property $path slack]
  set effective_period [expr $period - $slack]
  return $effective_period
}

proc update_sdc { {handoff "./syn_handoff"} } {
  set design [dbget top.name]
  set path [report_timing -max_paths 1 -collection]
  set launch_period [get_property [get_property $path launching_clock ] period]
  set capture_period [get_property [get_property $path capturing_clock ] period]
  set period [expr max($launch_period, $capture_period)]
  # set period [get_property $path period]
  set slack [get_property $path slack]
  set effective_period [expr $period - $slack]
  write_sdc ${design}_updated.sdc
  exec sed -i "s@period ${period}.*@period ${effective_period}@" \
     ${design}_updated.sdc
  # update_constraint_mode -ilm_sdc_files "${handoff}/${design}.sdc" -name CON \
  # -sdc_files ${design}_updated.sdc
  update_constraint_mode -name CON -sdc_files ${design}_updated.sdc
  
  set_propagated_clock [all_clocks]
  set_clock_propagation propagated 
  setDelayCalMode -reset

  setAnalysisMode -reset
  setAnalysisMode -honorClockDomains false
  setAnalysisMode -analysisType onChipVariation -cppr both
}

proc extract_report {stage} {
  setAnalysisMode -reset
  setAnalysisMode -analysisType onChipVariation -cppr both
  set drc_count [llength [dbget top.markers.type Geometry -p -e]]
  if { $stage == "preCTS" } {
    timeDesign -preCTS -prefix ${stage}
  } elseif { $stage == "postCTS" } {
    timeDesign -postCTS -prefix ${stage}
  } elseif { $stage == "postRoute" } {
    timeDesign -postRoute -prefix ${stage}
  } elseif { $stage == "postRouteOpt" } {
    timeDesign -postRoute -prefix ${stage}
  } elseif { $stage == "postRouteOptUpdate" } {
    update_sdc
    timeDesign -postRoute -prefix ${stage}
  } elseif { $stage == "postSynth" } {
    timeDesign -prePlace -prefix ${stage}
  }

  set rpt1 [extract_from_timing_rpt timingReports/${stage}.summary.gz]
  report_power > power_${stage}.rpt
  set rpt2 [extract_from_power_rpt power_${stage}.rpt]
  set rpt3 [extract_cell_area]
  set rpt4 [extract_wire_length]
  set eff_cp [effective_cp]
  # stage,core_area,standard_cell_area,macro_area,total_power,wire_length,wns,tns,h_c,v_c
  set ans "$stage,[dbget top.fplan.coreBox_area],[lindex $rpt3 1],[lindex $rpt3\
   0],$rpt2,$rpt4,[lindex $rpt1 0],[lindex $rpt1 1],[lindex $rpt1 2],[lindex \
   $rpt1 3],$drc_count,$eff_cp"
  return $ans
}
