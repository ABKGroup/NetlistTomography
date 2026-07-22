# This script was written and developed by ABKGroup students at UCSD VLSI CAD Lab. However,
# the underlying commands and reports are copyrighted by Cadence. 
# We thank Cadence for granting permission to share our research to help
# promote and foster the next generation of innovators.

## Make sure all the Macro has HALO ##
deselectAll
set top_module [dbget top.name]

exec mkdir -p def
### Remove Halo as OR do not support ###
deleteHaloFromBlock -allBlock

#### Below files can be used in the Code element to generate clustered netlist ####
defOut -netlist ./def/${top_module}.def
saveNetlist -removePowerGround ./def/${top_module}.v
saveNetlist -flat -removePowerGround ./def/${top_module}_flat.v


### Add halo ###
addHaloToBlock -allMacro $HALO_WIDTH $HALO_WIDTH $HALO_WIDTH $HALO_WIDTH

#### Unplace the standard cells ###
if { [dbget top.insts.cell.subClass core -p2 -e ] != "" } {
  dbset [dbget top.insts.cell.subClass core -p2 -e ].pStatus unplaced
}

#### Write out Macro Placed def ####
if { [dbget top.insts.cell.subClass block -p2 -e ] != "" } {
  dbset [dbget top.insts.cell.subClass block -p2 -e ].pStatus placed
}
defOut -floorplan ./def/${top_module}_fp_placed_macros.def

#### Unplace the macros ###
if { [dbget top.insts.cell.subClass block -p2 -e ] != "" } {
  dbset [dbget top.insts.cell.subClass block -p2 -e ].pStatus unplaced
}

### Remove Halo as OR do not support ###
deleteHaloFromBlock -allBlock

### Write out Pin Placed def only ###
defOut -floorplan ./def/${top_module}_fp.def


### Read the Macro Def ###
defIn ./def/${top_module}_fp_placed_macros.def

### Fix macros ###
if { [dbget top.insts.cell.subClass block -p2 -e ] != "" } {
  dbset [dbget top.insts.cell.subClass block -p2 -e ].pStatus fixed
}

### run global place during place opt ###
setPlaceMode -place_opt_run_global_place full
