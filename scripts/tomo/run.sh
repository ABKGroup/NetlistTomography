#!/bin/bash
export DESIGN=$1
export SITE_COUNT_V=$2
export SITE_COUNT_H=$3
export FLIP=$4
export REF_DIR=$5
base_dir=$6
run_dir="${base_dir}/${DESIGN}_${SITE_COUNT_V}_${SITE_COUNT_H}_${FLIP}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SCRIPT_DIR="${script_dir}"
mkdir -p ${run_dir}
cd ${run_dir}
mkdir -p log

module load innovus/21.1
innovus -64 -overwrite -log log/innovus.log -files ${script_dir}/run_invs.tcl
