#!/bin/bash
export DESIGN=$1
export REF_DIR=$(readlink -f "$2")
base_dir=$(readlink -f "$3")
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p ${base_dir}
site_count_hs="-2 -1 0 1 2"
site_count_vs="-1 0 1"
flips="f s"


placer_job_file="${base_dir}/${DESIGN}_place_job"
if [ -f "$placer_job_file" ]; then
  rm -rf $placer_job_file
fi

for ar in $ars; do
  for util in $utils; do
    echo "${script_dir}/run.sh $DESIGN $util $ar $REF_DIR $base_dir" >> $placer_job_file
  done
done

for flip in $flips; do
  for site_count_h in $site_count_hs; do
    for site_count_v in $site_count_vs; do
      echo "${script_dir}/run.sh $DESIGN $site_count_h $site_count_v $flip $REF_DIR $base_dir" >> $placer_job_file
    done
  done
done

user_name=`whoami`

node_file="${base_dir}/${DESIGN}_node"
echo "6/ ${user_name}@hgr" > ${node_file}
echo "6/ ${user_name}@hgr" >> ${node_file}

# Run jobs in parallel using GNU Parallel (assumes parallel is in PATH)
# parallel -j 6 :::: $placer_job_file

python3 batch_synth_mapping.py --base_dir ${base_dir} --pattern "${DESIGN}_*_*_*" --design ${DESIGN} --dir_b ${base_dir}/${DESIGN}_0_0_f/post_synth --output_dir ${base_dir}/merged_mapped_data
