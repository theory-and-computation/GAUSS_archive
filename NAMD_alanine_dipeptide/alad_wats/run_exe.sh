#----------------------------#
# Run GPR-ABF                #
#----------------------------#
# First simulation picks up from equilibration.
# Subsequent windows restart from previous simulations.


export jobstep=$1
export jobstep_prev=$(printf "%0${#jobstep}d" $((10#$jobstep - 1)))
export nsteps=2000

path=$2
export in_prefix=${path}/output/${jobstep_prev}/${jobstep_prev}
export out_prefix=${path}/output/${jobstep}/${jobstep}

# initialize

echo $path
mkdir -p "${path}/output/${jobstep}"
echo "running simulation: ${jobstep}"

# root='/dfs7/elizaml4/dkamp/poly_catal/enhanced_sampling/4_gpr/4_namd/alad_wats'
# root="${path}/output/alad_wats"
root="${path}/alad_wats"


# run simulation
/dfs7/elizaml4/sinail/namd/NAMD_3.0b7_Linux-x86_64-multicore/namd3 \
	+p8 +setcpuaffinity ${root}/run.conf \
	> "${path}/output/${jobstep}/${jobstep}.log"

