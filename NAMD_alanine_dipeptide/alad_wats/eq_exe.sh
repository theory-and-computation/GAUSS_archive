#----------------------------#
# Run GPR-ABF                #
#----------------------------#
# First simulation picks up from equilibration.
# Subsequent windows restart from previous simulations.


export jobstep=$1
export nsteps=2000

path=$2
export out_prefix=${path}/output/${jobstep}/${jobstep}
export root="${path}/alad_wats"

# initialize

echo $path
mkdir -p "${path}/output/${jobstep}"
echo "running simulation: ${jobstep}"

# root="${path}/output/alad_wats"
# root="${path}/alad_wats"

cd "$root" || { echo "Failed to cd to $root"; exit 1; }

# run simulation
/dfs7/elizaml4/sinail/namd/NAMD_3.0b7_Linux-x86_64-multicore/namd3 \
    +p8 +setcpuaffinity ${root}/eq.conf \
    > "${out_prefix}.log"




