#!/bin/bash

#SBATCH --job-name=profiler_01_communicators_pthread
#SBATCH --account=p_lv_nccl

#SBATCH --time=00:10:00

#SBATCH -e err_profiler_01_communicators_pthread-%J
#SBATCH -o out_profiler_01_communicators_pthread-%J

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=4000 # megabytes
#SBATCH --gres=gpu:4  # Single node, multiple GPUs for this example

# --- Job Commands ---

echo "loading modules..."
module purge
module load release/24.04 GCC/13.3.0 CUDA/12.6.0 OpenMPI/5.0.3

export LD_LIBRARY_PATH=/data/cat/ws/s0949177-nccl_profiler/nccl/build/lib:$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO

# Paths - adjust these to match your environment
NCCL_PATH="/data/cat/ws/s0949177-nccl_profiler/nccl"
NCCLINSPECTOR_PLUGIN="${NCCL_PATH}/ext-profiler/inspector/libnccl-profiler-inspector.so"
NCCLMINIMAL_PLUGIN="${NCCL_PATH}/ext-profiler/minimal-profiler/libnccl-profiler-minimal.so"
NCCLEXAMPLE_PLUGIN="${NCCL_PATH}/ext-profiler/example/libnccl-profiler-example.so"

# Example executable - adjust path as needed
EXAMPLE_DIR="${NCCL_PATH}/examples/01_communicators/02_one_device_per_pthread"
EXAMPLE_EXE="${EXAMPLE_DIR}/one_device_per_pthread"

echo "== ncclinspector plugin =="
export NCCL_PROFILER_PLUGIN="$NCCLINSPECTOR_PLUGIN"
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=1000000 # 1 second
export NCCL_INSPECTOR_DUMP_DIR="nccl-inspector-communicators-pthread-${SLURM_JOB_ID}"
export NCCL_INSPECTOR_DUMP_VERBOSE=1  # Enable verbose for detailed event traces
$EXAMPLE_EXE

echo "== minimal profiler plugin =="
export NCCL_PROFILER_PLUGIN="$NCCLMINIMAL_PLUGIN"
export NCCL_PROFILE_EVENT_MASK=4095  # all events enabled
unset NCCL_INSPECTOR_ENABLE
unset NCCL_INSPECTOR_DUMP_DIR
unset NCCL_INSPECTOR_DUMP_VERBOSE
unset NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS
unset NCCL_PROFILE_DUMP_FILE
$EXAMPLE_EXE

echo "== nccl example plugin =="
export NCCL_PROFILER_PLUGIN="$NCCLEXAMPLE_PLUGIN"
export NCCL_PROFILE_EVENT_MASK=4095  # all events enabled
export NCCL_PROFILE_DUMP_FILE="nccl_example_plugin_communicators_pthread_${SLURM_JOB_ID}"
unset NCCL_INSPECTOR_ENABLE
unset NCCL_INSPECTOR_DUMP_DIR
unset NCCL_INSPECTOR_DUMP_VERBOSE
$EXAMPLE_EXE

echo "success"

