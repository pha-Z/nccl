#!/bin/bash

#SBATCH --job-name=profiler_07_multi_stream
#SBATCH --account=p_lv_nccl

#SBATCH --time=00:10:00

#SBATCH -e err_profiler_07_multi_stream-%J
#SBATCH -o out_profiler_07_multi_stream-%J

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=800 # megabytes
#SBATCH --gres=gpu:4

# --- Job Commands ---

echo "loading modules..."
module purge
module load release/24.04 GCC/13.3.0 CUDA/12.6.0 OpenMPI/5.0.3

export LD_LIBRARY_PATH=/data/cat/ws/s0949177-example_ws/nccl/build/lib:$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO

# Paths
NCCL_PATH="/data/cat/ws/s0949177-example_ws/nccl"
NCCLINSPECTOR_PLUGIN="${NCCL_PATH}/ext-profiler/inspector/libnccl-profiler-inspector.so"
NCCLEXAMPLE_PLUGIN="${NCCL_PATH}/ext-profiler/example/libnccl-profiler-example.so"

EXAMPLE_DIR="${NCCL_PATH}/examples/07_advanced_features/04_multi_stream"
EXAMPLE_EXE="${EXAMPLE_DIR}/multi_stream_example"

echo "== ncclinspector plugin =="
export NCCL_PROFILER_PLUGIN="$NCCLINSPECTOR_PLUGIN"
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=1000000 # 1 second
export NCCL_INSPECTOR_DUMP_DIR="nccl-inspector-multi-stream-${SLURM_JOB_ID}"
export NCCL_INSPECTOR_DUMP_VERBOSE=1
$EXAMPLE_EXE

echo "== nccl example plugin =="
export NCCL_PROFILER_PLUGIN="$NCCLEXAMPLE_PLUGIN"
export NCCL_PROFILE_EVENT_MASK=4095
export NCCL_PROFILE_DUMP_FILE="nccl_example_plugin_multi_stream_${SLURM_JOB_ID}"
unset NCCL_INSPECTOR_ENABLE
unset NCCL_INSPECTOR_DUMP_DIR
unset NCCL_INSPECTOR_DUMP_VERBOSE
$EXAMPLE_EXE

echo "success"

