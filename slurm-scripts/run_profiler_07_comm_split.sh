#!/bin/bash

#SBATCH --job-name=profiler_07_comm_split
#SBATCH --account=p_lv_nccl

#SBATCH --time=00:10:00

#SBATCH -e /slurm-logs/err_profiler_07_comm_split-%J
#SBATCH -o /slurm-logs/out_profiler_07_comm_split-%J

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=4000 # megabytes
#SBATCH --gres=gpu:4  # Single node, multiple GPUs for comm split test

# --- Job Commands ---

echo "=========================================="
echo "NCCL CommSplit Profiler Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

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

# Example executable
EXAMPLE_DIR="${NCCL_PATH}/examples/07_advanced_features/05_comm_split"
EXAMPLE_EXE="${EXAMPLE_DIR}/comm_split"

# Check if example exists
if [ ! -f "$EXAMPLE_EXE" ]; then
    echo "ERROR: Example not built. Please run 'make' in ${EXAMPLE_DIR}"
    echo "Attempting to build..."
    cd ${EXAMPLE_DIR}
    make
    cd -
fi

echo ""
echo "=========================================="
echo "Test 1: ncclinspector plugin"
echo "=========================================="
export NCCL_PROFILER_PLUGIN="$NCCLINSPECTOR_PLUGIN"
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=1000000 # 1 second
export NCCL_INSPECTOR_DUMP_DIR="nccl-inspector-comm-split-${SLURM_JOB_ID}"
export NCCL_INSPECTOR_DUMP_VERBOSE=1
$EXAMPLE_EXE

echo ""
echo "=========================================="
echo "Test 2: minimal profiler plugin"
echo "=========================================="
echo "This test should show SEPARATE init() calls for:"
echo "  - Parent communicator (all GPUs)"
echo "  - Each child communicator (even and odd groups)"
echo ""
export NCCL_PROFILER_PLUGIN="$NCCLMINIMAL_PLUGIN"
export NCCL_PROFILE_EVENT_MASK=4095
unset NCCL_INSPECTOR_ENABLE
unset NCCL_INSPECTOR_DUMP_DIR
unset NCCL_INSPECTOR_DUMP_VERBOSE
unset NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS
unset NCCL_PROFILE_DUMP_FILE
$EXAMPLE_EXE

echo ""
echo "=========================================="
echo "Test 3: nccl example plugin"
echo "=========================================="
export NCCL_PROFILER_PLUGIN="$NCCLEXAMPLE_PLUGIN"
export NCCL_PROFILE_EVENT_MASK=4095
export NCCL_PROFILE_DUMP_FILE="nccl_example_plugin_comm_split_${SLURM_JOB_ID}"
unset NCCL_INSPECTOR_ENABLE
unset NCCL_INSPECTOR_DUMP_DIR
unset NCCL_INSPECTOR_DUMP_VERBOSE
$EXAMPLE_EXE

echo ""
echo "=========================================="
echo "Profiler Test Complete"
echo "=========================================="
echo ""
echo "Check the following for profiler verification:"
echo "1. minimal_profiler_dump-${SLURM_JOB_ID}/ directory for log files"
echo "2. Each log file should show events from BOTH parent and child communicators"
echo "3. Look for different commId values for parent vs child comms"
echo ""
echo "success"
