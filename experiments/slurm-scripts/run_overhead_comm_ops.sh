#!/bin/bash

#SBATCH --job-name=overhead_comm_ops
#SBATCH --account=p_lv_nccl

#SBATCH --time=00:10:00

#SBATCH -e err_overhead_comm_ops-%J
#SBATCH -o out_overhead_comm_ops-%J

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2000
#SBATCH --gres=gpu:4

# --- Job Commands ---

echo "=========================================="
echo "NCCL Profiler Overhead - Communicator Operations"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo ""

echo "Loading modules..."
module purge
module load release/24.04 GCC/13.3.0 CUDA/12.6.0 OpenMPI/5.0.3

export LD_LIBRARY_PATH=/data/cat/ws/s0949177-example_ws/nccl/build/lib:$LD_LIBRARY_PATH

# Paths
NCCL_PATH="/data/cat/ws/s0949177-example_ws/nccl"
EMPTY_PLUGIN="${NCCL_PATH}/ext-profiler/empty-profiler/libnccl-profiler-empty.so"
EXE="${NCCL_PATH}/experiment/03_comm_ops_overhead/comm_ops_overhead"

# Check if executable exists
if [ ! -f "$EXE" ]; then
    echo "ERROR: Executable not found. Please run 'make' in experiment/03_comm_ops_overhead"
    exit 1
fi

# Check if plugin exists
if [ ! -f "$EMPTY_PLUGIN" ]; then
    echo "WARNING: Empty profiler plugin not found at $EMPTY_PLUGIN"
    echo "Please build it first: cd ext-profiler/empty-profiler && make"
fi

# Results directory
RESULTS_DIR="overhead_results_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

echo ""
echo "=========================================="
echo "Test 1: Run WITHOUT profiler"
echo "=========================================="
unset NCCL_PROFILER_PLUGIN
unset NCCL_PROFILE_EVENT_MASK
$EXE > "$RESULTS_DIR/without_profiler.json" 2>&1
cat "$RESULTS_DIR/without_profiler.json"

echo ""
echo "=========================================="
echo "Test 2: Run WITH empty profiler"
echo "=========================================="
export NCCL_PROFILER_PLUGIN="$EMPTY_PLUGIN"
export NCCL_PROFILE_EVENT_MASK=4095
$EXE > "$RESULTS_DIR/with_empty_profiler.json" 2>&1
cat "$RESULTS_DIR/with_empty_profiler.json"

echo ""
echo "=========================================="
echo "Results Summary"
echo "=========================================="
echo "Results saved in: $RESULTS_DIR/"
echo ""
echo "To compare results, extract the JSON from the output files:"
echo "  grep -A 20 '\"experiment\"' $RESULTS_DIR/without_profiler.json"
echo "  grep -A 20 '\"experiment\"' $RESULTS_DIR/with_empty_profiler.json"
echo ""
echo "success"
