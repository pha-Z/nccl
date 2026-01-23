#!/bin/bash

#SBATCH --job-name=overhead_p2p_multinode
#SBATCH --account=p_lv_nccl

#SBATCH --time=00:10:00

#SBATCH -e /slurm-logs/err_overhead_p2p_multinode-%J
#SBATCH -o /slurm-logs/out_overhead_p2p_multinode-%J

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=2000
#SBATCH --gres=gpu:4

# --- Job Commands ---

echo "=========================================="
echo "NCCL Profiler Overhead - P2P Operations (Multi-Node)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Total tasks: $SLURM_NTASKS"
echo "GPUs per node: 4"
echo "Total GPUs: 8"
echo ""

echo "Loading modules..."
module purge
module load release/24.04 GCC/13.3.0 CUDA/12.6.0 OpenMPI/5.0.3

export LD_LIBRARY_PATH=/data/cat/ws/s0949177-nccl_profiler/nccl/build/lib:$LD_LIBRARY_PATH

# Paths
NCCL_PATH="/data/cat/ws/s0949177-nccl_profiler/nccl"
EMPTY_PLUGIN="${NCCL_PATH}/ext-profiler/empty-profiler/libnccl-profiler-empty.so"
EXE="${NCCL_PATH}/experiment/02_p2p_overhead/p2p_overhead_mpi"

# Check if executable exists
if [ ! -f "$EXE" ]; then
    echo "ERROR: Executable not found. Please run 'make' in experiment/02_p2p_overhead"
    exit 1
fi

# Check if plugin exists
if [ ! -f "$EMPTY_PLUGIN" ]; then
    echo "WARNING: Empty profiler plugin not found at $EMPTY_PLUGIN"
    echo "Please build it first: cd ext-profiler/empty-profiler && make"
fi

# Results directory
RESULTS_DIR="overhead_results_multinode_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

echo ""
echo "=========================================="
echo "Test 1: Run WITHOUT profiler (Multi-Node)"
echo "=========================================="
unset NCCL_PROFILER_PLUGIN
unset NCCL_PROFILE_EVENT_MASK
srun $EXE > "$RESULTS_DIR/without_profiler.json" 2>&1
cat "$RESULTS_DIR/without_profiler.json"

echo ""
echo "=========================================="
echo "Test 2: Run WITH empty profiler (Multi-Node)"
echo "=========================================="
export NCCL_PROFILER_PLUGIN="$EMPTY_PLUGIN"
export NCCL_PROFILE_EVENT_MASK=4095
srun $EXE > "$RESULTS_DIR/with_empty_profiler.json" 2>&1
cat "$RESULTS_DIR/with_empty_profiler.json"

echo ""
echo "=========================================="
echo "Results Summary"
echo "=========================================="
echo "Results saved in: $RESULTS_DIR/"
echo ""
echo "This multi-node test measures overhead including network-related profiler callbacks."
echo "To compare results, extract the JSON from the output files:"
echo "  grep -A 10 '\"experiment\"' $RESULTS_DIR/without_profiler.json"
echo "  grep -A 10 '\"experiment\"' $RESULTS_DIR/with_empty_profiler.json"
echo ""
echo "success"
