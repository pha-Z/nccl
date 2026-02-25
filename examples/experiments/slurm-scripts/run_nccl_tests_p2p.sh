#!/bin/bash
# NCCL Profiler overhead: P2P single-node using NVIDIA nccl-tests (sendrecv_perf).
# Reproducible with upstream nccl-tests.
# nccl-tests: https://github.com/NVIDIA/nccl-tests

#SBATCH --job-name=nccl_tests_p2p
#SBATCH --account=p_lv_nccl

#SBATCH --time=00:15:00

#SBATCH -e err_nccl_tests_p2p-%J
#SBATCH -o out_nccl_tests_p2p-%J

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2000
#SBATCH --gres=gpu:4

# --- Paths ---
NCCL_TESTS_HOME="/projects/p047/p_lv_nccl/nccl-tests"
NCCL_PATH="/data/cat/ws/s0949177-nccl_profiler/nccl"

# --- Job Commands ---

echo "=========================================="
echo "NCCL Profiler Overhead - P2P (nccl-tests sendrecv_perf, single-node)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo ""

module purge
module load release/24.04 GCC/13.3.0 CUDA/12.6.0 OpenMPI/5.0.3

export LD_LIBRARY_PATH=/data/cat/ws/s0949177-nccl_profiler/nccl/build/lib:$LD_LIBRARY_PATH

EXE="${NCCL_TESTS_HOME}/build/sendrecv_perf"
EMPTY_PLUGIN="${NCCL_PATH}/ext-profiler/empty-profiler/libnccl-profiler-empty.so"

if [ ! -f "$EXE" ]; then
    echo "ERROR: nccl-tests binary not found: $EXE"
    echo "Build nccl-tests: cd $NCCL_TESTS_HOME && make CUDA_HOME=\$CUDA_HOME NCCL_HOME=$NCCL_PATH"
    exit 1
fi

if [ ! -f "$EMPTY_PLUGIN" ]; then
    echo "WARNING: Empty profiler plugin not found at $EMPTY_PLUGIN"
fi

RESULTS_DIR="nccl_tests_results_p2p_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

NITER=1000000
WARMUP=100
OPTS="-b 8 -e 64 -f 2 -g 4 -n $NITER -w $WARMUP"

echo ""
echo "=========================================="
echo "Test 1: Run WITHOUT profiler"
echo "=========================================="
unset NCCL_PROFILER_PLUGIN
unset NCCL_PROFILE_EVENT_MASK
$EXE $OPTS 2>&1 | tee "$RESULTS_DIR/without_profiler.txt"

echo ""
echo "=========================================="
echo "Test 2: Run WITH empty profiler"
echo "=========================================="
export NCCL_PROFILER_PLUGIN="$EMPTY_PLUGIN"
export NCCL_PROFILE_EVENT_MASK=4095
$EXE $OPTS 2>&1 | tee "$RESULTS_DIR/with_empty_profiler.txt"

echo ""
echo "Results saved in: $RESULTS_DIR/"
echo "success"
