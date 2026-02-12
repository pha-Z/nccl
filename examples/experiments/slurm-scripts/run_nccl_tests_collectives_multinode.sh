#!/bin/bash
# NCCL Profiler overhead: collectives multi-node using NVIDIA nccl-tests (MPI).
# Uses all_reduce_perf; launch with srun. Build nccl-tests with: make MPI=1 ...
# nccl-tests: https://github.com/NVIDIA/nccl-tests

#SBATCH --job-name=nccl_tests_coll_mn
#SBATCH --account=p_lv_nccl

#SBATCH --time=00:15:00

#SBATCH -e err_nccl_tests_coll_mn-%J
#SBATCH -o out_nccl_tests_coll_mn-%J

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=2000
#SBATCH --gres=gpu:4

# --- Paths ---
NCCL_TESTS_HOME="/projects/p047/p_lv_nccl/nccl-tests"
NCCL_PATH="/data/cat/ws/s0949177-nccl_profiler/nccl"

# --- Job Commands ---

echo "=========================================="
echo "NCCL Profiler Overhead - Collectives (nccl-tests, multi-node)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Total tasks: $SLURM_NTASKS"
echo ""

module purge
module load release/24.04 GCC/13.3.0 CUDA/12.6.0 OpenMPI/5.0.3

export LD_LIBRARY_PATH=/data/cat/ws/s0949177-nccl_profiler/nccl/build/lib:$LD_LIBRARY_PATH

EXE="${NCCL_TESTS_HOME}/build/all_reduce_perf"
EMPTY_PLUGIN="${NCCL_PATH}/ext-profiler/empty-profiler/libnccl-profiler-empty.so"

if [ ! -f "$EXE" ]; then
    echo "ERROR: nccl-tests binary not found: $EXE"
    echo "Build with MPI: cd $NCCL_TESTS_HOME && make MPI=1 MPI_HOME=\$MPI_HOME CUDA_HOME=\$CUDA_HOME NCCL_HOME=$NCCL_PATH"
    exit 1
fi

if [ ! -f "$EMPTY_PLUGIN" ]; then
    echo "WARNING: Empty profiler plugin not found at $EMPTY_PLUGIN"
fi

RESULTS_DIR="nccl_tests_results_coll_multinode_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

NITER=1000000
WARMUP=100
# 8 ranks, 1 GPU per rank
OPTS="-b 8 -e 128M -f 2 -g 1 -n $NITER -w $WARMUP"

echo ""
echo "=========================================="
echo "Test 1: Run WITHOUT profiler (multi-node)"
echo "=========================================="
unset NCCL_PROFILER_PLUGIN
unset NCCL_PROFILE_EVENT_MASK
srun $EXE $OPTS 2>&1 | tee "$RESULTS_DIR/without_profiler.txt"

echo ""
echo "=========================================="
echo "Test 2: Run WITH empty profiler (multi-node)"
echo "=========================================="
export NCCL_PROFILER_PLUGIN="$EMPTY_PLUGIN"
export NCCL_PROFILE_EVENT_MASK=4095
srun $EXE $OPTS 2>&1 | tee "$RESULTS_DIR/with_empty_profiler.txt"

echo ""
echo "Results saved in: $RESULTS_DIR/"
echo "success"
