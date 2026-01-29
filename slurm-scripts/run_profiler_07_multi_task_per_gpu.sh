#!/bin/bash

#SBATCH --job-name=profiler_07_multi_task_per_gpu
#SBATCH --account=p_lv_nccl

#SBATCH --time=00:4:00

#SBATCH -e err_profiler_07_multi_task_per_gpu-%J
#SBATCH -o out_profiler_07_multi_task_per_gpu-%J

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=2000 # megabytes
#SBATCH --gres=gpu:2  # 2 GPUs per node, 2 nodes = 4 total GPUs
#SBATCH --gpu-bind=single:2   # Assign 2 tasks to each GPU

# --- Job Commands ---

echo "=========================================="
echo "NCCL Multi-Task-Per-GPU Communication Setup (2 nodes, 2 GPUs each, 2 tasks/GPU)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Total tasks: $SLURM_NTASKS"
echo "GPUs per node: 2"
echo "Total GPUs: 4"
echo "Tasks per GPU: 2"
echo ""

echo "loading modules..."
module purge
module load release/24.04 GCC/13.3.0 CUDA/12.6.0 OpenMPI/5.0.3

export LD_LIBRARY_PATH=/data/cat/ws/s0949177-nccl_profiler/nccl/build/lib:$LD_LIBRARY_PATH

# NCCL debugging - use TRACE for detailed init debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH

# Timeout for NCCL operations (seconds) - prevents infinite hangs
export NCCL_TIMEOUT=200

# Paths - adjust these to match your environment
NCCL_PATH="/data/cat/ws/s0949177-nccl_profiler/nccl"
NCCLMINIMAL_PLUGIN="${NCCL_PATH}/ext-profiler/minimal-profiler/libnccl-profiler-minimal.so"

# Executable (multi-task-per-GPU communication setup example)
DIR="${NCCL_PATH}/examples/07_advanced_features/08_multi_task_per_gpu"
EXE="${DIR}/multi_task_per_gpu"

# Check if executable exists
if [ ! -f "$EXE" ]; then
    echo "ERROR: Executable not built. Please run 'make' in ${DIR}"
    echo "Attempting to build..."
    cd ${DIR}
    make
    cd -
fi

echo "=========================================="
echo "Test 3: minimal profiler plugin"
echo "=========================================="
export NCCL_PROFILER_PLUGIN="$NCCLMINIMAL_PLUGIN"
export NCCL_PROFILE_EVENT_MASK=4095
unset NCCL_INSPECTOR_ENABLE
unset NCCL_INSPECTOR_DUMP_DIR
unset NCCL_INSPECTOR_DUMP_VERBOSE
unset NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS
unset NCCL_PROFILE_DUMP_FILE
srun $EXE

echo ""
echo "=========================================="
echo "Profiler Test Complete"
echo "=========================================="
echo ""
echo "Check the following for profiler verification:"
echo "1. minimal_profiler_dump-${SLURM_JOB_ID}/ directory - for minimal profiler"
echo "2. Each node should have log files for all communicators created on that node"
echo "3. Look for different commId values for world, intra-GPU, inter-GPU, intra-node comms"
echo "4. Verify seqNum increments independently per communicator"
echo "5. Note: This tests both intra-node (NVLink) and inter-node (IB) communication"
echo "6. Verify rank assignments: worldRank, localRank, gpuId, intraGpuRank"
echo ""
echo "success"

