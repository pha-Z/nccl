#!/bin/bash

#SBATCH --job-name=profiler_07_multi_gpu_per_task_single_thread
#SBATCH --account=p_lv_nccl

#SBATCH --time=00:2:00

#SBATCH -e /slurm-logs/err_profiler_07_multi_gpu_per_task_single_thread-%J
#SBATCH -o /slurm-logs/out_profiler_07_multi_gpu_per_task_single_thread-%J

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=2000 # megabytes
#SBATCH --gres=gpu:4
#SBATCH --gpu-bind=per_task:2

# --- Job Commands ---

echo "=========================================="
echo "NCCL multi-gpu-per-task with single thread per device"
echo "(2 nodes, 2 tasks/node, 2 GPUs/task = 8 total GPUs)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Total tasks: $SLURM_NTASKS"
echo ""

echo "loading modules..."
module purge
module load release/24.04 GCC/13.3.0 CUDA/12.6.0 OpenMPI/5.0.3

export LD_LIBRARY_PATH=/data/cat/ws/s0949177-nccl_profiler/nccl/build/lib:$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO

# Topology dumping for debugging:
# NCCL_TOPO_DUMP_FILE: Dump system topology XML (GPU, CPU, PCI, network topology)
#   Source: src/init.cc:976, src/graph/topo.cc:1531
#   Only rank specified by NCCL_TOPO_DUMP_FILE_RANK (default 0) will dump
export NCCL_TOPO_DUMP_FILE="/data/cat/ws/s0949177-nccl_profiler/nccl_topo_${SLURM_JOB_ID}.xls"
export NCCL_TOPO_DUMP_FILE_RANK=0  # Only rank 0 dumps topology

# NCCL_GRAPH_DUMP_FILE: Dump graph topology XML (ring, tree, collnet graphs)
#   Source: src/graph/search.cc:1231
#   Dumps the communication graphs NCCL will use
export NCCL_GRAPH_DUMP_FILE="/data/cat/ws/s0949177-nccl_profiler/nccl_graph_${SLURM_JOB_ID}.xls"

# Paths - adjust these to match your environment
NCCL_PATH="/data/cat/ws/s0949177-nccl_profiler/nccl"
NCCLMINIMAL_PLUGIN="${NCCL_PATH}/ext-profiler/minimal-profiler/libnccl-profiler-minimal.so"

# Executable
DIR="${NCCL_PATH}/examples/07_advanced_features/10_multiple_gpu_per_task_single_thread_per_device"
EXE="${DIR}/multiple_gpu_per_task_single_thread_per_device"

# Check if executable exists
if [ ! -f "$EXE" ]; then
    echo "ERROR: Executable not built. Please run 'make' in ${DIR}"
    echo "Attempting to build..."
    cd ${DIR}
    make
    cd -
fi

echo "=========================================="
echo "Test 1: Run without profiler"
echo "=========================================="
unset NCCL_PROFILER_PLUGIN
unset NCCL_PROFILE_EVENT_MASK
unset NCCL_INSPECTOR_ENABLE
unset NCCL_INSPECTOR_DUMP_DIR
unset NCCL_INSPECTOR_DUMP_VERBOSE
unset NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS
unset NCCL_PROFILE_DUMP_FILE
srun $EXE

echo ""
echo "=========================================="
echo "Test 2: minimal profiler plugin"
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
echo "3. Look for different commId values for root vs split vs shrunk comms"
echo "4. Verify seqNum increments independently per communicator"
echo "5. Note: This tests both intra-node (NVLink) and inter-node (IB) communication"
echo ""
echo "Topology dumps (for debugging):"
echo "6. ${NCCL_TOPO_DUMP_FILE} - System topology XML (GPUs, CPUs, PCI, network)"
echo "7. ${NCCL_GRAPH_DUMP_FILE} - Communication graphs XML (ring, tree, collnet)"
echo ""
echo "success"
