#!/bin/bash

#SBATCH --job-name=profiler_03_collectives_mpi_2node_4gpu
#SBATCH --account=p_lv_nccl

#SBATCH --time=00:10:00

#SBATCH -e err_profiler_03_collectives_mpi_2node_4gpu-%J
#SBATCH -o out_profiler_03_collectives_mpi_2node_4gpu-%J

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=4000 # megabytes
#SBATCH --gres=gpu:4  # 4 GPUs per node, 2 nodes = 8 total GPUs, 8 MPI ranks

# --- Job Commands ---

echo "=========================================="
echo "NCCL AllReduce MPI Test (2 nodes, 4 GPUs each)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Total tasks: $SLURM_NTASKS"
echo "GPUs per node: 4"
echo "Total GPUs: 8"
echo ""
echo "Node configuration:"
echo "  - Node 1: MPI ranks 0,1,2,3 → GPUs 0,1,2,3 (local)"
echo "  - Node 2: MPI ranks 4,5,6,7 → GPUs 0,1,2,3 (local)"
echo ""

echo "loading modules..."
module purge
module load release/24.04 GCC/13.3.0 CUDA/12.6.0 OpenMPI/5.0.3

export LD_LIBRARY_PATH=/data/cat/ws/s0949177-nccl_profiler/nccl/build/lib:$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO

# Paths
NCCL_PATH="/data/cat/ws/s0949177-nccl_profiler/nccl"
NCCLMINIMAL_PLUGIN="${NCCL_PATH}/ext-profiler/minimal-profiler/libnccl-profiler-minimal.so"

# Example executable
EXAMPLE_DIR="${NCCL_PATH}/examples/03_collectives/02_allreduce_mpi"
EXAMPLE_EXE="${EXAMPLE_DIR}/allreduce_mpi"

# Check if example exists
if [ ! -f "$EXAMPLE_EXE" ]; then
    echo "ERROR: Example executable not found: $EXAMPLE_EXE"
    echo "Please build the example first: cd $EXAMPLE_DIR && make"
    exit 1
fi

echo "=========================================="
echo "Running AllReduce test with minimal profiler plugin"
echo "=========================================="
echo "This test should show:"
echo "  - Single communicator init() call"
echo "  - Multiple AllReduce operations (5 iterations)"
echo "  - Multi-node communication (InfiniBand between nodes)"
echo "  - Intra-node communication (NVLink within each node)"
echo ""

export NCCL_PROFILER_PLUGIN="$NCCLMINIMAL_PLUGIN"
export NCCL_PROFILE_EVENT_MASK=4095

srun $EXAMPLE_EXE

echo ""
echo "=========================================="
echo "Profiler Test Complete"
echo "=========================================="
echo ""
echo "Check the following for profiler verification:"
echo "1. minimal_profiler_dump-${SLURM_JOB_ID}/ directories (one per node)"
echo "2. Each node should have log files for all communicators created on that node"
echo "3. Look for AllReduce operations in the logs"
echo "4. Note: This tests both intra-node (NVLink) and inter-node (IB) communication"
echo ""
echo "To visualize:"
echo "  1. Gather all log files from both nodes into one directory"
echo "  2. cd ${NCCL_PATH}/ext-profiler/minimal-profiler/visualizer"
echo "  3. python parse_logs.py /path/to/combined_logs/"
echo "  4. Open visualize_timeline.html"
echo ""

echo "success"

