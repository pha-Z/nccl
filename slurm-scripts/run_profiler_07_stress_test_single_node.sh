#!/bin/bash

#SBATCH --job-name=profiler_07_stress_test_single_node
#SBATCH --account=p_lv_nccl

#SBATCH --time=00:15:00

#SBATCH -e /slurm-logs/err_profiler_07_stress_test_single_node-%J
#SBATCH -o /slurm-logs/out_profiler_07_stress_test_single_node-%J

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=4000 # megabytes
#SBATCH --gres=gpu:8  # 8 GPUs on single node, 8 MPI ranks

# --- Job Commands ---

echo "=========================================="
echo "NCCL Single-Node Stress Test (8 GPUs)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Total tasks: $SLURM_NTASKS"
echo "GPUs per node: 8"
echo ""

echo "loading modules..."
module purge
module load release/24.04 GCC/13.3.0 CUDA/12.6.0 OpenMPI/5.0.3

export LD_LIBRARY_PATH=/data/cat/ws/s0949177-nccl_profiler/nccl/build/lib:$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO

# For single-node, we can use NVLink/P2P which is faster
# These environment variables help optimize for single-node
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_DISABLE=0  # Can use IB if available, but NVLink will be preferred
export NCCL_P2P_DISABLE=0  # Enable P2P (NVLink) for single-node

# Paths - adjust these to match your environment
NCCL_PATH="/data/cat/ws/s0949177-nccl_profiler/nccl"
NCCLMINIMAL_PLUGIN="${NCCL_PATH}/ext-profiler/minimal-profiler/libnccl-profiler-minimal.so"

# Example executable
EXAMPLE_DIR="${NCCL_PATH}/examples/07_advanced_features/07_stress_test"
EXAMPLE_EXE="${EXAMPLE_DIR}/stress_test"

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
echo "Running single-node stress test with minimal profiler plugin"
echo "=========================================="
echo "This test should show:"
echo "  - Multiple communicator init() calls (root, split, shrink)"
echo "  - Various collective operations (AllReduce, Broadcast, AllGather, etc.)"
echo "  - P2P operations (NVLink will be used for intra-node communication)"
echo "  - Different seqNum progressions per communicator"
echo "  - Single-node optimizations (NVLink/P2P instead of network)"
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
echo "1. minimal_profiler_dump-${SLURM_JOB_ID}/ directory (single node)"
echo "2. Should have log files for all 8 ranks and all communicators"
echo "3. Look for different commId values for root vs split vs shrunk comms"
echo "4. Verify seqNum increments independently per communicator"
echo "5. Note: Single-node should use NVLink/P2P instead of network"
echo ""
echo "To visualize:"
echo "  1. cd ${NCCL_PATH}/ext-profiler/minimal-profiler/visualizer"
echo "  2. python parse_logs.py minimal_profiler_dump-${SLURM_JOB_ID}/"
echo "  3. Open visualize_timeline.html"
echo ""
echo "success"

