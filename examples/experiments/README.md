# NCCL Profiler Overhead Experiments

This directory contains experiments to measure the overhead of having the NCCL profiler plugin active, even when using an empty profiler that does minimal work.

## Overview

The experiments measure the performance impact of:
- Function call overhead for profiler callbacks
- Memory allocation overhead for event/context structures
- Plugin loading overhead

## Experiments with NVIDIA nccl-tests (recommended for collectives and P2P)

**Collectives and P2P** are run using [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests) so that results are easily reproducible by others. Communicator operations (init/destroy, split, etc.) are not covered by nccl-tests; use the custom experiments below for those.

- **Location on HPC:** `/projects/p047/p_lv_nccl/nccl-tests`
- **Single-node:** `all_reduce_perf`, `sendrecv_perf` (no MPI build required).
- **Multi-node:** same binaries, launched with `srun`; build with `make MPI=1 MPI_HOME=... CUDA_HOME=... NCCL_HOME=...`.

Slurm scripts (in `slurm-scripts/`):

| Script | Test | Nodes | Binary |
|--------|------|--------|--------|
| `run_nccl_tests_collectives.sh` | AllReduce | 1 | `all_reduce_perf` |
| `run_nccl_tests_collectives_multinode.sh` | AllReduce | 2 | `all_reduce_perf` (srun) |
| `run_nccl_tests_p2p.sh` | SendRecv (P2P) | 1 | `sendrecv_perf` |
| `run_nccl_tests_p2p_multinode.sh` | SendRecv (P2P) | 2 | `sendrecv_perf` (srun) |

Scripts use the same `NCCL_PATH` as the other slurm scripts; `NCCL_TESTS_HOME` is set to `/projects/p047/p_lv_nccl/nccl-tests`. Each run uses 1M iterations, 100 warmup, and writes results into `nccl_tests_results_*_<jobid>/`.

## Custom experiments (optional)

### 1. Collectives Overhead (`01_collectives_overhead`)
Measures overhead for collective operations (AllReduce) by running 100,000 iterations with very small buffers (4 floats) to emphasize call overhead over data transfer time.

### 2. P2P Overhead (`02_p2p_overhead`)
Measures overhead for point-to-point operations (Send/Recv) by running 100,000 iterations with very small buffers.

### 3. Communicator Operations Overhead (`03_comm_ops_overhead`)
Measures overhead for communicator operations (not in nccl-tests):
- Init/Destroy cycles (100,000 iterations)
- CommSplit cycles (100,000 iterations)

## Building

To build all experiments:
```bash
cd examples/experiments
make
```

To build a specific experiment:
```bash
cd examples/experiments/01_collectives_overhead
make
```

To clean:
```bash
make clean
```

## Running

### Local Execution

First, ensure the empty profiler is built:
```bash
cd ext-profiler/empty-profiler
make
```

Then run experiments:

**Without profiler:**
```bash
unset NCCL_PROFILER_PLUGIN
./01_collectives_overhead/collectives_overhead
```

**With empty profiler:**
```bash
export NCCL_PROFILER_PLUGIN=/path/to/ext-profiler/empty-profiler/libnccl-profiler-empty.so
export NCCL_PROFILE_EVENT_MASK=4095
./01_collectives_overhead/collectives_overhead
```

### SLURM Execution

**nccl-tests (collectives and P2P, recommended):**
```bash
cd examples/experiments/slurm-scripts
# Single-node
sbatch run_nccl_tests_collectives.sh
sbatch run_nccl_tests_p2p.sh
# Multi-node (requires nccl-tests built with MPI=1)
sbatch run_nccl_tests_collectives_multinode.sh
sbatch run_nccl_tests_p2p_multinode.sh
```

**Custom single-node scripts:**
```bash
cd examples/experiments/slurm-scripts
sbatch run_overhead_collectives.sh
sbatch run_overhead_p2p.sh
sbatch run_overhead_comm_ops.sh
```

**Custom multi-node scripts (for network-related overhead):**
```bash
cd examples/experiments/slurm-scripts
sbatch run_overhead_collectives_multinode.sh
sbatch run_overhead_p2p_multinode.sh
sbatch run_overhead_comm_ops_multinode.sh
```

The multi-node scripts use 2 nodes with 4 GPUs each (8 total GPUs) and are important for measuring overhead from network-related profiler callbacks (InfiniBand communication between nodes).

## Output Format

Each experiment outputs JSON with timing information:
- `total_time_ms`: Total execution time in milliseconds
- `avg_time_per_iteration_us`: Average time per iteration in microseconds
- `total_time_ns`: Total time in nanoseconds (for precise comparison)

## Comparing Results

The overhead can be calculated as:
```
overhead_percentage = ((time_with_profiler - time_without_profiler) / time_without_profiler) * 100
overhead_per_iteration = (time_with_profiler - time_without_profiler) / num_iterations
```

## Requirements

- NCCL library built and available
- CUDA toolkit
- At least 2 GPUs (for P2P and comm ops experiments)
- GCC compiler with C++14 support
