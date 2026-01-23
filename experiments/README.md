# NCCL Profiler Overhead Experiments

This directory contains experiments to measure the overhead of having the NCCL profiler plugin active, even when using an empty profiler that does minimal work.

## Overview

The experiments measure the performance impact of:
- Function call overhead for profiler callbacks
- Memory allocation overhead for event/context structures
- Plugin loading overhead

## Experiments

### 1. Collectives Overhead (`01_collectives_overhead`)
Measures overhead for collective operations (AllReduce) by running 100,000 iterations with very small buffers (4 floats) to emphasize call overhead over data transfer time.

### 2. P2P Overhead (`02_p2p_overhead`)
Measures overhead for point-to-point operations (Send/Recv) by running 100,000 iterations with very small buffers.

### 3. Communicator Operations Overhead (`03_comm_ops_overhead`)
Measures overhead for communicator operations:
- Init/Destroy cycles (100,000 iterations)
- CommSplit cycles (100,000 iterations)

## Building

To build all experiments:
```bash
cd experiment
make
```

To build a specific experiment:
```bash
cd experiment/01_collectives_overhead
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

**Single-node scripts:**
```bash
cd experiment/slurm-scripts
sbatch run_overhead_collectives.sh
sbatch run_overhead_p2p.sh
sbatch run_overhead_comm_ops.sh
```

**Multi-node scripts (for network-related overhead):**
```bash
cd experiment/slurm-scripts
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
