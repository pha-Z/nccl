# NCCL Advanced Features Examples

This directory contains advanced examples demonstrating complex NCCL and CUDA features that are commonly used in production distributed training and HPC applications.

## Overview

These examples showcase:
- **CUDA Graphs with NCCL**: Capturing computation and communication in reusable graphs
- **Interleaved Computation-Communication**: Pipeline parallelism to hide communication latency
- **Multiple Collectives in Groups**: Fusing operations for better performance
- **Kernel Fusion**: Combining computation kernels with communication

## Examples

### 1. CUDA Graphs with NCCL (`01_cuda_graphs/`)

Demonstrates how to capture sequences of computational kernels and NCCL collectives into CUDA Graphs for reduced launch overhead and improved performance consistency.

**Key Features:**
- Graph capture of computation + communication
- Graph instantiation and replay
- Iterative workload patterns

### 2. Interleaved Computation-Communication (`02_interleaved_computation_communication/`)

Shows pipeline parallelism by overlapping computation and communication stages using multiple CUDA streams.

**Key Features:**
- Multi-stage pipeline
- Stream-based overlap
- Event synchronization
- Latency hiding

### 3. Multiple Collectives in Group (`03_multiple_collectives_group/`)

Demonstrates grouping multiple NCCL collective operations to enable fusion and concurrent execution.

**Key Features:**
- Multiple collectives (AllReduce, Broadcast, AllGather)
- Group API usage
- Operation fusion
- Concurrent execution

### 4. Multi-Stream Independent Operations (`04_multi_stream/`)

Demonstrates using multiple independent CUDA streams for concurrent NCCL operations on different data buffers.

**Key Features:**
- Multiple independent streams per GPU
- Concurrent AllReduce operations
- Resource utilization optimization
- Parallel gradient synchronization pattern

### 5. Device API Simple (`05_device_api_simple/`)

Simplified example of the NCCL Device API for direct GPU-to-GPU communication from within CUDA kernels.

**Key Features:**
- Peer-to-peer memory access
- Direct GPU memory read patterns
- Foundation for kernel-fused communication
- See `examples/06_device_api/` for full LSA barrier examples

## Building

Each example can be built independently:

```bash
cd examples/07_advanced_features/<example_name>
make [NCCL_HOME=<path>] [CUDA_HOME=<path>]
```

Or build all examples:

```bash
cd examples/07_advanced_features
make
```

## Running

Each example can be run directly:

```bash
./<example_executable>
```

## Profiling

These examples are designed to be profiled with NCCL profiler plugins to verify that:
- All operations are correctly captured
- Timeline reconstruction is accurate
- GPU activity can be fully reconstructed from profiler output

Use the SLURM scripts in `slurm-scripts/` to profile these examples:

```bash
sbatch slurm-scripts/run_profiler_07_advanced_features_<example>.sh
```

## Use Cases

These patterns are commonly used in:
- **Distributed Training**: Overlapping gradient computation with synchronization
- **HPC Applications**: Pipeline parallelism for iterative algorithms
- **Performance Optimization**: Reducing launch overhead and hiding latency
- **Production Workloads**: Real-world patterns for maximum efficiency

## Additional Features to Explore

Other advanced features you might want to explore:
- **Non-blocking collectives**: Using `ncclComm*Async` variants
- **Multi-stream execution**: Multiple independent communication streams
- **Custom reduction operations**: User-defined reduction functions
- **Device API with graphs**: Combining device API with CUDA graphs
- **Window-based communication**: Using symmetric memory windows

## Notes

- These examples are educational and demonstrate concepts
- Production implementations may require additional optimizations
- Performance characteristics depend on hardware and network topology
- Always profile to verify behavior matches expectations

