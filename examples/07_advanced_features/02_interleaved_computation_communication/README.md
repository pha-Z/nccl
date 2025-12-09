# NCCL Advanced Example: Interleaved Computation and Communication

This example demonstrates pipeline parallelism by interleaving computation and communication operations to maximize GPU utilization and hide communication latency.

## Overview

In distributed training and HPC applications, it's common to overlap computation with communication to improve overall performance. This example shows:
- **Pipeline parallelism**: While one stage communicates, another stage computes
- **Multi-stage processing**: Breaking work into stages that can overlap
- **Stream management**: Using multiple CUDA streams for concurrent operations

## What This Example Does

1. **Creates a multi-stage pipeline**:
   - Stage 1: Pre-processing computation
   - Stage 2: NCCL AllReduce communication
   - Stage 3: Post-processing computation
2. **Overlaps stages** using separate CUDA streams
3. **Demonstrates pipeline parallelism** where computation and communication happen concurrently

## Building and Running

### Build
```shell
cd examples/07_advanced_features/02_interleaved_computation_communication
make [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```

### Run
```shell
./interleaved_example
```

## Key Concepts

### Pipeline Stages
```cpp
// Stage 1: Pre-processing (compute stream)
preprocess_kernel<<<...>>>(data, compute_stream);

// Stage 2: Communication (comm stream) - can overlap with next iteration's compute
ncclAllReduce(..., comm, comm_stream);

// Stage 3: Post-processing (compute stream)
postprocess_kernel<<<...>>>(data, compute_stream);
```

### Stream Synchronization
```cpp
// Use events to synchronize between streams
cudaEvent_t compute_done;
cudaEventCreate(&compute_done);
cudaEventRecord(compute_done, compute_stream);

// Communication can start after compute is ready
cudaStreamWaitEvent(comm_stream, compute_done, 0);
```

## Expected Output

```
Starting Interleaved Computation-Communication example
Using 4 GPUs
Initializing pipeline with 3 stages per iteration
Running 5 pipeline iterations...
  Iteration 0: Stage 1 (compute) -> Stage 2 (comm) -> Stage 3 (compute)
  Iteration 1: Stage 1 (compute) -> Stage 2 (comm) -> Stage 3 (compute)
  ...
Pipeline completed successfully
Results verified: PASSED
```

## When to Use Interleaving

- **Training loops**: Overlap gradient computation with gradient synchronization
- **Multi-stage pipelines**: When stages can be pipelined
- **Latency hiding**: When communication time is significant
- **Resource utilization**: Maximize GPU and network utilization

## Performance Considerations

- **Stream overhead**: Multiple streams require careful management
- **Memory bandwidth**: Overlapping operations compete for memory bandwidth
- **Synchronization**: Proper event-based synchronization is critical
- **Pipeline depth**: Deeper pipelines can hide more latency but require more memory

## Notes

- Use `cudaEvent` for fine-grained synchronization between streams
- Ensure data dependencies are respected (compute before comm, comm before next compute)
- Monitor GPU utilization to verify overlap is working
- Consider using CUDA graphs for fixed pipeline patterns

