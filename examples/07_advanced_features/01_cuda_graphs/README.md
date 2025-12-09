# NCCL Advanced Example: CUDA Graphs with NCCL

This example demonstrates how to use CUDA Graphs to capture and replay sequences of computational kernels and NCCL collective operations, reducing launch overhead and improving performance consistency.

## Overview

CUDA Graphs allow you to capture a sequence of GPU operations (kernels, memory copies, NCCL collectives) into a single executable graph. This is particularly useful for:
- Reducing CPU overhead from repeated kernel launches
- Improving performance consistency in training loops
- Enabling better optimization by the CUDA runtime

NCCL operations can be captured in CUDA Graphs starting from NCCL 2.9 and CUDA 11.3.

## What This Example Does

1. **Captures a computation-communication pattern** in a CUDA Graph:
   - Computational kernel (vector addition)
   - NCCL AllReduce collective
   - Post-processing kernel
2. **Instantiates and launches the graph** multiple times
3. **Demonstrates graph replay** for iterative workloads

## Building and Running

### Build
```shell
cd examples/07_advanced_features/01_cuda_graphs
make [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```

### Run
```shell
./cuda_graphs_example
```

## Key Concepts

### CUDA Graph Capture
```cpp
cudaGraph_t graph;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// All operations in this stream are captured
kernel_preprocess<<<...>>>(...);
ncclAllReduce(..., comm, stream);
kernel_postprocess<<<...>>>(...);

cudaStreamEndCapture(stream, &graph);
```

### Graph Instantiation and Launch
```cpp
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

// Replay the graph multiple times
for (int iter = 0; iter < num_iterations; iter++) {
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);
}
```

## Expected Output

```
Starting CUDA Graphs with NCCL example
Using 4 GPUs
Initializing NCCL communicators...
Capturing CUDA graph with computation and NCCL AllReduce...
Graph captured successfully
Instantiating graph...
Graph instantiated successfully
Launching graph 10 times...
  Iteration 0 completed
  Iteration 1 completed
  ...
  Iteration 9 completed
All iterations completed successfully
Verifying results...
Results verified: PASSED
```

## When to Use CUDA Graphs

- **Training loops**: Repeated patterns of computation and communication
- **Inference pipelines**: Fixed sequences of operations
- **Performance-critical paths**: Where launch overhead matters
- **Consistent timing**: When you need predictable execution times

## Performance Considerations

- **First capture**: May have overhead, but subsequent launches are fast
- **Graph updates**: Can update parameters without recapturing (CUDA 11.8+)
- **Memory**: Graphs consume device memory for captured operations
- **Debugging**: Graph execution is less debuggable than normal launches

## Notes

- All communicators in a `ncclGroup()` must either all be captured or none captured
- Graph capture mode must be `cudaStreamCaptureModeGlobal` for NCCL
- NCCL operations in graphs are still asynchronous and require proper synchronization

