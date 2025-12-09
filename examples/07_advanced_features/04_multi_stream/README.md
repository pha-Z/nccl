# NCCL Advanced Example: Multi-Stream Independent Operations

This example demonstrates using multiple independent CUDA streams for concurrent NCCL operations, enabling parallel communication on different data buffers.

## Overview

When you have independent data transfers that don't depend on each other, using multiple streams allows them to execute concurrently. This example shows:
- Multiple independent NCCL operations on separate streams
- Concurrent execution of AllReduce operations on different buffers
- Proper synchronization of independent streams

## What This Example Does

1. **Creates multiple independent streams** per GPU
2. **Launches independent AllReduce operations** on each stream
3. **Demonstrates concurrent execution** of unrelated collectives
4. **Verifies results** from all streams

## Building and Running

### Build
```shell
cd examples/07_advanced_features/04_multi_stream
make [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```

### Run
```shell
./multi_stream_example
```

## Key Concepts

### Multiple Independent Streams
```cpp
// Create multiple streams per GPU
cudaStream_t stream_a, stream_b, stream_c;
cudaStreamCreate(&stream_a);
cudaStreamCreate(&stream_b);
cudaStreamCreate(&stream_c);

// Independent operations on different buffers
ncclGroupStart();
ncclAllReduce(buf_a, buf_a, size, ncclFloat, ncclSum, comm, stream_a);
ncclAllReduce(buf_b, buf_b, size, ncclFloat, ncclSum, comm, stream_b);
ncclAllReduce(buf_c, buf_c, size, ncclFloat, ncclSum, comm, stream_c);
ncclGroupEnd();
```

## Expected Output

```
Starting Multi-Stream Independent Operations example
Using 4 GPUs with 3 independent streams each
Initializing buffers for each stream...
Launching independent AllReduce operations:
  - Stream A: AllReduce on buffer A
  - Stream B: AllReduce on buffer B
  - Stream C: AllReduce on buffer C
All operations launched
Synchronizing all streams...
Verifying results...
  Stream A: PASSED
  Stream B: PASSED
  Stream C: PASSED
All verifications passed!
```

## When to Use Multi-Stream

- **Independent data**: Multiple buffers that don't depend on each other
- **Parallel gradient sync**: Different model layers synchronized in parallel
- **Maximizing bandwidth**: Saturate network with concurrent transfers
- **Hiding latency**: Overlap multiple independent operations

## Performance Considerations

- **Resource contention**: Multiple streams compete for GPU and network resources
- **Memory bandwidth**: Concurrent operations share memory bandwidth
- **Network saturation**: Can better utilize available network bandwidth
- **Synchronization overhead**: Need to manage multiple stream sync points

