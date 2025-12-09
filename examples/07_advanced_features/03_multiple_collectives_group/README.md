# NCCL Advanced Example: Multiple Collectives in a Group

This example demonstrates how to fuse multiple NCCL collective operations using `ncclGroupStart/End`, enabling better performance through operation aggregation and concurrent execution.

## Overview

NCCL's group API allows you to batch multiple collective operations together. When operations are grouped:
- They can be fused/aggregated for better performance
- Multiple operations can progress concurrently
- Launch overhead is reduced

This example shows:
- Multiple collectives in a single group
- Different collective types (AllReduce, Broadcast, AllGather)
- Grouping operations across multiple GPUs

## What This Example Does

1. **Groups multiple collectives**:
   - AllReduce on one buffer
   - Broadcast on another buffer
   - AllGather on a third buffer
2. **Executes all operations concurrently** within the group
3. **Demonstrates group semantics** for operation fusion

## Building and Running

### Build
```shell
cd examples/07_advanced_features/03_multiple_collectives_group
make [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```

### Run
```shell
./multiple_collectives_group
```

## Key Concepts

### Grouping Multiple Collectives
```cpp
NCCLCHECK(ncclGroupStart());
for (int i = 0; i < num_gpus; i++) {
    // All operations in the group are fused
    NCCLCHECK(ncclAllReduce(..., comms[i], streams[i]));
    NCCLCHECK(ncclBroadcast(..., comms[i], streams[i]));
    NCCLCHECK(ncclAllGather(..., comms[i], streams[i]));
}
NCCLCHECK(ncclGroupEnd());
```

### Group Benefits
- **Fusion**: NCCL can optimize multiple operations together
- **Concurrency**: Operations can overlap when possible
- **Reduced overhead**: Single launch instead of multiple

## Expected Output

```
Starting Multiple Collectives in Group example
Using 4 GPUs
Initializing buffers for 3 different collectives...
Executing grouped collectives:
  - AllReduce on buffer 1
  - Broadcast on buffer 2
  - AllGather on buffer 3
Group execution completed
Verifying results...
  AllReduce: PASSED
  Broadcast: PASSED
  AllGather: PASSED
All verifications passed!
```

## When to Use Groups

- **Multiple collectives**: When you have several operations to perform
- **Performance optimization**: To enable NCCL's fusion capabilities
- **Concurrent operations**: When operations can safely overlap
- **Reduced launch overhead**: For frequently called patterns

## Performance Considerations

- **Fusion opportunities**: NCCL may fuse operations when beneficial
- **Memory bandwidth**: Multiple operations compete for bandwidth
- **Network utilization**: Can better utilize network resources
- **CPU overhead**: Reduced compared to individual launches

## Notes

- All operations in a group must use the same communicator set
- Operations are enqueued but not necessarily executed in order
- Group end only guarantees enqueueing, not completion
- Use `cudaStreamSynchronize` to wait for completion

