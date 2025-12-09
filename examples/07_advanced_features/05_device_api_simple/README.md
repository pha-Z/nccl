# NCCL Advanced Example: Device API (Simplified)

This example demonstrates the NCCL Device API, which enables GPU kernels to directly perform inter-GPU communication, allowing fusion of computation and communication within a single kernel.

## Overview

The Device API provides device-side functions that allow CUDA kernels to:
- Access peer GPU memory directly (Load Store Accessible - LSA)
- Synchronize across GPUs using LSA barriers
- Fuse computation and communication in a single kernel

This is a simplified example showing the basic Device API usage pattern.

## What This Example Does

1. **Creates device communicators** using `ncclDevCommCreate`
2. **Registers symmetric memory windows** for peer access
3. **Launches a kernel** that reads from peer GPUs and performs reduction
4. **Uses LSA barriers** for cross-GPU synchronization

## Building and Running

### Build
```shell
cd examples/07_advanced_features/05_device_api_simple
make [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```

### Run
```shell
./device_api_simple
```

## Key Concepts

### Device Communicator
```cpp
ncclDevComm devComm;
ncclDevCommRequirements reqs;
reqs.lsaBarrierCount = num_thread_blocks;
ncclDevCommCreate(comm, &reqs, &devComm);
```

### Symmetric Memory Windows
```cpp
ncclWindow_t window;
ncclCommWindowRegister(comm, buffer, size, &window, NCCL_WIN_COLL_SYMMETRIC);
```

### Device-Side Peer Access
```cpp
__global__ void kernel(ncclDevComm devComm, ncclWindow_t window, ...) {
    // Access peer memory directly
    float* peerPtr = (float*)ncclGetLsaPointer(window, offset, peer_rank);
    
    // Read from peer
    float value = peerPtr[idx];
}
```

## Expected Output

```
Starting Device API Simple example
Using 4 GPUs
Initializing device communicators...
Registering symmetric memory windows...
Launching device-side collective kernel...
Kernel completed
Verifying results...
Results verified: PASSED
```

## When to Use Device API

- **Kernel fusion**: Combine computation and communication
- **Low latency**: Reduce host-device synchronization overhead
- **Custom collectives**: Implement specialized patterns
- **Iterative algorithms**: Minimize CPU involvement

## Performance Considerations

- **Programming complexity**: More complex than host API
- **Memory ordering**: Must use proper barriers and memory ordering
- **GPU requirements**: Requires P2P-capable GPUs (NVLink or PCIe)
- **Debug difficulty**: Harder to debug than host API

## Notes

- Requires NCCL 2.19+ for Device API support
- GPUs must support symmetric memory (NVLink or PCIe P2P)
- LSA barrier count must match thread block count
- See `examples/06_device_api/` for full-featured examples

