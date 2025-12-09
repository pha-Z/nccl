/*
 * NCCL Advanced Example: Device API (Simplified)
 * 
 * Demonstrates basic Device API usage for direct GPU-to-GPU communication
 * from within a CUDA kernel. This is a simplified version - see
 * examples/06_device_api/ for full-featured examples.
 */

#include "cuda_runtime.h"
#include "nccl.h"
#include <stdio.h>
#include <stdlib.h>

#define NCCLCHECK(cmd)                                                         \
  do {                                                                         \
    ncclResult_t res = cmd;                                                    \
    if (res != ncclSuccess) {                                                  \
      fprintf(stderr, "Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__,   \
              ncclGetErrorString(res));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t err = cmd;                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,   \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/*
 * Note: This is a simplified example that demonstrates the Device API concepts
 * without requiring the full NCCL device headers. For a complete implementation
 * with ncclDevComm and LSA barriers, see examples/06_device_api/.
 * 
 * This example uses a host-side pattern that mimics what the device API enables:
 * - Multiple independent operations that could be fused in a device kernel
 * - Direct peer memory access patterns
 * - Synchronization patterns similar to LSA barriers
 */

// Simple reduction kernel that reads from multiple source buffers
__global__ void simpleReduceKernel(float** src_ptrs, float* dst, int num_srcs, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        for (int s = 0; s < num_srcs; s++) {
            sum += src_ptrs[s][idx];
        }
        dst[idx] = sum;
    }
}

int main(int argc, char* argv[]) {
    printf("Starting Device API Simple example\n");
    printf("Note: This simplified example demonstrates Device API concepts.\n");
    printf("      See examples/06_device_api/ for full LSA barrier examples.\n\n");

    int num_gpus;
    CUDACHECK(cudaGetDeviceCount(&num_gpus));
    printf("Using %d GPUs\n", num_gpus);

    if (num_gpus < 2) {
        printf("This example requires at least 2 GPUs\n");
        return 1;
    }

    // Check P2P access capability
    printf("Checking P2P access capability...\n");
    bool p2p_available = true;
    for (int i = 0; i < num_gpus && p2p_available; i++) {
        for (int j = 0; j < num_gpus && p2p_available; j++) {
            if (i != j) {
                int can_access;
                CUDACHECK(cudaDeviceCanAccessPeer(&can_access, i, j));
                if (!can_access) {
                    printf("Warning: P2P access not available between GPU %d and %d\n", i, j);
                    p2p_available = false;
                }
            }
        }
    }

    // Enable P2P if available
    if (p2p_available) {
        printf("Enabling P2P access between all GPUs...\n");
        for (int i = 0; i < num_gpus; i++) {
            CUDACHECK(cudaSetDevice(i));
            for (int j = 0; j < num_gpus; j++) {
                if (i != j) {
                    CUDACHECK(cudaDeviceEnablePeerAccess(j, 0));
                }
            }
        }
    }

    // Allocate arrays
    ncclComm_t* comms = (ncclComm_t*)malloc(num_gpus * sizeof(ncclComm_t));
    cudaStream_t* streams = (cudaStream_t*)malloc(num_gpus * sizeof(cudaStream_t));
    float** d_send = (float**)malloc(num_gpus * sizeof(float*));
    float** d_recv = (float**)malloc(num_gpus * sizeof(float*));

    // Initialize NCCL
    printf("Initializing NCCL communicators...\n");
    NCCLCHECK(ncclCommInitAll(comms, num_gpus, NULL));

    const size_t size = 1024 * 1024; // 1M floats
    const size_t size_bytes = size * sizeof(float);

    // Setup each GPU
    for (int i = 0; i < num_gpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamCreate(&streams[i]));
        CUDACHECK(cudaMalloc((void**)&d_send[i], size_bytes));
        CUDACHECK(cudaMalloc((void**)&d_recv[i], size_bytes));

        // Initialize send buffer with rank value
        float* h_data = (float*)malloc(size_bytes);
        for (size_t j = 0; j < size; j++) {
            h_data[j] = (float)i;
        }
        CUDACHECK(cudaMemcpy(d_send[i], h_data, size_bytes, cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemset(d_recv[i], 0, size_bytes));
        free(h_data);
    }

    printf("Performing AllReduce (simulating device API pattern)...\n");
    
    // Use standard NCCL AllReduce - in a real Device API implementation,
    // this would be done directly in a CUDA kernel using ncclDevComm
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < num_gpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclAllReduce(d_send[i], d_recv[i], size, ncclFloat, ncclSum, 
                               comms[i], streams[i]));
    }
    NCCLCHECK(ncclGroupEnd());

    // If P2P is available, demonstrate direct peer memory access pattern
    if (p2p_available) {
        printf("Demonstrating direct peer memory access pattern...\n");
        
        // On GPU 0, we can directly read from other GPUs' memory
        CUDACHECK(cudaSetDevice(0));
        
        // Create array of pointers to all GPUs' receive buffers (accessible via P2P)
        float** h_src_ptrs = (float**)malloc(num_gpus * sizeof(float*));
        for (int i = 0; i < num_gpus; i++) {
            h_src_ptrs[i] = d_recv[i]; // These are accessible from GPU 0 via P2P
        }
        
        float** d_src_ptrs;
        CUDACHECK(cudaMalloc((void**)&d_src_ptrs, num_gpus * sizeof(float*)));
        CUDACHECK(cudaMemcpy(d_src_ptrs, h_src_ptrs, num_gpus * sizeof(float*), 
                            cudaMemcpyHostToDevice));
        
        // Allocate result buffer
        float* d_reduced;
        CUDACHECK(cudaMalloc((void**)&d_reduced, size_bytes));
        
        // Launch kernel that directly accesses peer memory
        const int threadsPerBlock = 256;
        const int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
        simpleReduceKernel<<<blocks, threadsPerBlock, 0, streams[0]>>>(
            d_src_ptrs, d_reduced, num_gpus, size);
        
        CUDACHECK(cudaStreamSynchronize(streams[0]));
        
        // Verify the reduction result
        float* h_reduced = (float*)malloc(size_bytes);
        CUDACHECK(cudaMemcpy(h_reduced, d_reduced, size_bytes, cudaMemcpyDeviceToHost));
        
        // Each recv buffer has the AllReduce result (sum of 0+1+2+...+(n-1))
        // Summing n copies gives n * sum
        float allreduce_sum = (float)(num_gpus * (num_gpus - 1) / 2);
        float expected = allreduce_sum * num_gpus;
        
        bool passed = true;
        for (size_t j = 0; j < size && j < 10; j++) {
            if (fabs(h_reduced[j] - expected) > 1e-5f) {
                printf("Mismatch at %zu: expected %f, got %f\n", j, expected, h_reduced[j]);
                passed = false;
                break;
            }
        }
        printf("Direct peer access reduction: %s\n", passed ? "PASSED" : "FAILED");
        
        free(h_reduced);
        free(h_src_ptrs);
        CUDACHECK(cudaFree(d_src_ptrs));
        CUDACHECK(cudaFree(d_reduced));
    }

    // Synchronize all streams
    for (int i = 0; i < num_gpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }

    // Verify AllReduce results
    printf("Verifying AllReduce results...\n");
    CUDACHECK(cudaSetDevice(0));
    float* h_result = (float*)malloc(size_bytes);
    CUDACHECK(cudaMemcpy(h_result, d_recv[0], size_bytes, cudaMemcpyDeviceToHost));

    float expected_sum = (float)(num_gpus * (num_gpus - 1) / 2);
    bool passed = true;
    for (size_t j = 0; j < size && j < 10; j++) {
        if (fabs(h_result[j] - expected_sum) > 1e-5f) {
            passed = false;
            break;
        }
    }
    printf("AllReduce results: %s\n", passed ? "PASSED" : "FAILED");
    free(h_result);

    // Cleanup
    for (int i = 0; i < num_gpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(d_send[i]));
        CUDACHECK(cudaFree(d_recv[i]));
        CUDACHECK(cudaStreamDestroy(streams[i]));
        NCCLCHECK(ncclCommDestroy(comms[i]));
        
        // Disable P2P
        if (p2p_available) {
            for (int j = 0; j < num_gpus; j++) {
                if (i != j) {
                    cudaDeviceDisablePeerAccess(j);
                }
            }
        }
    }

    free(comms);
    free(streams);
    free(d_send);
    free(d_recv);

    printf("Example completed successfully!\n");
    return 0;
}

