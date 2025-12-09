/*
 * NCCL Advanced Example: Multi-Stream Independent Operations
 * 
 * Demonstrates using multiple independent CUDA streams for concurrent
 * NCCL operations on different data buffers.
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

#define NUM_STREAMS 3

int main(int argc, char* argv[]) {
    printf("Starting Multi-Stream Independent Operations example\n");

    int num_gpus;
    CUDACHECK(cudaGetDeviceCount(&num_gpus));
    printf("Using %d GPUs with %d independent streams each\n", num_gpus, NUM_STREAMS);

    // Allocate arrays
    ncclComm_t* comms = (ncclComm_t*)malloc(num_gpus * sizeof(ncclComm_t));
    cudaStream_t** streams = (cudaStream_t**)malloc(num_gpus * sizeof(cudaStream_t*));
    float*** d_buffers = (float***)malloc(num_gpus * sizeof(float**));

    // Initialize NCCL
    printf("Initializing NCCL communicators...\n");
    NCCLCHECK(ncclCommInitAll(comms, num_gpus, NULL));

    const size_t size = 16 * 1024 * 1024; // 16M floats per buffer
    const size_t size_bytes = size * sizeof(float);

    // Setup each GPU with multiple streams
    printf("Initializing buffers for each stream...\n");
    for (int i = 0; i < num_gpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        
        streams[i] = (cudaStream_t*)malloc(NUM_STREAMS * sizeof(cudaStream_t));
        d_buffers[i] = (float**)malloc(NUM_STREAMS * sizeof(float*));

        for (int s = 0; s < NUM_STREAMS; s++) {
            CUDACHECK(cudaStreamCreate(&streams[i][s]));
            CUDACHECK(cudaMalloc((void**)&d_buffers[i][s], size_bytes));

            // Initialize each buffer with different values
            float* h_buf = (float*)malloc(size_bytes);
            for (size_t j = 0; j < size; j++) {
                // Stream s, GPU i: value = (s+1) * (i+1)
                h_buf[j] = (float)((s + 1) * (i + 1));
            }
            CUDACHECK(cudaMemcpy(d_buffers[i][s], h_buf, size_bytes, cudaMemcpyHostToDevice));
            free(h_buf);
        }
    }

    printf("Launching independent AllReduce operations:\n");
    printf("  - Stream A: AllReduce on buffer A\n");
    printf("  - Stream B: AllReduce on buffer B\n");
    printf("  - Stream C: AllReduce on buffer C\n");

    // Launch independent AllReduce operations on different streams
    // Each stream operates on a different buffer - no dependencies
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < num_gpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        for (int s = 0; s < NUM_STREAMS; s++) {
            // Each stream does an independent AllReduce on its buffer
            NCCLCHECK(ncclAllReduce(d_buffers[i][s], d_buffers[i][s], size, 
                                   ncclFloat, ncclSum, comms[i], streams[i][s]));
        }
    }
    NCCLCHECK(ncclGroupEnd());

    printf("All operations launched\n");

    // Synchronize all streams
    printf("Synchronizing all streams...\n");
    for (int i = 0; i < num_gpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        for (int s = 0; s < NUM_STREAMS; s++) {
            CUDACHECK(cudaStreamSynchronize(streams[i][s]));
        }
    }

    // Verify results
    printf("Verifying results...\n");
    bool all_passed = true;
    const char* stream_names[] = {"A", "B", "C"};

    for (int s = 0; s < NUM_STREAMS; s++) {
        CUDACHECK(cudaSetDevice(0));
        float* h_result = (float*)malloc(size_bytes);
        CUDACHECK(cudaMemcpy(h_result, d_buffers[0][s], size_bytes, cudaMemcpyDeviceToHost));

        // Expected: sum of (s+1) * (i+1) for all GPUs i
        // = (s+1) * sum(i+1 for i in 0..num_gpus-1)
        // = (s+1) * (1 + 2 + ... + num_gpus)
        // = (s+1) * num_gpus * (num_gpus + 1) / 2
        float expected = (float)((s + 1) * num_gpus * (num_gpus + 1) / 2);
        
        bool passed = true;
        for (size_t j = 0; j < size && j < 10; j++) {
            if (fabs(h_result[j] - expected) > 1e-5f) {
                passed = false;
                break;
            }
        }
        printf("  Stream %s: %s\n", stream_names[s], passed ? "PASSED" : "FAILED");
        if (!passed) all_passed = false;
        free(h_result);
    }

    if (all_passed) {
        printf("All verifications passed!\n");
    } else {
        printf("Some verifications failed!\n");
    }

    // Cleanup
    for (int i = 0; i < num_gpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        for (int s = 0; s < NUM_STREAMS; s++) {
            CUDACHECK(cudaFree(d_buffers[i][s]));
            CUDACHECK(cudaStreamDestroy(streams[i][s]));
        }
        free(d_buffers[i]);
        free(streams[i]);
        NCCLCHECK(ncclCommDestroy(comms[i]));
    }

    free(comms);
    free(streams);
    free(d_buffers);

    printf("Example completed successfully!\n");
    return 0;
}

