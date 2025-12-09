/*
 * NCCL Advanced Example: CUDA Graphs with NCCL
 * 
 * Demonstrates capturing computational kernels and NCCL collectives
 * into a CUDA graph for reduced launch overhead.
 */

#include "cuda_runtime.h"
#include "nccl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

// Simple vector addition kernel
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Post-processing kernel (multiply by scalar)
__global__ void vectorScale(float* data, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

int main(int argc, char* argv[]) {
    printf("Starting CUDA Graphs with NCCL example\n");

    // Get number of GPUs
    int num_gpus;
    CUDACHECK(cudaGetDeviceCount(&num_gpus));
    printf("Using %d GPUs\n", num_gpus);

    // Allocate arrays for communicators and streams
    ncclComm_t* comms = (ncclComm_t*)malloc(num_gpus * sizeof(ncclComm_t));
    cudaStream_t* streams = (cudaStream_t*)malloc(num_gpus * sizeof(cudaStream_t));
    float** d_a = (float**)malloc(num_gpus * sizeof(float*));
    float** d_b = (float**)malloc(num_gpus * sizeof(float*));
    float** d_c = (float**)malloc(num_gpus * sizeof(float*));
    float** d_result = (float**)malloc(num_gpus * sizeof(float*));

    // Initialize NCCL
    printf("Initializing NCCL communicators...\n");
    NCCLCHECK(ncclCommInitAll(comms, num_gpus, NULL));

    const size_t size = 32 * 1024 * 1024; // 32M floats
    const size_t size_bytes = size * sizeof(float);
    const int num_iterations = 10;

    // Setup each GPU
    for (int i = 0; i < num_gpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamCreate(&streams[i]));

        // Allocate device memory
        CUDACHECK(cudaMalloc((void**)&d_a[i], size_bytes));
        CUDACHECK(cudaMalloc((void**)&d_b[i], size_bytes));
        CUDACHECK(cudaMalloc((void**)&d_c[i], size_bytes));
        CUDACHECK(cudaMalloc((void**)&d_result[i], size_bytes));

        // Initialize data
        float* h_a = (float*)malloc(size_bytes);
        float* h_b = (float*)malloc(size_bytes);
        for (size_t j = 0; j < size; j++) {
            h_a[j] = (float)(i * size + j); // Unique value per GPU
            h_b[j] = 1.0f;
        }
        CUDACHECK(cudaMemcpy(d_a[i], h_a, size_bytes, cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(d_b[i], h_b, size_bytes, cudaMemcpyHostToDevice));
        free(h_a);
        free(h_b);
    }

    // Capture CUDA graph
    printf("Capturing CUDA graph with computation and NCCL AllReduce...\n");
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    // Use GPU 0's stream for capture (all GPUs will use their own streams in the graph)
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal));

    // Launch kernels and NCCL operations to be captured
    const int threadsPerBlock = 256;
    const int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Group all operations
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < num_gpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        
        // Computational kernel: vector addition
        vectorAdd<<<blocks, threadsPerBlock, 0, streams[i]>>>(d_a[i], d_b[i], d_c[i], size);
        
        // NCCL AllReduce
        NCCLCHECK(ncclAllReduce(d_c[i], d_result[i], size, ncclFloat, ncclSum, comms[i], streams[i]));
        
        // Post-processing kernel
        vectorScale<<<blocks, threadsPerBlock, 0, streams[i]>>>(d_result[i], 0.5f, size);
    }
    NCCLCHECK(ncclGroupEnd());

    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaStreamEndCapture(streams[0], &graph));
    printf("Graph captured successfully\n");

    // Instantiate the graph
    printf("Instantiating graph...\n");
    CUDACHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    printf("Graph instantiated successfully\n");

    // Launch the graph multiple times
    printf("Launching graph %d times...\n", num_iterations);
    for (int iter = 0; iter < num_iterations; iter++) {
        CUDACHECK(cudaSetDevice(0));
        CUDACHECK(cudaGraphLaunch(graphExec, streams[0]));
        
        // Synchronize all streams
        for (int i = 0; i < num_gpus; i++) {
            CUDACHECK(cudaSetDevice(i));
            CUDACHECK(cudaStreamSynchronize(streams[i]));
        }
        printf("  Iteration %d completed\n", iter);
    }

    printf("All iterations completed successfully\n");

    // Verify results (check GPU 0)
    printf("Verifying results...\n");
    CUDACHECK(cudaSetDevice(0));
    float* h_result = (float*)malloc(size_bytes);
    CUDACHECK(cudaMemcpy(h_result, d_result[0], size_bytes, cudaMemcpyDeviceToHost));

    // Expected: sum of all ranks (0+1+2+3) = 6, then scaled by 0.5 = 3.0
    // Plus the constant 1.0 from vector b
    float expected = (float)(num_gpus * (num_gpus - 1) / 2) * 0.5f + 1.0f;
    bool passed = true;
    for (size_t i = 0; i < size && i < 10; i++) { // Check first 10 elements
        if (fabs(h_result[i] - expected) > 1e-5f) {
            printf("Mismatch at index %zu: expected %f, got %f\n", i, expected, h_result[i]);
            passed = false;
        }
    }
    if (passed) {
        printf("Results verified: PASSED\n");
    } else {
        printf("Results verified: FAILED\n");
    }
    free(h_result);

    // Cleanup
    CUDACHECK(cudaGraphExecDestroy(graphExec));
    CUDACHECK(cudaGraphDestroy(graph));

    for (int i = 0; i < num_gpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(d_a[i]));
        CUDACHECK(cudaFree(d_b[i]));
        CUDACHECK(cudaFree(d_c[i]));
        CUDACHECK(cudaFree(d_result[i]));
        CUDACHECK(cudaStreamDestroy(streams[i]));
        NCCLCHECK(ncclCommDestroy(comms[i]));
    }

    free(comms);
    free(streams);
    free(d_a);
    free(d_b);
    free(d_c);
    free(d_result);

    printf("Example completed successfully!\n");
    return 0;
}

