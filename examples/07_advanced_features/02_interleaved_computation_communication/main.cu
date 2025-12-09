/*
 * NCCL Advanced Example: Interleaved Computation and Communication
 * 
 * Demonstrates pipeline parallelism by overlapping computation and
 * communication stages using multiple CUDA streams.
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

// Pre-processing kernel: multiply by rank
__global__ void preprocessKernel(float* data, float rank_scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= rank_scale;
    }
}

// Post-processing kernel: add constant
__global__ void postprocessKernel(float* data, float constant, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += constant;
    }
}

int main(int argc, char* argv[]) {
    printf("Starting Interleaved Computation-Communication example\n");

    int num_gpus;
    CUDACHECK(cudaGetDeviceCount(&num_gpus));
    printf("Using %d GPUs\n", num_gpus);

    // Allocate arrays
    ncclComm_t* comms = (ncclComm_t*)malloc(num_gpus * sizeof(ncclComm_t));
    cudaStream_t* compute_streams = (cudaStream_t*)malloc(num_gpus * sizeof(cudaStream_t));
    cudaStream_t* comm_streams = (cudaStream_t*)malloc(num_gpus * sizeof(cudaStream_t));
    cudaEvent_t* compute_events = (cudaEvent_t*)malloc(num_gpus * sizeof(cudaEvent_t));
    
    float** d_data = (float**)malloc(num_gpus * sizeof(float*));
    float** d_temp = (float**)malloc(num_gpus * sizeof(float*));

    // Initialize NCCL
    printf("Initializing NCCL communicators...\n");
    NCCLCHECK(ncclCommInitAll(comms, num_gpus, NULL));

    const size_t size = 64 * 1024 * 1024; // 64M floats
    const size_t size_bytes = size * sizeof(float);
    const int num_iterations = 5;
    const int threadsPerBlock = 256;
    const int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Setup each GPU
    for (int i = 0; i < num_gpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        
        // Create separate streams for computation and communication
        CUDACHECK(cudaStreamCreate(&compute_streams[i]));
        CUDACHECK(cudaStreamCreate(&comm_streams[i]));
        CUDACHECK(cudaEventCreate(&compute_events[i]));

        // Allocate device memory
        CUDACHECK(cudaMalloc((void**)&d_data[i], size_bytes));
        CUDACHECK(cudaMalloc((void**)&d_temp[i], size_bytes));

        // Initialize data
        float* h_data = (float*)malloc(size_bytes);
        for (size_t j = 0; j < size; j++) {
            h_data[j] = 1.0f;
        }
        CUDACHECK(cudaMemcpy(d_data[i], h_data, size_bytes, cudaMemcpyHostToDevice));
        free(h_data);
    }

    printf("Initializing pipeline with 3 stages per iteration\n");
    printf("Running %d pipeline iterations...\n", num_iterations);

    // Pipeline: Interleave computation and communication
    for (int iter = 0; iter < num_iterations; iter++) {
        printf("  Iteration %d: ", iter);
        
        // Stage 1: Pre-processing computation (on compute stream)
        printf("Stage 1 (compute) -> ");
        NCCLCHECK(ncclGroupStart());
        for (int i = 0; i < num_gpus; i++) {
            CUDACHECK(cudaSetDevice(i));
            preprocessKernel<<<blocks, threadsPerBlock, 0, compute_streams[i]>>>(
                d_data[i], (float)i, size);
        }
        NCCLCHECK(ncclGroupEnd());

        // Record event when compute is done
        for (int i = 0; i < num_gpus; i++) {
            CUDACHECK(cudaSetDevice(i));
            CUDACHECK(cudaEventRecord(compute_events[i], compute_streams[i]));
        }

        // Stage 2: Communication (on comm stream, waits for compute)
        printf("Stage 2 (comm) -> ");
        for (int i = 0; i < num_gpus; i++) {
            CUDACHECK(cudaSetDevice(i));
            // Wait for compute to finish before starting communication
            CUDACHECK(cudaStreamWaitEvent(comm_streams[i], compute_events[i], 0));
        }

        NCCLCHECK(ncclGroupStart());
        for (int i = 0; i < num_gpus; i++) {
            CUDACHECK(cudaSetDevice(i));
            // AllReduce on communication stream
            NCCLCHECK(ncclAllReduce(d_data[i], d_temp[i], size, ncclFloat, ncclSum, 
                                   comms[i], comm_streams[i]));
        }
        NCCLCHECK(ncclGroupEnd());

        // Stage 3: Post-processing (on compute stream, can overlap with next iter's stage 1)
        printf("Stage 3 (compute)\n");
        for (int i = 0; i < num_gpus; i++) {
            CUDACHECK(cudaSetDevice(i));
            // Copy result back to data buffer
            CUDACHECK(cudaMemcpyAsync(d_data[i], d_temp[i], size_bytes, 
                                      cudaMemcpyDeviceToDevice, comm_streams[i]));
            // Post-process on compute stream (waits for copy)
            CUDACHECK(cudaStreamWaitEvent(compute_streams[i], compute_events[i], 0));
            postprocessKernel<<<blocks, threadsPerBlock, 0, compute_streams[i]>>>(
                d_data[i], 10.0f, size);
        }

        // Synchronize all streams at end of iteration
        for (int i = 0; i < num_gpus; i++) {
            CUDACHECK(cudaSetDevice(i));
            CUDACHECK(cudaStreamSynchronize(compute_streams[i]));
            CUDACHECK(cudaStreamSynchronize(comm_streams[i]));
        }
    }

    printf("Pipeline completed successfully\n");

    // Verify results
    printf("Verifying results...\n");
    CUDACHECK(cudaSetDevice(0));
    float* h_result = (float*)malloc(size_bytes);
    CUDACHECK(cudaMemcpy(h_result, d_data[0], size_bytes, cudaMemcpyDeviceToHost));

    // Expected: preprocess multiplies by rank, AllReduce sums all ranks,
    // postprocess adds 10.0
    // For rank 0: 1.0 * 0 = 0, sum = 0+1+2+3 = 6, +10 = 16
    float expected = (float)(num_gpus * (num_gpus - 1) / 2) + 10.0f;
    bool passed = true;
    for (size_t i = 0; i < size && i < 10; i++) {
        if (fabs(h_result[i] - expected) > 1e-5f) {
            printf("Mismatch at index %zu: expected %f, got %f\n", i, expected, h_result[i]);
            passed = false;
            break;
        }
    }
    if (passed) {
        printf("Results verified: PASSED\n");
    } else {
        printf("Results verified: FAILED\n");
    }
    free(h_result);

    // Cleanup
    for (int i = 0; i < num_gpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamDestroy(compute_streams[i]));
        CUDACHECK(cudaStreamDestroy(comm_streams[i]));
        CUDACHECK(cudaEventDestroy(compute_events[i]));
        CUDACHECK(cudaFree(d_data[i]));
        CUDACHECK(cudaFree(d_temp[i]));
        NCCLCHECK(ncclCommDestroy(comms[i]));
    }

    free(comms);
    free(compute_streams);
    free(comm_streams);
    free(compute_events);
    free(d_data);
    free(d_temp);

    printf("Example completed successfully!\n");
    return 0;
}

