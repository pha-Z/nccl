/*
 * NCCL Advanced Example: Multiple Collectives in a Group
 * 
 * Demonstrates grouping multiple NCCL collectives (AllReduce, Broadcast,
 * AllGather) for fusion and concurrent execution.
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

int main(int argc, char* argv[]) {
    printf("Starting Multiple Collectives in Group example\n");

    int num_gpus;
    CUDACHECK(cudaGetDeviceCount(&num_gpus));
    printf("Using %d GPUs\n", num_gpus);

    // Allocate arrays
    ncclComm_t* comms = (ncclComm_t*)malloc(num_gpus * sizeof(ncclComm_t));
    cudaStream_t* streams = (cudaStream_t*)malloc(num_gpus * sizeof(cudaStream_t));
    
    // Three different buffers for three different collectives
    float** d_allreduce_buf = (float**)malloc(num_gpus * sizeof(float*));
    float** d_broadcast_buf = (float**)malloc(num_gpus * sizeof(float*));
    float** d_allgather_buf = (float**)malloc(num_gpus * sizeof(float*));
    float** d_allgather_result = (float**)malloc(num_gpus * sizeof(float*));

    // Initialize NCCL
    printf("Initializing NCCL communicators...\n");
    NCCLCHECK(ncclCommInitAll(comms, num_gpus, NULL));

    const size_t allreduce_size = 16 * 1024 * 1024; // 16M floats
    const size_t broadcast_size = 8 * 1024 * 1024;  // 8M floats
    const size_t allgather_size = 4 * 1024 * 1024;  // 4M floats per rank
    const size_t allgather_result_size = allgather_size * num_gpus; // Total gathered size

    // Setup each GPU
    for (int i = 0; i < num_gpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamCreate(&streams[i]));

        // Allocate device memory
        CUDACHECK(cudaMalloc((void**)&d_allreduce_buf[i], allreduce_size * sizeof(float)));
        CUDACHECK(cudaMalloc((void**)&d_broadcast_buf[i], broadcast_size * sizeof(float)));
        CUDACHECK(cudaMalloc((void**)&d_allgather_buf[i], allgather_size * sizeof(float)));
        CUDACHECK(cudaMalloc((void**)&d_allgather_result[i], allgather_result_size * sizeof(float)));

        // Initialize buffers
        float* h_allreduce = (float*)malloc(allreduce_size * sizeof(float));
        float* h_broadcast = (float*)malloc(broadcast_size * sizeof(float));
        float* h_allgather = (float*)malloc(allgather_size * sizeof(float));
        
        for (size_t j = 0; j < allreduce_size; j++) {
            h_allreduce[j] = (float)i; // Each rank contributes its rank value
        }
        for (size_t j = 0; j < broadcast_size; j++) {
            h_broadcast[j] = (float)(i * 10); // Different pattern for broadcast
        }
        for (size_t j = 0; j < allgather_size; j++) {
            h_allgather[j] = (float)(i * 100 + j); // Unique pattern per rank
        }

        CUDACHECK(cudaMemcpy(d_allreduce_buf[i], h_allreduce, allreduce_size * sizeof(float), 
                            cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(d_broadcast_buf[i], h_broadcast, broadcast_size * sizeof(float), 
                            cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(d_allgather_buf[i], h_allgather, allgather_size * sizeof(float), 
                            cudaMemcpyHostToDevice));

        free(h_allreduce);
        free(h_broadcast);
        free(h_allgather);
    }

    printf("Initializing buffers for 3 different collectives...\n");
    printf("Executing grouped collectives:\n");
    printf("  - AllReduce on buffer 1\n");
    printf("  - Broadcast on buffer 2\n");
    printf("  - AllGather on buffer 3\n");

    // Execute multiple collectives in a single group
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < num_gpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        
        // AllReduce: sum all ranks' values
        NCCLCHECK(ncclAllReduce(d_allreduce_buf[i], d_allreduce_buf[i], allreduce_size, 
                               ncclFloat, ncclSum, comms[i], streams[i]));
        
        // Broadcast: rank 0 broadcasts to all
        NCCLCHECK(ncclBroadcast(d_broadcast_buf[i], d_broadcast_buf[i], broadcast_size, 
                               ncclFloat, 0, comms[i], streams[i]));
        
        // AllGather: gather from all ranks
        NCCLCHECK(ncclAllGather(d_allgather_buf[i], d_allgather_result[i], allgather_size, 
                               ncclFloat, comms[i], streams[i]));
    }
    NCCLCHECK(ncclGroupEnd());

    // Synchronize all streams
    for (int i = 0; i < num_gpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }

    printf("Group execution completed\n");

    // Verify results
    printf("Verifying results...\n");
    
    // Verify AllReduce: should be sum of all ranks
    CUDACHECK(cudaSetDevice(0));
    float* h_allreduce_result = (float*)malloc(allreduce_size * sizeof(float));
    CUDACHECK(cudaMemcpy(h_allreduce_result, d_allreduce_buf[0], allreduce_size * sizeof(float), 
                        cudaMemcpyDeviceToHost));
    float expected_sum = (float)(num_gpus * (num_gpus - 1) / 2);
    bool allreduce_passed = true;
    for (size_t i = 0; i < allreduce_size && i < 10; i++) {
        if (fabs(h_allreduce_result[i] - expected_sum) > 1e-5f) {
            allreduce_passed = false;
            break;
        }
    }
    printf("  AllReduce: %s\n", allreduce_passed ? "PASSED" : "FAILED");
    free(h_allreduce_result);

    // Verify Broadcast: all ranks should have rank 0's value
    float* h_broadcast_result = (float*)malloc(broadcast_size * sizeof(float));
    CUDACHECK(cudaMemcpy(h_broadcast_result, d_broadcast_buf[0], broadcast_size * sizeof(float), 
                        cudaMemcpyDeviceToHost));
    float expected_broadcast = 0.0f; // Rank 0's value
    bool broadcast_passed = true;
    for (size_t i = 0; i < broadcast_size && i < 10; i++) {
        if (fabs(h_broadcast_result[i] - expected_broadcast) > 1e-5f) {
            broadcast_passed = false;
            break;
        }
    }
    printf("  Broadcast: %s\n", broadcast_passed ? "PASSED" : "FAILED");
    free(h_broadcast_result);

    // Verify AllGather: should contain data from all ranks in order
    float* h_allgather_result = (float*)malloc(allgather_result_size * sizeof(float));
    CUDACHECK(cudaMemcpy(h_allgather_result, d_allgather_result[0], 
                        allgather_result_size * sizeof(float), cudaMemcpyDeviceToHost));
    bool allgather_passed = true;
    for (int rank = 0; rank < num_gpus && allgather_passed; rank++) {
        for (size_t i = 0; i < allgather_size && i < 10; i++) {
            size_t idx = rank * allgather_size + i;
            float expected = (float)(rank * 100 + i);
            if (fabs(h_allgather_result[idx] - expected) > 1e-5f) {
                allgather_passed = false;
                break;
            }
        }
    }
    printf("  AllGather: %s\n", allgather_passed ? "PASSED" : "FAILED");
    free(h_allgather_result);

    if (allreduce_passed && broadcast_passed && allgather_passed) {
        printf("All verifications passed!\n");
    } else {
        printf("Some verifications failed!\n");
    }

    // Cleanup
    for (int i = 0; i < num_gpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(d_allreduce_buf[i]));
        CUDACHECK(cudaFree(d_broadcast_buf[i]));
        CUDACHECK(cudaFree(d_allgather_buf[i]));
        CUDACHECK(cudaFree(d_allgather_result[i]));
        CUDACHECK(cudaStreamDestroy(streams[i]));
        NCCLCHECK(ncclCommDestroy(comms[i]));
    }

    free(comms);
    free(streams);
    free(d_allreduce_buf);
    free(d_broadcast_buf);
    free(d_allgather_buf);
    free(d_allgather_result);

    printf("Example completed successfully!\n");
    return 0;
}

