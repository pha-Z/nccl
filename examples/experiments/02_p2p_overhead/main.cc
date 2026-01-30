/*************************************************************************
 * NCCL Profiler Overhead Measurement - Point-to-Point Operations
 *
 * Measures the overhead of having the profiler plugin active by running
 * small P2P operations with and without profiler.
 ************************************************************************/

#include "cuda_runtime.h"
#include "nccl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

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
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr, "Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,   \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

static inline uint64_t get_time_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

int main(int argc, char *argv[]) {
  const int num_iterations = 1000000;
  const int warmup_iterations = 100;
  const size_t count = 4;  // Very small buffer to emphasize call overhead

  int num_gpus = 0;
  CUDACHECK(cudaGetDeviceCount(&num_gpus));
  if (num_gpus < 2) {
    fprintf(stderr, "At least 2 GPUs required for P2P operations (found %d)\n", num_gpus);
    return EXIT_FAILURE;
  }

  printf("========================================\n");
  printf("NCCL P2P Overhead Measurement\n");
  printf("========================================\n");
  printf("GPUs: %d\n", num_gpus);
  printf("Iterations: %d\n", num_iterations);
  printf("Buffer size: %zu floats (%zu bytes)\n", count, count * sizeof(float));
  printf("Warmup iterations: %d\n", warmup_iterations);
  printf("\n");

  // Allocate resources
  ncclComm_t *comms = (ncclComm_t *)malloc(num_gpus * sizeof(ncclComm_t));
  cudaStream_t *streams = (cudaStream_t *)malloc(num_gpus * sizeof(cudaStream_t));
  float **sendbuff = (float **)malloc(num_gpus * sizeof(float *));
  float **recvbuff = (float **)malloc(num_gpus * sizeof(float *));

  // Initialize NCCL
  NCCLCHECK(ncclCommInitAll(comms, num_gpus, NULL));

  // Create streams and allocate buffers
  for (int i = 0; i < num_gpus; i++) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamCreate(&streams[i]));
    CUDACHECK(cudaMalloc((void **)&sendbuff[i], count * sizeof(float)));
    CUDACHECK(cudaMalloc((void **)&recvbuff[i], count * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff[i], 0, count * sizeof(float)));
  }

  // Warmup
  printf("Warming up...\n");
  for (int iter = 0; iter < warmup_iterations; iter++) {
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < num_gpus; i++) {
      int next = (i + 1) % num_gpus;
      int prev = (i - 1 + num_gpus) % num_gpus;
      NCCLCHECK(ncclSend(sendbuff[i], count, ncclFloat, next, comms[i], streams[i]));
      NCCLCHECK(ncclRecv(recvbuff[i], count, ncclFloat, prev, comms[i], streams[i]));
    }
    NCCLCHECK(ncclGroupEnd());
    for (int i = 0; i < num_gpus; i++) {
      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaStreamSynchronize(streams[i]));
    }
  }

  // Timed run
  printf("Running timed iterations...\n");
  uint64_t start_time = get_time_ns();

  for (int iter = 0; iter < num_iterations; iter++) {
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < num_gpus; i++) {
      int next = (i + 1) % num_gpus;
      int prev = (i - 1 + num_gpus) % num_gpus;
      NCCLCHECK(ncclSend(sendbuff[i], count, ncclFloat, next, comms[i], streams[i]));
      NCCLCHECK(ncclRecv(recvbuff[i], count, ncclFloat, prev, comms[i], streams[i]));
    }
    NCCLCHECK(ncclGroupEnd());
    for (int i = 0; i < num_gpus; i++) {
      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaStreamSynchronize(streams[i]));
    }
  }

  uint64_t end_time = get_time_ns();
  uint64_t total_time_ns = end_time - start_time;
  double total_time_ms = total_time_ns / 1e6;
  double avg_time_us = total_time_ns / (double)num_iterations / 1000.0;

  // Output results in JSON format
  printf("\n");
  printf("{\n");
  printf("  \"experiment\": \"p2p_overhead\",\n");
  printf("  \"num_gpus\": %d,\n", num_gpus);
  printf("  \"num_iterations\": %d,\n", num_iterations);
  printf("  \"buffer_size_elements\": %zu,\n", count);
  printf("  \"total_time_ms\": %.3f,\n", total_time_ms);
  printf("  \"avg_time_per_iteration_us\": %.3f,\n", avg_time_us);
  printf("  \"total_time_ns\": %lu\n", total_time_ns);
  printf("}\n");

  // Cleanup
  for (int i = 0; i < num_gpus; i++) {
    NCCLCHECK(ncclCommDestroy(comms[i]));
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
    CUDACHECK(cudaStreamDestroy(streams[i]));
  }

  free(comms);
  free(streams);
  free(sendbuff);
  free(recvbuff);

  return 0;
}
