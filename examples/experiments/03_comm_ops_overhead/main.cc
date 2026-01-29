/*************************************************************************
 * NCCL Profiler Overhead Measurement - Communicator Operations
 *
 * Measures the overhead of having the profiler plugin active by running
 * communicator init/destroy/split/shrink cycles.
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
  const int num_iterations = 100000;
  const int warmup_iterations = 100;

  int num_gpus = 0;
  CUDACHECK(cudaGetDeviceCount(&num_gpus));
  if (num_gpus < 2) {
    fprintf(stderr, "At least 2 GPUs required (found %d)\n", num_gpus);
    return EXIT_FAILURE;
  }

  printf("========================================\n");
  printf("NCCL Comm Ops Overhead Measurement\n");
  printf("========================================\n");
  printf("GPUs: %d\n", num_gpus);
  printf("Iterations: %d\n", num_iterations);
  printf("Warmup iterations: %d\n", warmup_iterations);
  printf("\n");

  // Allocate resources
  ncclComm_t *parent_comms = (ncclComm_t *)malloc(num_gpus * sizeof(ncclComm_t));
  ncclComm_t *child_comms = (ncclComm_t *)malloc(num_gpus * sizeof(ncclComm_t));
  int *devices = (int *)malloc(num_gpus * sizeof(int));
  for (int i = 0; i < num_gpus; i++) {
    devices[i] = i;
  }

  // Warmup - init/destroy cycles
  printf("Warming up (init/destroy cycles)...\n");
  for (int iter = 0; iter < warmup_iterations; iter++) {
    NCCLCHECK(ncclCommInitAll(parent_comms, num_gpus, devices));
    for (int i = 0; i < num_gpus; i++) {
      NCCLCHECK(ncclCommDestroy(parent_comms[i]));
    }
  }

  // Test 1: Init/Destroy cycles
  printf("Running init/destroy cycles...\n");
  uint64_t start_time = get_time_ns();
  for (int iter = 0; iter < num_iterations; iter++) {
    NCCLCHECK(ncclCommInitAll(parent_comms, num_gpus, devices));
    for (int i = 0; i < num_gpus; i++) {
      NCCLCHECK(ncclCommDestroy(parent_comms[i]));
    }
  }
  uint64_t end_time = get_time_ns();
  uint64_t init_destroy_time_ns = end_time - start_time;
  double init_destroy_time_ms = init_destroy_time_ns / 1e6;
  double avg_init_destroy_us = init_destroy_time_ns / (double)num_iterations / 1000.0;

  // Test 2: CommSplit cycles
  printf("Running comm split cycles...\n");
  start_time = get_time_ns();
  for (int iter = 0; iter < num_iterations; iter++) {
    // Create parent comm
    NCCLCHECK(ncclCommInitAll(parent_comms, num_gpus, devices));
    
    // Split into two groups (even/odd ranks)
    for (int i = 0; i < num_gpus; i++) {
      int color = i % 2;
      int key = i;
      NCCLCHECK(ncclCommSplit(parent_comms[i], color, key, &child_comms[i], NULL));
    }
    
    // Destroy child comms
    for (int i = 0; i < num_gpus; i++) {
      NCCLCHECK(ncclCommDestroy(child_comms[i]));
    }
    
    // Destroy parent comms
    for (int i = 0; i < num_gpus; i++) {
      NCCLCHECK(ncclCommDestroy(parent_comms[i]));
    }
  }
  end_time = get_time_ns();
  uint64_t split_time_ns = end_time - start_time;
  double split_time_ms = split_time_ns / 1e6;
  double avg_split_us = split_time_ns / (double)num_iterations / 1000.0;

  // Output results in JSON format
  printf("\n");
  printf("{\n");
  printf("  \"experiment\": \"comm_ops_overhead\",\n");
  printf("  \"num_gpus\": %d,\n", num_gpus);
  printf("  \"num_iterations\": %d,\n", num_iterations);
  printf("  \"init_destroy\": {\n");
  printf("    \"total_time_ms\": %.3f,\n", init_destroy_time_ms);
  printf("    \"avg_time_per_iteration_us\": %.3f,\n", avg_init_destroy_us);
  printf("    \"total_time_ns\": %lu\n", init_destroy_time_ns);
  printf("  },\n");
  printf("  \"comm_split\": {\n");
  printf("    \"total_time_ms\": %.3f,\n", split_time_ms);
  printf("    \"avg_time_per_iteration_us\": %.3f,\n", avg_split_us);
  printf("    \"total_time_ns\": %lu\n", split_time_ns);
  printf("  }\n");
  printf("}\n");

  // Cleanup
  free(parent_comms);
  free(child_comms);
  free(devices);

  return 0;
}
