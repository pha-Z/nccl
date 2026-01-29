/*************************************************************************
 * NCCL Profiler Overhead Measurement - Communicator Operations (MPI Multi-Node)
 *
 * Measures the overhead of having the profiler plugin active by running
 * communicator init/destroy/split cycles across multiple nodes.
 ************************************************************************/

#include "cuda_runtime.h"
#include "mpi.h"
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

#define MPICHECK(cmd)                                                          \
  do {                                                                         \
    int err = cmd;                                                             \
    if (err != MPI_SUCCESS) {                                                  \
      fprintf(stderr, "MPI error %s:%d\n", __FILE__, __LINE__);                \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

static inline uint64_t get_time_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

int getLocalRank(MPI_Comm comm) {
  int world_size, world_rank;
  MPICHECK(MPI_Comm_size(comm, &world_size));
  MPICHECK(MPI_Comm_rank(comm, &world_rank));

  MPI_Comm node_comm;
  MPICHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &node_comm));

  int node_rank, node_size;
  MPICHECK(MPI_Comm_rank(node_comm, &node_rank));
  MPICHECK(MPI_Comm_size(node_comm, &node_size));

  MPI_Comm_free(&node_comm);
  return node_rank;
}

int main(int argc, char *argv[]) {
  const int num_iterations = 100000;
  const int warmup_iterations = 100;

  // Initialize MPI
  MPICHECK(MPI_Init(&argc, &argv));
  int mpi_rank, mpi_size;
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));

  if (mpi_size < 2) {
    if (mpi_rank == 0) {
      fprintf(stderr, "At least 2 MPI ranks required (found %d)\n", mpi_size);
    }
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  int local_rank = getLocalRank(MPI_COMM_WORLD);
  int num_gpus = 0;
  CUDACHECK(cudaGetDeviceCount(&num_gpus));
  if (num_gpus < 1) {
    if (mpi_rank == 0) {
      fprintf(stderr, "No CUDA devices found\n");
    }
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  // Set device based on local rank
  CUDACHECK(cudaSetDevice(local_rank % num_gpus));

  if (mpi_rank == 0) {
    printf("========================================\n");
    printf("NCCL Comm Ops Overhead Measurement (Multi-Node)\n");
    printf("========================================\n");
    printf("MPI ranks: %d\n", mpi_size);
    printf("GPUs per node: %d\n", num_gpus);
    printf("Iterations: %d\n", num_iterations);
    printf("Warmup iterations: %d\n", warmup_iterations);
    printf("\n");
  }

  // Warmup - init/destroy cycles
  if (mpi_rank == 0) {
    printf("Warming up (init/destroy cycles)...\n");
  }
  for (int iter = 0; iter < warmup_iterations; iter++) {
    ncclUniqueId id;
    if (mpi_rank == 0) {
      NCCLCHECK(ncclGetUniqueId(&id));
    }
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, mpi_size, id, mpi_rank));
    NCCLCHECK(ncclCommDestroy(comm));
  }

  // Synchronize all ranks before timing
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Test 1: Init/Destroy cycles
  if (mpi_rank == 0) {
    printf("Running init/destroy cycles...\n");
  }
  uint64_t start_time = get_time_ns();
  for (int iter = 0; iter < num_iterations; iter++) {
    ncclUniqueId id;
    if (mpi_rank == 0) {
      NCCLCHECK(ncclGetUniqueId(&id));
    }
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, mpi_size, id, mpi_rank));
    NCCLCHECK(ncclCommDestroy(comm));
  }
  uint64_t end_time = get_time_ns();
  uint64_t init_destroy_time_ns = end_time - start_time;
  double init_destroy_time_ms = init_destroy_time_ns / 1e6;
  double avg_init_destroy_us = init_destroy_time_ns / (double)num_iterations / 1000.0;

  // Test 2: CommSplit cycles
  if (mpi_rank == 0) {
    printf("Running comm split cycles...\n");
  }
  start_time = get_time_ns();
  for (int iter = 0; iter < num_iterations; iter++) {
    // Create parent comm
    ncclUniqueId id;
    if (mpi_rank == 0) {
      NCCLCHECK(ncclGetUniqueId(&id));
    }
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    ncclComm_t parent_comm;
    NCCLCHECK(ncclCommInitRank(&parent_comm, mpi_size, id, mpi_rank));

    // Split into two groups (even/odd ranks)
    ncclComm_t child_comm;
    int color = mpi_rank % 2;
    int key = mpi_rank;
    NCCLCHECK(ncclCommSplit(parent_comm, color, key, &child_comm, NULL));

    // Destroy child and parent comms
    NCCLCHECK(ncclCommDestroy(child_comm));
    NCCLCHECK(ncclCommDestroy(parent_comm));
  }
  end_time = get_time_ns();
  uint64_t split_time_ns = end_time - start_time;
  double split_time_ms = split_time_ns / 1e6;
  double avg_split_us = split_time_ns / (double)num_iterations / 1000.0;

  // Gather timing from all ranks (use max for worst-case)
  double max_init_destroy_ms = init_destroy_time_ms;
  double max_split_ms = split_time_ms;
  MPICHECK(MPI_Allreduce(&init_destroy_time_ms, &max_init_destroy_ms, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
  MPICHECK(MPI_Allreduce(&split_time_ms, &max_split_ms, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

  // Output results (only rank 0)
  if (mpi_rank == 0) {
    printf("\n");
    printf("{\n");
    printf("  \"experiment\": \"comm_ops_overhead_multinode\",\n");
    printf("  \"mpi_ranks\": %d,\n", mpi_size);
    printf("  \"gpus_per_node\": %d,\n", num_gpus);
    printf("  \"num_iterations\": %d,\n", num_iterations);
    printf("  \"init_destroy\": {\n");
    printf("    \"total_time_ms\": %.3f,\n", max_init_destroy_ms);
    printf("    \"avg_time_per_iteration_us\": %.3f,\n", avg_init_destroy_us);
    printf("    \"total_time_ns\": %lu\n", init_destroy_time_ns);
    printf("  },\n");
    printf("  \"comm_split\": {\n");
    printf("    \"total_time_ms\": %.3f,\n", max_split_ms);
    printf("    \"avg_time_per_iteration_us\": %.3f,\n", avg_split_us);
    printf("    \"total_time_ns\": %lu\n", split_time_ns);
    printf("  }\n");
    printf("}\n");
  }

  MPICHECK(MPI_Finalize());
  return 0;
}
