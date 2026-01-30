/*************************************************************************
 * NCCL Profiler Overhead Measurement - Collectives (MPI Multi-Node)
 *
 * Measures the overhead of having the profiler plugin active by running
 * small collective operations with and without profiler across multiple nodes.
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
  const int num_iterations = 1000000;
  const int warmup_iterations = 100;
  const size_t count = 4;  // Very small buffer to emphasize call overhead

  // Initialize MPI
  MPICHECK(MPI_Init(&argc, &argv));
  int mpi_rank, mpi_size;
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));

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
    printf("NCCL Collectives Overhead Measurement (Multi-Node)\n");
    printf("========================================\n");
    printf("MPI ranks: %d\n", mpi_size);
    printf("GPUs per node: %d\n", num_gpus);
    printf("Iterations: %d\n", num_iterations);
    printf("Buffer size: %zu floats (%zu bytes)\n", count, count * sizeof(float));
    printf("Warmup iterations: %d\n", warmup_iterations);
    printf("\n");
  }

  // Initialize NCCL
  ncclUniqueId id;
  if (mpi_rank == 0) {
    NCCLCHECK(ncclGetUniqueId(&id));
  }
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, mpi_size, id, mpi_rank));

  // Allocate buffers
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));
  float *sendbuff, *recvbuff;
  CUDACHECK(cudaMalloc((void **)&sendbuff, count * sizeof(float)));
  CUDACHECK(cudaMalloc((void **)&recvbuff, count * sizeof(float)));
  CUDACHECK(cudaMemset(sendbuff, 0, count * sizeof(float)));

  // Warmup
  if (mpi_rank == 0) {
    printf("Warming up...\n");
  }
  for (int iter = 0; iter < warmup_iterations; iter++) {
    NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
  }

  // Synchronize all ranks before timing
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Timed run
  if (mpi_rank == 0) {
    printf("Running timed iterations...\n");
  }
  uint64_t start_time = get_time_ns();

  for (int iter = 0; iter < num_iterations; iter++) {
    NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
  }

  uint64_t end_time = get_time_ns();
  uint64_t total_time_ns = end_time - start_time;
  double total_time_ms = total_time_ns / 1e6;
  double avg_time_us = total_time_ns / (double)num_iterations / 1000.0;

  // Gather timing from all ranks (use max for worst-case)
  double max_time_ms = total_time_ms;
  MPICHECK(MPI_Allreduce(&total_time_ms, &max_time_ms, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

  // Output results (only rank 0)
  if (mpi_rank == 0) {
    printf("\n");
    printf("{\n");
    printf("  \"experiment\": \"collectives_overhead_multinode\",\n");
    printf("  \"mpi_ranks\": %d,\n", mpi_size);
    printf("  \"gpus_per_node\": %d,\n", num_gpus);
    printf("  \"num_iterations\": %d,\n", num_iterations);
    printf("  \"buffer_size_elements\": %zu,\n", count);
    printf("  \"total_time_ms\": %.3f,\n", max_time_ms);
    printf("  \"avg_time_per_iteration_us\": %.3f,\n", avg_time_us);
    printf("  \"total_time_ns\": %lu\n", total_time_ns);
    printf("}\n");
  }

  // Cleanup
  NCCLCHECK(ncclCommDestroy(comm));
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));
  CUDACHECK(cudaStreamDestroy(stream));

  MPICHECK(MPI_Finalize());
  return 0;
}
