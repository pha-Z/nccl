/*************************************************************************
 * Minimal NCCL Profiler Plugin Example
 * 
 * This is a simplified profiler plugin that demonstrates the essential
 * structure needed to create your own plugin.
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stdint.h>

// Include the profiler API headers
// (Copy these from ext-profiler/example/nccl/)
#include "nccl/profiler_v5.h"
#include "nccl/err.h"
#include "nccl/types.h"

// Simple context structure to hold plugin state
typedef struct {
  FILE* logFile;
  uint64_t commId;
  int rank;
  int nranks;
  double startTime;
} MyContext;

// Simple event structure (minimal - stores just what we need)
typedef struct {
  uint64_t type;
  double startTime;
  const char* func;
  MyContext* ctx;  // Keep reference to context for logging
} MyEvent;

// Get current timestamp in microseconds
static double getTime() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

// ============================================================================
// Function 1: Initialize the profiler
// ============================================================================
ncclResult_t myInit(void** context, uint64_t commId, int* eActivationMask,
                    const char* commName, int nNodes, int nranks, int rank,
                    ncclDebugLogger_t logfn) {
  // Allocate context
  MyContext* ctx = calloc(1, sizeof(MyContext));
  if (!ctx) return ncclSystemError;
  
  ctx->commId = commId;
  ctx->rank = rank;
  ctx->nranks = nranks;
  ctx->startTime = getTime();
  
  // Open log file (one per rank)
  char filename[256];
  snprintf(filename, sizeof(filename), "my_profiler_rank%d_comm%lu.log", 
           rank, commId);
  ctx->logFile = fopen(filename, "w");
  if (!ctx->logFile) {
    free(ctx);
    return ncclSystemError;
  }
  
  fprintf(ctx->logFile, "=== Profiler initialized ===\n");
  fprintf(ctx->logFile, "Rank: %d/%d, CommId: %lu, CommName: %s\n", 
           rank, nranks, commId, commName ? commName : "(null)");
  
  // Optionally set activation mask (or let NCCL use defaults)
  // *eActivationMask = ncclProfileColl | ncclProfileP2p;
  
  *context = ctx;
  return ncclSuccess;
}

// ============================================================================
// Function 2: Start tracking an event
// ============================================================================
ncclResult_t myStartEvent(void* context, void** eHandle, 
                          ncclProfilerEventDescr_v5_t* eDescr) {
  MyContext* ctx = (MyContext*)context;
  MyEvent* event = NULL;
  
  // Allocate event
  event = calloc(1, sizeof(MyEvent));
  if (!event) {
    *eHandle = NULL;
    return ncclSuccess;  // Fail gracefully
  }
  
  event->type = eDescr->type;
  event->ctx = ctx;
  event->startTime = getTime();
  
  // Extract information based on event type
  switch (eDescr->type) {
    case ncclProfileGroupApi:
      event->func = "GroupAPI";
      fprintf(ctx->logFile, "[%.3f] START GroupAPI (depth=%d)\n",
              (event->startTime - ctx->startTime) / 1000.0,
              eDescr->groupApi.groupDepth);
      break;
      
    case ncclProfileCollApi:
      event->func = eDescr->collApi.func;
      fprintf(ctx->logFile, "[%.3f] START Collective: %s(count=%zu, dtype=%s, root=%d)\n",
              (event->startTime - ctx->startTime) / 1000.0,
              eDescr->collApi.func,
              eDescr->collApi.count,
              eDescr->collApi.datatype,
              eDescr->collApi.root);
      break;
      
    case ncclProfileP2pApi:
      event->func = eDescr->p2pApi.func;
      fprintf(ctx->logFile, "[%.3f] START P2P: %s(count=%zu, dtype=%s)\n",
              (event->startTime - ctx->startTime) / 1000.0,
              eDescr->p2pApi.func,
              eDescr->p2pApi.count,
              eDescr->p2pApi.datatype);
      break;
      
    case ncclProfileColl:
      event->func = eDescr->coll.func;
      fprintf(ctx->logFile, "[%.3f] START Coll Exec: %s(seq=%lu, algo=%s, proto=%s, channels=%d)\n",
              (event->startTime - ctx->startTime) / 1000.0,
              eDescr->coll.func,
              eDescr->coll.seqNumber,
              eDescr->coll.algo,
              eDescr->coll.proto,
              eDescr->coll.nChannels);
      break;
      
    case ncclProfileP2p:
      event->func = eDescr->p2p.func;
      fprintf(ctx->logFile, "[%.3f] START P2P Exec: %s(peer=%d, channels=%d)\n",
              (event->startTime - ctx->startTime) / 1000.0,
              eDescr->p2p.func,
              eDescr->p2p.peer,
              eDescr->p2p.nChannels);
      break;
      
    case ncclProfileProxyOp:
      fprintf(ctx->logFile, "[%.3f] START ProxyOp: channel=%d, peer=%d, steps=%d, send=%d\n",
              (event->startTime - ctx->startTime) / 1000.0,
              eDescr->proxyOp.channelId,
              eDescr->proxyOp.peer,
              eDescr->proxyOp.nSteps,
              eDescr->proxyOp.isSend);
      event->func = "ProxyOp";
      break;
      
    case ncclProfileKernelCh:
      fprintf(ctx->logFile, "[%.3f] START KernelCh: channel=%d\n",
              (event->startTime - ctx->startTime) / 1000.0,
              eDescr->kernelCh.channelId);
      event->func = "KernelCh";
      break;
      
    default:
      event->func = "Unknown";
      break;
  }
  
  *eHandle = event;
  return ncclSuccess;
}

// ============================================================================
// Function 3: Stop tracking an event
// ============================================================================
ncclResult_t myStopEvent(void* eHandle) {
  if (!eHandle) return ncclSuccess;
  
  MyEvent* event = (MyEvent*)eHandle;
  MyContext* ctx = event->ctx;
  double endTime = getTime();
  double duration = (endTime - event->startTime) / 1000.0;  // Convert to ms
  
  fprintf(ctx->logFile, "[%.3f] STOP %s (duration=%.3f ms)\n",
          (endTime - ctx->startTime) / 1000.0,
          event->func,
          duration);
  
  // Free the event
  free(event);
  return ncclSuccess;
}

// ============================================================================
// Function 4: Record event state transitions
// ============================================================================
ncclResult_t myRecordEventState(void* eHandle, 
                                 ncclProfilerEventState_v5_t eState,
                                 ncclProfilerEventStateArgs_v5_t* eStateArgs) {
  if (!eHandle) return ncclSuccess;
  
  MyEvent* event = (MyEvent*)eHandle;
  MyContext* ctx = event->ctx;
  double currentTime = getTime();
  
  // Log state transitions (simplified)
  const char* stateName = "Unknown";
  switch (eState) {
    case ncclProfilerProxyStepSendGPUWait:
      stateName = "SendGPUWait";
      break;
    case ncclProfilerProxyStepSendWait:
      stateName = "SendWait";
      break;
    case ncclProfilerProxyStepRecvWait:
      stateName = "RecvWait";
      break;
    case ncclProfilerProxyStepRecvFlushWait:
      stateName = "RecvFlushWait";
      break;
    case ncclProfilerProxyStepRecvGPUWait:
      stateName = "RecvGPUWait";
      break;
    case ncclProfilerProxyCtrlIdle:
      stateName = "ProxyCtrlIdle";
      break;
    case ncclProfilerProxyCtrlActive:
      stateName = "ProxyCtrlActive";
      break;
    default:
      break;
  }
  
  fprintf(ctx->logFile, "[%.3f] STATE %s -> %s\n",
          (currentTime - ctx->startTime) / 1000.0,
          event->func,
          stateName);
  
  return ncclSuccess;
}

// ============================================================================
// Function 5: Finalize and cleanup
// ============================================================================
ncclResult_t myFinalize(void* context) {
  MyContext* ctx = (MyContext*)context;
  
  if (ctx->logFile) {
    fprintf(ctx->logFile, "=== Profiler finalized ===\n");
    fclose(ctx->logFile);
  }
  
  free(ctx);
  return ncclSuccess;
}

// ============================================================================
// EXPORT: This is what NCCL looks for!
// ============================================================================
ncclProfiler_v5_t ncclProfiler_v5 = {
  "MyMinimalProfiler",           // Plugin name
  myInit,                        // Initialize function
  myStartEvent,                  // Start event function
  myStopEvent,                   // Stop event function
  myRecordEventState,            // Record state function
  myFinalize,                    // Finalize function
};

