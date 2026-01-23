/*************************************************************************
 Empty NCCL Profiler Plugin Example
 ************************************************************************/
#include <stdlib.h>

#include "nccl/common.h"
#include "nccl/err.h"
#include "nccl/profiler.h"

struct MyEvent {
  char dummy;
};

struct MyContext {
  char dummy;
};

ncclResult_t myInit(void** context, uint64_t commId, int* eActivationMask, const char* commName, int nNodes, int nranks, int rank, ncclDebugLogger_t logfn) {
  *context = malloc(sizeof(MyContext));
  *eActivationMask = 4095; /* enable ALL event types */
  return ncclSuccess;
}

ncclResult_t myStartEvent(void* context, void** eHandle, ncclProfilerEventDescr_v5_t* eDescr) {
  *eHandle = malloc(sizeof(MyEvent));
  return ncclSuccess;
}

ncclResult_t myStopEvent(void* eHandle) {
  free(eHandle);
  return ncclSuccess;
}

ncclResult_t myRecordEventState(void* eHandle, ncclProfilerEventState_v5_t eState, ncclProfilerEventStateArgs_v5_t* eStateArgs) {
  return ncclSuccess;
}

ncclResult_t myFinalize(void* context) {
  free(context);
  return ncclSuccess;
}

ncclProfiler_v5_t ncclProfiler_v5 = {
  "EmptyProfiler",
  myInit,
  myStartEvent,
  myStopEvent,
  myRecordEventState,
  myFinalize,
};
