# Writing Your Own NCCL Profiler Plugin - A Guide

This guide breaks down what you need to implement to create your own custom profiler plugin.

## Overview: What is a Profiler Plugin?

A profiler plugin is a **shared library** (`libnccl-profiler.so`) that NCCL dynamically loads. NCCL calls your plugin functions at key points during execution to give you visibility into:
- Collective operations (AllReduce, Broadcast, etc.)
- Point-to-point operations (Send/Recv)
- Network activity (proxy operations)
- Kernel launches
- Group API calls

## Minimal Structure: What You MUST Implement

At the core, you need to implement **5 functions** and export **1 struct**:

```c
// This struct MUST be exported as ncclProfiler_v5
ncclProfiler_v5_t ncclProfiler_v5 = {
  "YourPluginName",           // Plugin name (for logging)
  yourInitFunction,          // Called when NCCL initializes
  yourStartEventFunction,    // Called when an event starts
  yourStopEventFunction,     // Called when an event stops
  yourRecordEventStateFunction, // Called for state transitions
  yourFinalizeFunction,      // Called when NCCL shuts down
};
```

## The 5 Required Functions

### 1. `init` - Initialize Your Profiler

**Called:** Once per NCCL communicator when it's created

**Purpose:** Set up your profiler context (allocate memory, read env vars, etc.)

**Signature:**
```c
ncclResult_t yourInit(void** context, 
                      uint64_t commId, 
                      int* eActivationMask,
                      const char* commName, 
                      int nNodes, 
                      int nranks, 
                      int rank, 
                      ncclDebugLogger_t logfn)
```

**What to do:**
- Allocate and initialize your context structure
- Set `*context` to point to your context
- Optionally set `*eActivationMask` to enable specific events (or let NCCL use defaults)
- Return `ncclSuccess` on success, `ncclSystemError` to disable the plugin

**Example:**
```c
ncclResult_t myInit(void** context, uint64_t commId, int* eActivationMask, 
                    const char* commName, int nNodes, int nranks, int rank, 
                    ncclDebugLogger_t logfn) {
  // Allocate your context
  MyContext* ctx = calloc(1, sizeof(MyContext));
  if (!ctx) return ncclSystemError;
  
  // Store info
  ctx->commId = commId;
  ctx->rank = rank;
  ctx->nranks = nranks;
  
  // Maybe read env vars to configure behavior
  const char* dumpFile = getenv("MY_PROFILER_DUMP_FILE");
  ctx->dumpFile = dumpFile ? strdup(dumpFile) : NULL;
  
  *context = ctx;
  return ncclSuccess;
}
```

### 2. `startEvent` - Start Tracking an Event

**Called:** Every time NCCL wants to start profiling something

**Purpose:** Create an event handle and store event metadata

**Signature:**
```c
ncclResult_t yourStartEvent(void* context, 
                            void** eHandle, 
                            ncclProfilerEventDescr_v5_t* eDescr)
```

**What to do:**
- Check `eDescr->type` to see what kind of event it is
- Allocate/store event data (or use a pool like the example)
- Set `*eHandle` to your event object (this will be passed back in other calls)
- Extract relevant info from `eDescr->union.{groupApi|collApi|p2pApi|...}`

**Key Event Types:**
- `ncclProfileGroupApi` - Group API call (ncclGroupStart/End)
- `ncclProfileCollApi` - Collective API call (ncclAllReduce, etc.)
- `ncclProfileP2pApi` - Point-to-point API call (ncclSend, ncclRecv)
- `ncclProfileColl` - Actual collective operation execution
- `ncclProfileP2p` - Actual point-to-point operation execution
- `ncclProfileProxyOp` - Network proxy operation
- `ncclProfileProxyStep` - Individual network transfer step
- `ncclProfileKernelCh` - GPU kernel activity
- `ncclProfileNetPlugin` - Network plugin events

**Example:**
```c
ncclResult_t myStartEvent(void* context, void** eHandle, 
                          ncclProfilerEventDescr_v5_t* eDescr) {
  MyContext* ctx = (MyContext*)context;
  
  if (eDescr->type == ncclProfileCollApi) {
    // Allocate event structure
    MyCollApiEvent* event = malloc(sizeof(MyCollApiEvent));
    event->func = eDescr->collApi.func;  // e.g., "ncclAllReduce"
    event->count = eDescr->collApi.count;
    event->startTime = getCurrentTime();
    
    *eHandle = event;
  }
  // Handle other event types...
  
  return ncclSuccess;
}
```

### 3. `stopEvent` - Stop Tracking an Event

**Called:** When an event completes or is enqueued

**Purpose:** Finalize event tracking, record end time

**Signature:**
```c
ncclResult_t yourStopEvent(void* eHandle)
```

**What to do:**
- Record the end time
- If using reference counting (like the example), decrement counter
- Free resources if the event is truly complete
- Note: For some events, `stopEvent` just means "enqueued", not "completed"

**Example:**
```c
ncclResult_t myStopEvent(void* eHandle) {
  if (!eHandle) return ncclSuccess;
  
  MyCollApiEvent* event = (MyCollApiEvent*)eHandle;
  event->endTime = getCurrentTime();
  event->duration = event->endTime - event->startTime;
  
  // Maybe write to file, aggregate stats, etc.
  
  return ncclSuccess;
}
```

### 4. `recordEventState` - Update Event State

**Called:** For events that have state transitions (ProxyOp, ProxyStep, etc.)

**Purpose:** Track state changes during an event's lifetime

**Signature:**
```c
ncclResult_t yourRecordEventState(void* eHandle, 
                                   ncclProfilerEventState_v5_t eState, 
                                   ncclProfilerEventStateArgs_v5_t* eStateArgs)
```

**What to do:**
- Check `eState` to see what state transition occurred
- Update your event structure accordingly
- Extract additional info from `eStateArgs` if provided

**Common States:**
- `ncclProfilerProxyStepSendGPUWait` - Waiting for GPU data
- `ncclProfilerProxyStepRecvWait` - Waiting for network data
- `ncclProfilerProxyCtrlIdle` - Proxy thread is idle
- etc.

**Example:**
```c
ncclResult_t myRecordEventState(void* eHandle, 
                                 ncclProfilerEventState_v5_t eState,
                                 ncclProfilerEventStateArgs_v5_t* eStateArgs) {
  if (!eHandle) return ncclSuccess;
  
  // Check event type and update accordingly
  if (eventType == ncclProfileProxyStep) {
    MyProxyStepEvent* event = (MyProxyStepEvent*)eHandle;
    event->currentState = eState;
    event->stateTimestamp = getCurrentTime();
  }
  
  return ncclSuccess;
}
```

### 5. `finalize` - Cleanup

**Called:** When the communicator is destroyed

**Purpose:** Clean up resources, dump collected data

**Signature:**
```c
ncclResult_t yourFinalize(void* context)
```

**What to do:**
- Write collected traces/stats to file
- Free all allocated memory
- Close files, etc.

**Example:**
```c
ncclResult_t myFinalize(void* context) {
  MyContext* ctx = (MyContext*)context;
  
  // Dump collected events to file
  if (ctx->dumpFile) {
    FILE* f = fopen(ctx->dumpFile, "w");
    // Write events...
    fclose(f);
  }
  
  // Free context
  free(ctx);
  return ncclSuccess;
}
```

## Minimal Working Example

Here's a **minimal** profiler that just logs events to a file:

```c
#include "nccl/profiler_v5.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
  FILE* logFile;
  uint64_t commId;
  int rank;
} MyContext;

double getTime() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

ncclResult_t myInit(void** context, uint64_t commId, int* eActivationMask,
                    const char* commName, int nNodes, int nranks, int rank,
                    ncclDebugLogger_t logfn) {
  MyContext* ctx = calloc(1, sizeof(MyContext));
  ctx->commId = commId;
  ctx->rank = rank;
  
  char filename[256];
  snprintf(filename, sizeof(filename), "my_profiler_rank%d.log", rank);
  ctx->logFile = fopen(filename, "w");
  
  *context = ctx;
  return ncclSuccess;
}

ncclResult_t myStartEvent(void* context, void** eHandle, 
                          ncclProfilerEventDescr_v5_t* eDescr) {
  MyContext* ctx = (MyContext*)context;
  
  // Just store a simple event info
  typedef struct {
    uint64_t type;
    double startTime;
    const char* func;
  } SimpleEvent;
  
  SimpleEvent* event = malloc(sizeof(SimpleEvent));
  event->type = eDescr->type;
  event->startTime = getTime();
  
  if (eDescr->type == ncclProfileCollApi) {
    event->func = eDescr->collApi.func;
    fprintf(ctx->logFile, "[%f] START Collective: %s\n", 
            event->startTime, event->func);
  }
  
  *eHandle = event;
  return ncclSuccess;
}

ncclResult_t myStopEvent(void* eHandle) {
  if (!eHandle) return ncclSuccess;
  
  SimpleEvent* event = (SimpleEvent*)eHandle;
  double endTime = getTime();
  
  // Note: Would need to store context pointer in event to log
  // This is simplified - see example plugin for proper structure
  
  free(event);
  return ncclSuccess;
}

ncclResult_t myRecordEventState(void* eHandle, 
                                 ncclProfilerEventState_v5_t eState,
                                 ncclProfilerEventStateArgs_v5_t* eStateArgs) {
  // Simple implementation - do nothing
  return ncclSuccess;
}

ncclResult_t myFinalize(void* context) {
  MyContext* ctx = (MyContext*)context;
  if (ctx->logFile) fclose(ctx->logFile);
  free(ctx);
  return ncclSuccess;
}

// EXPORT THIS STRUCT - NCCL looks for this symbol!
ncclProfiler_v5_t ncclProfiler_v5 = {
  "MySimpleProfiler",
  myInit,
  myStartEvent,
  myStopEvent,
  myRecordEventState,
  myFinalize,
};
```

## Building Your Plugin

1. **Copy the header files** from `ext-profiler/example/nccl/` to your project
2. **Compile as a shared library:**

```bash
gcc -fPIC -shared -o libnccl-profiler-myplugin.so your_plugin.c \
    -I./nccl -I${CUDA_HOME}/include
```

3. **Create a symlink** (optional but recommended):
```bash
ln -s libnccl-profiler-myplugin.so libnccl-profiler.so
```

4. **Use it:**
```bash
export NCCL_PROFILER_PLUGIN=myplugin  # or full path
# or put libnccl-profiler.so in LD_LIBRARY_PATH
```

## Key Files in the Example Plugin

To understand the full implementation, study these files:

1. **`plugin.cc`** - Main implementation with all 5 functions
2. **`event.h`** - Event data structures (how events are stored)
3. **`print_event.cc`** - How to output events (JSON format for Chrome tracing)
4. **`nccl/profiler_v5.h`** - API definitions (what NCCL passes to you)
5. **`Makefile`** - Build instructions

## Understanding Event Hierarchy

Events form a hierarchy:
```
Group API Event
  ├── Collective API Event
  │   └── Collective Event
  │       ├── ProxyOp Event
  │       │   └── ProxyStep Event
  │       │       └── NetPlugin Event
  │       └── KernelCh Event
  └── P2P API Event
      └── P2P Event
          └── ...
```

Use `eDescr->parentObj` to link events together!

## Environment Variables

- `NCCL_PROFILER_PLUGIN` - Plugin library name or path
- `NCCL_PROFILE_EVENT_MASK` - Which events to enable (bitmask)
- `NCCL_DEBUG=INFO` - See plugin loading messages

## Tips

1. **Start simple:** Just implement `init`, `startEvent`, `stopEvent`, and `finalize`
2. **Use the example as reference:** It's well-commented and handles all edge cases
3. **Memory pools:** The example uses pools to avoid malloc overhead - optional but recommended for performance
4. **Reference counting:** Some events (like collectives) only complete when all child events finish - use ref counting
5. **Thread safety:** Multiple threads may call your plugin - use atomic operations or locks

## What Makes It Work?

NCCL uses `dlopen()` to load your library, then looks for the symbol `ncclProfiler_v5`. If found, it fills function pointers from your struct and calls them at appropriate times.

That's it! The rest is up to you - how you store events, what you do with them, how you output them.

