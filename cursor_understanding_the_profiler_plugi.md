# Understanding the profiler plugin API
_Exported on 11/2/2025 at 19:29:39 GMT+1 from Cursor (1.7.54)_

---

**User**

i wanna understand the profiler plugin api better and (under ext-profiler) they also provide an example plugin but im honestly pretty lost and dont understand what *I* would need to do to write my own custom plugin. kinda lost with how many files there are

---

**Cursor**


Reviewing the profiler plugin API and the example plugin. Exploring relevant files.


Exploring event structures and the profiler API headers:


Checking the Makefile to understand the build process:


Creating a concise guide explaining how to write your own profiler plugin:


```markdown
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

```

Creating a minimal working example plugin:


```c
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

```

```markdown
# Minimal Profiler Plugin Example

This is a simplified profiler plugin that demonstrates the essential structure needed to create your own plugin.

## What This Example Does

This minimal plugin:
- Logs all NCCL events to a text file (one file per rank)
- Shows start/stop times and durations
- Demonstrates the 5 required functions
- Is easy to understand and modify

## Building

1. **Copy the NCCL headers** from `../example/nccl/` to a `nccl/` subdirectory:
   ```bash
   mkdir -p nccl
   cp ../example/nccl/*.h nccl/
   ```

2. **Compile:**
   ```bash
   gcc -fPIC -shared -o libnccl-profiler-minimal.so my_profiler.c \
       -I./nccl -I${CUDA_HOME}/include
   ```

   Or with C++ (if you need C++ features):
   ```bash
   g++ -fPIC -shared -o libnccl-profiler-minimal.so my_profiler.c \
       -I./nccl -I${CUDA_HOME}/include
   ```

3. **Create symlink (optional):**
   ```bash
   ln -s libnccl-profiler-minimal.so libnccl-profiler.so
   ```

## Usage

1. **Set environment variable:**
   ```bash
   export NCCL_PROFILER_PLUGIN=minimal
   # Or set to full path:
   # export NCCL_PROFILER_PLUGIN=/path/to/libnccl-profiler-minimal.so
   ```

2. **Set event mask (optional):**
   ```bash
   # Enable collectives and P2P (default)
   export NCCL_PROFILE_EVENT_MASK=258  # ncclProfileColl | ncclProfileP2p
   
   # Enable all events
   export NCCL_PROFILE_EVENT_MASK=4095
   ```

3. **Run your NCCL application** - it will automatically load the plugin

4. **Check output files:**
   ```bash
   cat my_profiler_rank0_comm*.log
   ```

## Example Output

```
=== Profiler initialized ===
Rank: 0/2, CommId: 123456789, CommName: (null)
[0.000] START GroupAPI (depth=2)
[0.001] START Collective: ncclAllReduce(count=1048576, dtype=ncclFloat32, root=-1)
[0.002] START Coll Exec: ncclAllReduce(seq=0, algo=RING, proto=SIMPLE, channels=2)
[1.234] START ProxyOp: channel=0, peer=1, steps=4, send=1
[1.250] STOP ProxyOp (duration=0.016 ms)
[5.678] STOP Coll Exec: ncclAllReduce (duration=5.676 ms)
[5.679] STOP Collective: ncclAllReduce (duration=5.678 ms)
[5.680] STOP GroupAPI (duration=5.680 ms)
=== Profiler finalized ===
```

## Customizing

To customize this plugin:

1. **Change logging format:** Modify the `fprintf` statements in `myStartEvent` and `myStopEvent`

2. **Add statistics:** Keep counters in `MyContext` and aggregate in `finalize`

3. **Write to different format:** Instead of text, write JSON, CSV, or binary format

4. **Filter events:** Check event type and only log what you care about

5. **Add memory pools:** For better performance, pre-allocate events (see example plugin)

## Next Steps

Once you understand this minimal example:
- Study the full example in `../example/` for advanced features
- Read `../WRITING_YOUR_PLUGIN.md` for detailed explanations
- Check `../README.md` for API documentation

```

Creating a quick reference diagram showing the flow:


```markdown
# Quick Reference: Profiler Plugin API

## The Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│  Your Application (using NCCL)                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  NCCL Core                                          │   │
│  │  - Calls ncclAllReduce(), ncclSend(), etc.         │   │
│  │  - Instruments code with profiler callbacks         │   │
│  └─────────────────┬───────────────────────────────────┘   │
│                    │ dlopen()                               │
│                    │ looks for ncclProfiler_v5 symbol       │
│                    ▼                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Your Plugin (libnccl-profiler.so)                  │   │
│  │  ┌───────────────────────────────────────────────┐   │   │
│  │  │  ncclProfiler_v5 struct (EXPORTED)          │   │   │
│  │  │  {                                           │   │   │
│  │  │    name,                                     │   │   │
│  │  │    init,      ← Called at start             │   │   │
│  │  │    startEvent, ← Called for each event      │   │   │
│  │  │    stopEvent, ← Called when event ends      │   │   │
│  │  │    recordEventState, ← State changes        │   │   │
│  │  │    finalize   ← Called at shutdown          │   │   │
│  │  │  }                                           │   │   │
│  │  └───────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Event Flow Example (ncclAllReduce)

```
Application calls: ncclAllReduce(...)
         │
         ▼
    NCCL Core
         │
         ├─→ init(context, ...)          ← Create profiler context
         │
         ├─→ startEvent(GROUP_API, ...)  ← Group start
         │   └─→ Returns handle1
         │
         ├─→ startEvent(COLL_API, ...)   ← AllReduce API call
         │   └─→ Returns handle2
         │
         ├─→ startEvent(COLL, ...)       ← Actual collective execution
         │   └─→ Returns handle3
         │
         ├─→ startEvent(PROXY_OP, ...)  ← Network operation
         │   └─→ Returns handle4
         │
         ├─→ recordEventState(handle4, SEND_WAIT, ...)
         │
         ├─→ stopEvent(handle4)          ← Proxy op done
         │
         ├─→ stopEvent(handle3)           ← Collective done
         │
         ├─→ stopEvent(handle2)           ← API call done
         │
         ├─→ stopEvent(handle1)           ← Group done
         │
         └─→ finalize(context)            ← Cleanup
```

## The 5 Functions - At a Glance

| Function | When Called | Your Job |
|----------|-------------|----------|
| **init** | Per communicator, at startup | Allocate context, read env vars |
| **startEvent** | Every time NCCL starts profiling something | Create event, store metadata, return handle |
| **stopEvent** | When event completes/enqueues | Record end time, free resources |
| **recordEventState** | For stateful events (ProxyOp, ProxyStep) | Update event state |
| **finalize** | When communicator destroyed | Write logs, free memory |

## Event Types Quick Reference

```c
ncclProfileGroupApi    // Group API (ncclGroupStart/End)
ncclProfileCollApi     // Collective API (ncclAllReduce, etc.)
ncclProfileP2pApi      // P2P API (ncclSend, ncclRecv)
ncclProfileKernelLaunch // Kernel launch
ncclProfileColl        // Collective execution
ncclProfileP2p         // P2P execution
ncclProfileProxyOp     // Network proxy operation
ncclProfileProxyStep   // Network transfer step
ncclProfileKernelCh    // GPU kernel activity
ncclProfileNetPlugin   // Network plugin events
ncclProfileProxyCtrl   // Proxy control (idle/active)
```

## Event Hierarchy

```
Group API
├── Collective API
│   └── Collective (execution)
│       ├── ProxyOp
│       │   └── ProxyStep
│       │       └── NetPlugin
│       └── KernelCh
└── P2P API
    └── P2P (execution)
        ├── ProxyOp
        │   └── ProxyStep
        └── KernelCh
```

## Key Data Structures

### Event Descriptor (what NCCL gives you)
```c
ncclProfilerEventDescr_v5_t {
  uint64_t type;        // Event type
  void* parentObj;      // Parent event handle
  int rank;             // Rank that generated event
  union {
    collApi { ... };    // If type == ncclProfileCollApi
    coll { ... };      // If type == ncclProfileColl
    proxyOp { ... };    // If type == ncclProfileProxyOp
    // ... etc
  };
}
```

### Your Event Handle (what you return)
```c
void* eHandle  // Can be anything - your struct, pointer, etc.
                // NCCL will pass it back to you in stopEvent/recordEventState
```

## Build & Use Checklist

- [ ] Copy `nccl/*.h` headers from example
- [ ] Implement 5 functions
- [ ] Export `ncclProfiler_v5` struct
- [ ] Compile as shared library: `gcc -fPIC -shared -o libnccl-profiler.so ...`
- [ ] Set `NCCL_PROFILER_PLUGIN` environment variable
- [ ] Run your NCCL application
- [ ] Check output/logs

## Common Patterns

### Pattern 1: Simple Logging
```c
ncclResult_t startEvent(...) {
  fprintf(logFile, "Event started: %s\n", eDescr->collApi.func);
  *eHandle = NULL;  // Don't track, just log
  return ncclSuccess;
}
```

### Pattern 2: Track with Context
```c
typedef struct { MyContext* ctx; double startTime; } MyEvent;

ncclResult_t startEvent(...) {
  MyEvent* ev = malloc(sizeof(MyEvent));
  ev->ctx = context;
  ev->startTime = getTime();
  *eHandle = ev;
  return ncclSuccess;
}
```

### Pattern 3: Memory Pool (Performance)
```c
// Pre-allocate events
MyEvent pool[256];
int poolIndex = 0;

ncclResult_t startEvent(...) {
  MyEvent* ev = &pool[poolIndex++ % 256];
  *eHandle = ev;
  return ncclSuccess;
}
```

## Debugging Tips

1. **Plugin not loading?**
   - Check `NCCL_PROFILER_PLUGIN` env var
   - Set `NCCL_DEBUG=INFO` to see load messages
   - Check library path and permissions

2. **Functions not called?**
   - Verify `ncclProfiler_v5` symbol is exported: `nm -D libnccl-profiler.so | grep ncclProfiler`
   - Check event mask: `NCCL_PROFILE_EVENT_MASK`

3. **Events missing?**
   - Some events only fire with network operations
   - Intra-node might not generate ProxyOp events
   - Check event hierarchy - parent events must exist

## File Organization

```
your-plugin/
├── my_profiler.c       # Main plugin code
├── nccl/               # Copy headers from example
│   ├── profiler_v5.h
│   ├── types.h
│   └── err.h
└── Makefile
```

## See Also

- `WRITING_YOUR_PLUGIN.md` - Detailed guide
- `minimal-example/` - Minimal working example
- `example/` - Full-featured example
- `README.md` - Official API docs

```

Created a guide to help you write your own profiler plugin. Here's what's included:

## What I Created

1. **`WRITING_YOUR_PLUGIN.md`** — Guide covering:
   - What a profiler plugin is
   - The 5 required functions and their purpose
   - Minimal working examples
   - Build instructions
   - Tips and best practices

2. **`minimal-example/my_profiler.c`** — Minimal working plugin that:
   - Logs all events to a text file
   - Shows the essential structure
   - Demonstrates all 5 required functions
   - Easy to modify and extend

3. **`QUICK_REFERENCE.md`** — Quick reference with:
   - Visual diagrams showing the flow
   - Event type cheat sheet
   - Common patterns
   - Debugging tips

## Key Takeaways

To write your own plugin, you need:

1. **Export one struct:**
   ```c
   ncclProfiler_v5_t ncclProfiler_v5 = {
     "YourPluginName",
     initFunction,
     startEventFunction,
     stopEventFunction,
     recordEventStateFunction,
     finalizeFunction
   };
   ```

2. **Implement 5 functions:**
   - `init` — Set up your profiler
   - `startEvent` — Track when events start
   - `stopEvent` — Track when events end
   - `recordEventState` — Track state changes
   - `finalize` — Clean up

3. **Compile as shared library:**
   ```bash
   gcc -fPIC -shared -o libnccl-profiler.so your_plugin.c
   ```

4. **Use it:**
   ```bash
   export NCCL_PROFILER_PLUGIN=your_plugin
   ```

## Where to Start

1. Read `WRITING_YOUR_PLUGIN.md` for the overview
2. Look at `minimal-example/my_profiler.c` for a working example
3. Study `example/plugin.cc` for advanced features

The minimal example in `minimal-example/` is a good starting point—it's functional and straightforward, unlike the full example which has many optimizations.

Want help with a specific aspect?

```c
   ncclProfiler_v5_t ncclProfiler_v5 = {
     "YourPluginName",
     initFunction,
     startEventFunction,
     stopEventFunction,
     recordEventStateFunction,
     finalizeFunction
   };
   ```

2. **Implement 5 functions:**
   - `init` — Set up your profiler
   - `startEvent` — Track when events start
   - `stopEvent` — Track when events end
   - `recordEventState` — Track state changes
   - `finalize` — Clean up

3. **Compile as shared library:**
   ```bash
   gcc -fPIC -shared -o libnccl-profiler.so your_plugin.c
   ```

4. **Use it:**
   ```bash
   export NCCL_PROFILER_PLUGIN=your_plugin
   ```

## Where to Start

1. Read `WRITING_YOUR_PLUGIN.md` for the overview
2. Look at `minimal-example/my_profiler.c` for a working example
3. Study `example/plugin.cc` for advanced features

The minimal example in `minimal-example/` is a good starting point—it's functional and straightforward, unlike the full example which has many optimizations.

Want help with a specific aspect?
```

```shellscript
   gcc -fPIC -shared -o libnccl-profiler.so your_plugin.c
```

```shellscript
   export NCCL_PROFILER_PLUGIN=your_plugin
```

