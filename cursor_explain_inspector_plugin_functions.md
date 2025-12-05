# Explain inspector plugin functionality in detail
_Exported on 11/30/2025 at 22:24:08 GMT+1 from Cursor (2.0.77)_

---

**User**

@inspector_plugin.cc explain how this works in detail

---

**Cursor**


Reviewing related files to understand the context and data structures.



## Overview

This file implements an NCCL profiler plugin that tracks collective operations and kernel channel events. It records timing, sequences, and performance metrics for NCCL collectives.

## Architecture

### 1. Plugin interface registration (lines 486-493)

The plugin registers with NCCL via a function pointer struct:

```c
ncclProfiler_t ncclProfiler_v5 = {
  "Inspector",
  inspectorPluginInit,              // Initialize plugin
  inspectorPluginStartEvent,        // Start tracking an event
  inspectorPluginStopEvent,         // Stop tracking an event
  inspectorPluginRecordEventState,  // Record state transitions
  inspectorPluginFinalize,          // Cleanup
};
```

### 2. Initialization (`inspectorPluginInit`, lines 86-120)

- Thread-safe via `gLock` mutex
- First call initializes global state (`inspectorGlobalInit`)
- Creates a communicator info structure (`inspectorAddComm`)
- Sets activation mask to track collectives (`ncclProfileColl`) and kernel channels (`ncclProfileKernelCh`)
- Returns a context handle for the communicator

### 3. Reference counting system (lines 151-179)

Manages lifecycle of `inspectorCollInfo` structures:

- `inspectorPluginCollInfoRef`: increments ref count
- `inspectorPluginCollInfoDeRef`: decrements; frees when count reaches 0
- Safe versions (`*Safe`) use read-write locks

### 4. Event tracking

#### Event trace recording (lines 51-58)

`inspectorRecordEventTrace`:
- Records timestamp (`inspectorGetTime()`)
- Assigns a sequence number from the collective's counter
- Used for timing analysis

#### Collective event initialization (lines 202-234)

`inspectorPluginCollInfoInit`:
- Allocates `inspectorCollInfo`
- Sets initial ref count (self-ref + extra if channels exist)
- Records function, sequence number, channel count, message size
- Records COLL_START event trace
- Initializes read-write lock

#### Kernel channel initialization (lines 258-292)

`inspectorPluginKernelChInfoInit`:
- Validates parent is a collective
- Locks the collective's guard
- Records KERNEL_START event trace
- Initializes kernel channel info (channel ID, GPU clock, timestamp)
- Updates collective's kernel channel counters
- Increments ref count for the collective

### 5. Event lifecycle management

#### Start event (`inspectorPluginStartEvent`, lines 313-334)

- Routes by event type:
  - `ncclProfileColl`: creates `inspectorCollInfo`
  - `ncclProfileKernelCh`: creates `inspectorKernelChInfo`
- Returns an event handle

#### Stop event (`inspectorPluginStopEvent`, lines 357-429)

For collectives:
- Records COLL_STOP event trace
- Decrements ref count
- If count reaches 0, structure is freed

For kernel channels:
- Records KERNEL_STOP event trace
- Updates completion timestamp
- Increments completed counter
- When all channels complete:
  - Updates collective performance metrics
  - Computes bandwidth
  - Marks communicator for dump
  - Decrements ref count

#### Record event state (`inspectorPluginRecordEventState`, lines 453-484)

Handles state transitions:
- For `ncclProfilerKernelChStop`:
  - Records KERNEL_RECORD event trace
  - Updates GPU clock stop time
  - Decrements ref count if start GPU clock was set

### 6. Thread safety

- Global initialization uses `gLock` mutex
- Each `inspectorCollInfo` has a `pthread_rwlock_t guard`
- Locking pattern:
  - Write lock (`inspectorLockWr`) for modifications
  - Unlock (`inspectorUnlockRWLock`) after operations
- Reference counting is protected by locks

### 7. Event trace system

Tracks three event types per collective:
- `NCCL_INSP_EVT_TRK_COLL_START/STOP`: collective boundaries
- `NCCL_INSP_EVT_TRK_KERNEL_START/STOP/RECORD`: kernel channel events

Each trace entry has:
- `ts`: timestamp
- `sn`: sequence number (monotonically increasing per collective)

### 8. Data flow

```
NCCL Collective Call
    ↓
inspectorPluginStartEvent (ncclProfileColl)
    ↓
inspectorPluginCollInfoInit
    ↓ [creates inspectorCollInfo with refCount]
    ↓
inspectorPluginStartEvent (ncclProfileKernelCh) [for each channel]
    ↓
inspectorPluginKernelChInfoInit
    ↓ [updates collective, increments refCount]
    ↓
inspectorPluginRecordEventState (KERNEL_STOP)
    ↓ [records GPU clock]
    ↓
inspectorPluginStopEvent (ncclProfileKernelCh)
    ↓ [when all channels done]
    ↓
inspectorUpdateCollPerf + inspectorComputeCollBw
    ↓
inspectorPluginStopEvent (ncclProfileColl)
    ↓
[collective marked for dump, eventually freed when refCount=0]
```

### 9. Finalization (`inspectorPluginFinalize`, lines 141-149)

- Removes communicator from tracking
- Decrements global initialization counter
- If last communicator, calls `inspectorGlobalFinalize`

## Design patterns

1. Reference counting: manages lifecycle of shared structures
2. Event tracing: timestamps and sequence numbers for analysis
3. Hierarchical tracking: kernel channels nested under collectives
4. Thread-safe operations: locks protect shared state
5. Plugin architecture: implements NCCL profiler interface

This enables profiling NCCL collectives with timing, bandwidth, and event trace data for performance analysis.

```c
ncclProfiler_t ncclProfiler_v5 = {
  "Inspector",
  inspectorPluginInit,              // Initialize plugin
  inspectorPluginStartEvent,        // Start tracking an event
  inspectorPluginStopEvent,         // Stop tracking an event
  inspectorPluginRecordEventState,  // Record state transitions
  inspectorPluginFinalize,          // Cleanup
};
```

```plaintext
NCCL Collective Call
    ↓
inspectorPluginStartEvent (ncclProfileColl)
    ↓
inspectorPluginCollInfoInit
    ↓ [creates inspectorCollInfo with refCount]
    ↓
inspectorPluginStartEvent (ncclProfileKernelCh) [for each channel]
    ↓
inspectorPluginKernelChInfoInit
    ↓ [updates collective, increments refCount]
    ↓
inspectorPluginRecordEventState (KERNEL_STOP)
    ↓ [records GPU clock]
    ↓
inspectorPluginStopEvent (ncclProfileKernelCh)
    ↓ [when all channels done]
    ↓
inspectorUpdateCollPerf + inspectorComputeCollBw
    ↓
inspectorPluginStopEvent (ncclProfileColl)
    ↓
[collective marked for dump, eventually freed when refCount=0]
```

