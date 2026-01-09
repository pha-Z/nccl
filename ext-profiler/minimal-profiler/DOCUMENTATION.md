# minimal_profiler.cc Documentation

## Table of Contents

### 1. Overview
   - 1.1 What the Profiler Does
   - 1.2 The NCCL Profiler API

### 2. Event Tracking
   - 2.1 Event Lifecycle
   - 2.2 Parent-Child Relationships
   - 2.3 Dealing with Address Reuse
      - 2.3.1 The Problem
      - 2.3.2 Memory Pools
      - 2.3.3 Blacklisting
      - 2.3.4 How Pools and Blacklist Work Together
   - 2.4 Cross-Process Events (PXN)
      - 2.4.1 What is PXN Routing
      - 2.4.2 The Invalid Pointer Problem
      - 2.4.3 Context Validation
      - 2.4.4 The Detach Pool

### 3. GPU Identification

### 4. Communicator Information

### 5. Process and Thread Tracking

### 6. Logging
   - 6.1 Per-Rank Log Files
   - 6.2 PXN Log File
   - 6.3 Log Entry Format
   - 6.4 Directory Naming

### 7. Thread Safety
   - 7.1 NCCL's Threading Model
   - 7.2 Mutexes
   - 7.3 Lock Ordering

### 8. Reference
   - 8.1 Data Structures
   - 8.2 API Functions
   - 8.3 Event Types
   - 8.4 Configuration

---

## 1. Overview

### 1.1 What the Profiler Does

The `minimal_profiler.cc` implements an NCCL profiler plugin that tracks and logs NCCL communication events. It is loaded dynamically by NCCL at runtime and receives callbacks whenever profiled events occur.

**Core functionality includes:**

1. **Event Lifecycle Tracking**: The profiler allocates metadata storage when events start, logs information during execution, and deallocates when events complete. Each event carries timing information, type-specific metadata, and relationship data.

2. **Parent-Child Relationship Tracking**: NCCL events form hierarchies. For example, a collective API call spawns internal collective operations, which spawn proxy operations for network transfer. The profiler maintains these relationships through pointer references, enabling trace reconstruction.

3. **Cross-Process Event Handling**: When NCCL uses PXN (PCI x NVLink) routing, some events may execute in a different process than where they originated. The profiler detects these cases and handles them with separate infrastructure.

4. **Context Isolation**: Each NCCL rank gets its own profiler context with dedicated resources (memory pool, blacklist, log file), preventing interference between concurrent ranks.

### 1.2 The NCCL Profiler API

NCCL's profiler plugin API (version 5) defines a contract between NCCL and profiler plugins. The API requires five callback functions that NCCL calls at specific points during execution.

| Function | When NCCL Calls It | Purpose |
|----------|-------------------|---------|
| `myInit` | During communicator creation (`ncclCommInit*`) | Create and initialize a profiler context for this rank |
| `myStartEvent` | When a profilable event begins | Allocate storage for the event and log its start |
| `myStopEvent` | When the event completes | Log completion, calculate duration, free storage |
| `myRecordEventState` | During event execution (state transitions) | Log intermediate state changes |
| `myFinalize` | During communicator destruction (`ncclCommDestroy`) | Clean up profiler context and close log files |

**Plugin Registration**

The plugin exports a struct named `ncclProfiler_v5` that contains pointers to these five functions plus the plugin name. NCCL looks for this symbol when loading the plugin library:

```cpp
ncclProfiler_v5_t ncclProfiler_v5 = {
  "MyMinimalProfiler",
  myInit,
  myStartEvent,
  myStopEvent,
  myRecordEventState,
  myFinalize,
};
```

**Event Activation Mask**

During `myInit`, the profiler receives and sets an event activation mask (`eActivationMask`). This bitmask controls which event types NCCL will report. The profiler reads this from the `NCCL_PROFILE_EVENT_MASK` environment variable, defaulting to 4095 (all event types enabled).

---

## 2. Event Tracking

### 2.1 Event Lifecycle

Each event tracked by the profiler goes through three phases:

**Phase 1: Start (`myStartEvent`)**

When NCCL calls `myStartEvent`, it passes:
- `context`: Pointer to the profiler context for this rank
- `eHandle`: Output pointer where the profiler stores a handle to its event data
- `eDescr`: Event descriptor containing type and type-specific fields

The profiler:
1. Validates the context (for PXN detection)
2. Allocates a `MyEvent` struct from the appropriate pool
3. Extracts type-specific information from `eDescr` (function name, parameters, etc.)
4. Records the start timestamp using `clock_gettime(CLOCK_MONOTONIC)`
5. Handles parent reference tracking (adds to blacklist if `parentObj` is set)
6. Logs the START entry to the appropriate log file
7. Returns the `MyEvent` pointer via `eHandle`

**Phase 2: State Changes (`myRecordEventState`)**

During event execution, NCCL may call `myRecordEventState` to report state transitions. This is common for proxy operations which go through multiple states (waiting, transmitting, etc.).

The profiler:
1. Retrieves the `MyEvent` from `eHandle`
2. Maps the state enum to a human-readable name
3. Extracts state-specific arguments if provided
4. Logs the STATE entry with current timestamp

**Phase 3: Stop (`myStopEvent`)**

When the event completes, NCCL calls `myStopEvent`. The profiler:
1. Retrieves the `MyEvent` from `eHandle`
2. Records the end timestamp
3. Calculates duration (end - start)
4. Logs the STOP entry
5. Decrements the parent's blacklist reference count (if applicable)
6. Returns the event to the pool (or deletes if dynamically allocated)

### 2.2 Parent-Child Relationships

Events in NCCL form hierarchical relationships. The event descriptor contains a `parentObj` field that points to the parent event's `MyEvent` struct.

**Example Event Hierarchy:**

```
ncclProfileGroupApi (depth=0)
├── ncclProfileCollApi (ncclAllReduce)
│   └── ncclProfileColl (AllReduce, RING)
│       ├── ncclProfileProxyOp (channel=0, send)
│       │   ├── ncclProfileProxyStep (step=0)
│       │   ├── ncclProfileProxyStep (step=1)
│       │   └── ...
│       └── ncclProfileProxyOp (channel=0, recv)
│           └── ...
└── ncclProfileKernelLaunch
```

The profiler stores `parentObj` in the `MyEvent` struct for two purposes:

1. **Logging**: The parent pointer is logged with each event, allowing trace reconstruction
2. **Address Reuse Prevention**: The parent's address is tracked in the blacklist to prevent it from being reused while children are still active

### 2.3 Dealing with Address Reuse

#### 2.3.1 The Problem

The parent-child relationship relies on pointer values. When a child event stores `parentObj = 0x1000`, it expects that address to refer to its actual parent throughout the child's lifetime.

With simple `malloc`/`free`, this assumption breaks:

```
Timeline:
1. Parent event A allocated at address 0x1000
2. Child event B starts, stores parentObj = 0x1000
3. Parent A completes, freed back to allocator
4. New event C starts, malloc returns 0x1000 (address reuse!)
5. Child B completes, sees parentObj = 0x1000 → thinks C is its parent
```

This corrupts the event hierarchy. Any post-processing that tries to reconstruct parent-child relationships will produce incorrect results.

The fundamental issue is that addresses get reused too quickly. The profiler's solution has two parts:
- **Memory pools**: Maximize the time before an address is reused
- **Blacklisting**: Prevent reuse of addresses still referenced as parents

#### 2.3.2 Memory Pools

The profiler uses memory pools (`EventPool`) to control allocation order and maximize time-to-reuse.

**Chunked Allocation**

Instead of a single large array, the pool uses multiple fixed-size chunks:

```
Pool Structure:
┌─────────────────────────────────────────────────┐
│ Chunk 0: [slot 0][slot 1][slot 2]...[slot 1023] │
├─────────────────────────────────────────────────┤
│ Chunk 1: [slot 0][slot 1][slot 2]...[slot 1023] │
├─────────────────────────────────────────────────┤
│ Chunk 2: [slot 0][slot 1][slot 2]...[slot 1023] │
└─────────────────────────────────────────────────┘
```

Each chunk is allocated separately and remains valid until pool destruction. Chunks are never deleted during normal operation - this ensures that event pointers remain valid even after the event is freed.

The `chunks` vector stores pointers to each chunk's array. A parallel `inUseChunks` vector stores boolean arrays tracking which slots are currently allocated.

**Round-Robin Allocation**

When allocating an event, the pool searches starting from where the last allocation occurred, wrapping around through all slots:

```cpp
for (size_t attempt = 0; attempt < totalSlots; attempt++) {
  size_t chunkIdx = (startChunk + (startSlot + attempt) / chunkSize) % chunks.size();
  size_t slotInChunk = (startSlot + attempt) % chunkSize;
  
  if (!inUseChunks[chunkIdx][slotInChunk]) {
    // Found free slot - check blacklist before allocating
    ...
  }
}
```

This round-robin approach ensures that recently freed addresses are used last. If you free slot 5, the allocator won't return to slot 5 until it has cycled through slots 6, 7, ..., N, 0, 1, 2, 3, 4 first.

With 1024 slots per chunk and typical event lifetimes, this provides substantial time before any address is reused.

**Pool Growth**

If all slots are in-use or blacklisted, the pool grows by allocating a new chunk:

```cpp
if (grow(logFile)) {
  // Allocate from first slot of new chunk
  size_t newChunkIdx = chunks.size() - 1;
  MyEvent* event = &chunks[newChunkIdx][0];
  ...
}
```

The pool logs a message when it grows, allowing monitoring of pool size under load.

**Fallback to Dynamic Allocation**

If chunk allocation fails (out of memory), the pool falls back to regular `new`:

```cpp
MyEvent* event = new (std::nothrow) MyEvent();
if (event) {
  event->fromPool = false;  // Mark as dynamically allocated
}
```

These events are deleted with `delete` rather than returned to the pool. The `fromPool` flag tracks this distinction.

#### 2.3.3 Blacklisting

Even with round-robin allocation, an address could be reused while still referenced as a parent. The blacklist prevents this by tracking addresses that must not be allocated.

**When Addresses Get Blacklisted**

When `myStartEvent` processes an event with a non-null `parentObj`:

```cpp
if (eDescr->parentObj) {
  if (ctx->blacklist.contains(eDescr->parentObj)) {
    ctx->blacklist.incrementRef(eDescr->parentObj);
  } else {
    ctx->blacklist.add(eDescr->parentObj, ctx->logFile);
  }
}
```

The parent's address is added to the blacklist (or its reference count is incremented if already present).

**Reference Counting**

Each blacklisted address has a `refCount` tracking how many active children reference it. This handles the case where multiple children share the same parent:

```
Parent A (address 0x1000)
├── Child B (parentObj = 0x1000) → refCount = 1
├── Child C (parentObj = 0x1000) → refCount = 2
└── Child D (parentObj = 0x1000) → refCount = 3
```

When children complete, they decrement the count:

```cpp
// In myStopEvent:
if (event->parentObj) {
  ctx->blacklist.decrementRef(event->parentObj);
}
```

When the reference count reaches zero, the address is removed from the blacklist and becomes available for allocation again.

**Hash Map + Doubly-Linked List**

The blacklist uses two data structures in combination:

1. **Hash Map** (`std::unordered_map<void*, BlacklistNode*>`): Provides O(1) lookup to check if an address is blacklisted
2. **Doubly-Linked List**: Maintains LRU (Least Recently Used) order for O(1) eviction

```
Hash Map:                    LRU Doubly-Linked List:
┌─────────┬──────────┐       
│ Address │ Node Ptr │       head                              tail
├─────────┼──────────┤       (oldest)                          (newest)
│ 0x1000  │    ──────┼──────►[0x1000]◄──►[0x2000]◄──►[0x3000]◄──►[0x4000]
│ 0x2000  │    ──────┼─────────────────►│
│ 0x3000  │    ──────┼─────────────────────────────►│
│ 0x4000  │    ──────┼─────────────────────────────────────────►│
└─────────┴──────────┘
```

Operations:
- `contains(addr)`: Hash map lookup, O(1)
- `add(addr)`: Hash map insert + link at tail, O(1)
- `incrementRef(addr)`: Hash lookup + move to tail, O(1)
- `decrementRef(addr)`: Hash lookup + potential unlink + delete, O(1)
- `evictLRU()`: Unlink head + delete from map, O(1)

**LRU Eviction**

The blacklist has a capacity (`BLACKLIST_CAP = 10,000,000`). If full when adding a new address, the least-recently-used entry is evicted:

```cpp
void add(void* addr, FILE* logFile = nullptr) {
  if (map.size() >= cap) {
    evictLRU(logFile);  // Remove oldest entry
  }
  // Add new entry at tail
  ...
}
```

Evicting an entry with non-zero reference count risks false parent-child matches (the evicted address may be reused while children still reference it). The profiler logs a warning when this occurs:

```
[WARNING] Blacklist cap reached (10000000), evicting address 0x1000 with refCount=1.
          Potential false parent-child match may occur.
```

The LRU strategy minimizes this risk by evicting the oldest entries (most likely to have completed children) rather than random or newest entries.

#### 2.3.4 How Pools and Blacklist Work Together

During `EventPool::allocate()`:

```cpp
for (size_t attempt = 0; attempt < totalSlots; attempt++) {
  size_t chunkIdx = ...;
  size_t slotInChunk = ...;
  
  if (!inUseChunks[chunkIdx][slotInChunk]) {
    MyEvent* event = &chunks[chunkIdx][slotInChunk];
    void* addr = static_cast<void*>(event);
    
    if (!blacklist.contains(addr)) {
      // This slot is free AND not blacklisted - use it
      inUseChunks[chunkIdx][slotInChunk] = true;
      ...
      return event;
    }
    // Slot is free but blacklisted - skip to next slot
  }
}
```

The allocator skips blacklisted addresses even if they're marked as free in the pool. This ensures an address isn't reused while still referenced as a parent.

**Complete Flow Example:**

```
Time 1: Parent event allocated at slot 5 (address 0x5000)
Time 2: Child event starts with parentObj=0x5000
        → 0x5000 added to blacklist (refCount=1)
Time 3: Parent event completes, slot 5 marked free in pool
Time 4: New allocation request comes in
        → Round-robin reaches slot 5
        → Slot 5 is free but 0x5000 is in blacklist
        → Skip slot 5, continue to slot 6
Time 5: Child event completes
        → Decrement 0x5000 refCount to 0
        → 0x5000 removed from blacklist
Time 6: Later allocation can now use slot 5
```

### 2.4 Cross-Process Events (PXN)

#### 2.4.1 What is PXN Routing

PXN (PCI x NVLink) is an NCCL optimization for multi-node communication. When data must travel between nodes, NCCL can route it through GPUs connected via NVLink rather than having each GPU communicate directly over the network.

In PXN routing, a proxy operation may be executed by a different process than the one that originated the NCCL collective. Process A submits a collective, but the actual network transfer is handled by Process B's proxy thread.

#### 2.4.2 The Invalid Pointer Problem

When PXN routing occurs, NCCL calls the profiler's functions in Process B (the executing process), but passes:
- `context`: Points to Process A's memory (the originating process)
- `eDescr->parentObj`: Points to Process A's memory

These pointers are invalid in Process B's address space. Process B cannot dereference them - doing so would cause a segmentation fault or read garbage data.

The profiler must:
1. Detect when this situation occurs
2. Handle the event without dereferencing invalid pointers

#### 2.4.3 Context Validation

The profiler maintains a **context registry** - a hash map of all valid context pointers created by this process:

```cpp
static std::unordered_map<void*, MyContext*> contextRegistry;
static pthread_mutex_t contextRegistryMutex;
```

When `myInit` creates a context, it registers it:

```cpp
registerContext(ctx);  // Adds ctx to contextRegistry
```

When `myFinalize` destroys a context, it unregisters it:

```cpp
unregisterContext(ctx);  // Removes ctx from contextRegistry
```

In `myStartEvent`, the profiler checks if the context is valid:

```cpp
bool contextValid = isValidContext(context);

if (eDescr->type == ncclProfileProxyOp) {
  pid_t eventPid = eDescr->proxyOp.pid;
  if (eventPid != myPid) {
    contextValid = false;  // Different process - definitely PXN
  }
}
```

For `ncclProfileProxyOp` events, NCCL provides the originating process's PID in `eDescr->proxyOp.pid`. If this doesn't match our PID, we know the event originated elsewhere.

For other event types, we rely solely on the context registry check.

#### 2.4.4 The Detach Pool

Events with invalid contexts are called "detached" events. They cannot use per-context resources because the context pointer cannot be dereferenced.

Instead, detached events use process-wide resources:

**Process-Wide State:**

```cpp
static EventPool* detachPool = nullptr;       // Shared pool for all detached events
static Blacklist* detachBlacklist = nullptr;  // Shared blacklist for detach pool
static FILE* detachLogFile = nullptr;         // Separate log file for PXN events
static pthread_mutex_t detachMutex;           // Protects pool and blacklist
static pthread_mutex_t detachLogMutex;        // Protects log file
static double detachStartTime = 0;            // Time base for PXN log timestamps
```

**Initialization and Cleanup:**

The detach pool is initialized when the first rank in a process calls `myInit`:

```cpp
pthread_mutex_lock(&processInitMutex);
if (processInitCount == 0) {
  myPid = getpid();
  initDetachPool();      // Create pool and blacklist
  initPxnLogFile(...);   // Open PXN log file
}
processInitCount++;
pthread_mutex_unlock(&processInitMutex);
```

It's cleaned up when the last rank calls `myFinalize`:

```cpp
pthread_mutex_lock(&processInitMutex);
processInitCount--;
if (processInitCount == 0) {
  cleanupDetachPool();  // Delete pool, blacklist, close log
}
pthread_mutex_unlock(&processInitMutex);
```

**Detached Event Handling:**

In `myStartEvent`, if the context is invalid:

```cpp
if (!contextValid) {
  event = allocateFromDetachPool(detachLogFile);
  event->ctx = nullptr;  // Don't store invalid context
  event->originPid = eventPid;  // Store originating PID for logging
  trackDetachParent(eDescr->parentObj, detachLogFile);  // Blacklist parent
}
```

In `myStopEvent`, detached events are handled differently:

```cpp
if (event->fromDetachPool) {
  // Log to detachLogFile instead of ctx->logFile
  releaseDetachParent(event->parentObj);  // Decrement blacklist ref
  freeToDetachPool(event);  // Return to detach pool
  return ncclSuccess;
}
```

**ParentObj Handling:**

For detached events, `parentObj` is an address in another process's memory. We cannot dereference it, but we still:
1. Log its value (as a pointer for correlation with the originating process's logs)
2. Track it in the detach blacklist (the address value itself, not what it points to)

This allows post-processing tools to match events across processes by correlating `parentObj` values.

---

## 3. GPU Identification

The profiler captures GPU information during `myInit` to associate events with physical hardware.

**CUDA Device ID**

The device ID is obtained via `cudaGetDevice()`:

```cpp
cudaError_t cudaErr = cudaGetDevice(&ctx->cudaDev);
```

This returns the device index (0, 1, 2, ...) for the GPU assigned to this rank. However, device IDs are process-local and affected by `CUDA_VISIBLE_DEVICES`. Device 0 in one process might be a different physical GPU than device 0 in another process.

**GPU UUID**

For reliable cross-process GPU identification, the profiler extracts the GPU UUID:

```cpp
cudaDeviceProp deviceProp;
cudaGetDeviceProperties(&deviceProp, ctx->cudaDev);
snprintf(ctx->gpuUuid, sizeof(ctx->gpuUuid),
         "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
         deviceProp.uuid.bytes[0], deviceProp.uuid.bytes[1],
         // ... remaining 14 bytes
);
```

The UUID is a 128-bit identifier that is:
- Globally unique across all NVIDIA GPUs
- Consistent regardless of `CUDA_VISIBLE_DEVICES`
- Consistent across processes and reboots

The formatted UUID string (36 characters: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`) is stored in `MyContext::gpuUuid` and included in every log entry.

**Error Handling**

If CUDA queries fail, the profiler continues with degraded functionality:
- Device ID set to -1
- UUID left as empty string
- Logs show `(unknown)` for GPU identification

---

## 4. Communicator Information

The profiler receives communicator information as parameters to `myInit`:

| Parameter | Type | Description |
|-----------|------|-------------|
| `commId` | `uint64_t` | Unique identifier for this communicator instance |
| `rank` | `int` | This rank's position within the communicator (0 to nranks-1) |
| `nranks` | `int` | Total number of ranks in the communicator |
| `commName` | `const char*` | Optional human-readable name for the communicator |
| `nNodes` | `int` | Number of nodes participating in the communicator |

These values are stored in `MyContext`:

```cpp
ctx->commId = commId;
ctx->rank = rank;
ctx->nranks = nranks;
```

**Usage in Logging:**

The `commId` and `rank` are used in log file naming:
```
minimal_profiler_rank{rank}_comm{commId}.log
```

The log file header includes all communicator details:
```
Rank: 0/4, CommId: 12345678, CommName: my_comm, ctx=0x1234
```

---

## 5. Process and Thread Tracking

**Process ID (PID)**

The profiler tracks its own PID for several purposes:

```cpp
static pid_t myPid = 0;  // Set during first rank init

// In myInit:
if (processInitCount == 0) {
  myPid = getpid();
}
```

PID is used for:
1. **PXN Detection**: Compare `eDescr->proxyOp.pid` with `myPid` to detect cross-process events
2. **Log File Naming**: PXN log file includes PID: `minimal_profiler_pxn_events_pid{pid}_comm{commId}.log`
3. **Log Entries**: Each log entry includes the current PID for process identification

**Thread ID (TID)**

Thread ID is obtained via the Linux-specific `syscall(SYS_gettid)`:

```cpp
static inline pid_t getTid() {
  return static_cast<pid_t>(syscall(SYS_gettid));
}
```

TID is included in every log entry, allowing identification of which thread generated each event. This is important because NCCL uses multiple threads (application threads, proxy threads, etc.) that all call into the profiler.

**Process-Wide State Management**

Some resources are shared across all ranks within a process:
- Detach pool and blacklist
- PXN log file
- Context registry

The profiler tracks how many ranks have initialized using `processInitCount`:

```cpp
static int processInitCount = 0;  // Protected by processInitMutex

// In myInit:
pthread_mutex_lock(&processInitMutex);
if (processInitCount == 0) {
  // First rank - initialize process-wide resources
  initDetachPool();
  initPxnLogFile(...);
}
processInitCount++;
pthread_mutex_unlock(&processInitMutex);

// In myFinalize:
pthread_mutex_lock(&processInitMutex);
processInitCount--;
if (processInitCount == 0) {
  // Last rank - clean up process-wide resources
  cleanupDetachPool();
}
pthread_mutex_unlock(&processInitMutex);
```

This ensures process-wide resources exist while any rank is active, and are cleaned up when the last rank finalizes.

---

## 6. Logging

### 6.1 Per-Rank Log Files

Each rank gets its own log file, created during `myInit`:

```cpp
char filename[512];
snprintf(filename, sizeof(filename), "%s/minimal_profiler_rank%d_comm%lu.log", 
         dirname, rank, commId);
ctx->logFile = fopen(filename, "w");
```

**File Name Format:**
```
{dump_dir}/minimal_profiler_rank{rank}_comm{commId}.log
```

**Log File Header:**

```
=== Profiler initialized ===
Init thread ID (tid): 12345
Start time: 1234567890123 us
Dump directory: minimal_profiler_dump-20250109-143022
Event activation mask: 0xfff
Event pool size: 1024
Blacklist cap: 10000000
Physical GPU device ID: 0
Physical GPU UUID: 12345678-1234-1234-1234-123456789abc
Rank: 0/4, CommId: 12345678, CommName: (null), ctx=0x1234
```

**Log File Footer:**

```
[1234567.890] === Profiler finalized (tid=12345, totalDuration=1234567.890 us = 1.234568 s, ctx=0x1234) ===
Final pool size: 2 chunks, 2048 total slots
Final blacklist size: 0
```

### 6.2 PXN Log File

A separate process-wide log file captures detached PXN events:

```cpp
snprintf(pxnLogPath, sizeof(pxnLogPath), "%s/minimal_profiler_pxn_events_pid%d_comm%lu.log", 
         dirname, static_cast<int>(myPid), commId);
detachLogFile = fopen(pxnLogPath, "w");
```

**File Name Format:**
```
{dump_dir}/minimal_profiler_pxn_events_pid{pid}_comm{commId}.log
```

**PXN Log Header:**

```
=== PXN Events Log (Process PID: 12345, CommId: 12345678) ===
CommId: 12345678
Start time: 1234567890123 us
This file contains ProxyOp events from other processes (PXN routing)
================================================
```

### 6.3 Log Entry Format

**START Entries (Normal Events):**

```
[timestamp] START type (pid=X, tid=Y, gpu_uuid=Z, type-specific-fields, parentObj=P, event=E, ctx=C)
```

Example:
```
[123.456] START ncclProfileProxyOp (pid=12345, tid=12346, gpu_uuid=12345678-..., channel=0, peer=1, steps=8, chunkSize=524288, isSend=1, eventPid=12345, parentObj=0x1234, event=0x5678, ctx=0x9abc)
```

**START Entries (PXN Events):**

```
[timestamp] START type (pid=X, tid=Y, PXN from pid=Z, type-specific-fields, parentObj=P, event=E)
```

Note: PXN events don't include `gpu_uuid` or `ctx` since those aren't available.

**STATE Entries:**

```
[timestamp] STATE type::func -> state (pid=X, tid=Y, gpu_uuid=Z, stateId=S, state-args, event=E, ctx=C)
```

Example:
```
[234.567] STATE ncclProfileProxyStep::ProxyStep -> ProxyStepSendWait (pid=12345, tid=12346, gpu_uuid=12345678-..., stateId=9, transSize=524288, event=0x5678, ctx=0x9abc)
```

**STOP Entries:**

```
[timestamp] STOP type::func (pid=X, tid=Y, gpu_uuid=Z, duration=D us, event=E, ctx=C)
```

Example:
```
[345.678] STOP ncclProfileProxyOp::ProxyOp (pid=12345, tid=12346, gpu_uuid=12345678-..., duration=222.222 us, event=0x5678, ctx=0x9abc)
```

**Timestamps:**

All timestamps are in microseconds, relative to the context's initialization time (`ctx->startTime`). For PXN events, timestamps are relative to `detachStartTime`.

### 6.4 Directory Naming

Log files are written to a dump directory. The directory name is determined by:

1. **SLURM Environment**: If `SLURM_JOB_ID` is set, use it:
   ```
   minimal_profiler_dump-{SLURM_JOB_ID}
   ```

2. **Timestamp Fallback**: Otherwise, generate from current time:
   ```
   minimal_profiler_dump-{YYYYMMDD-HHMMSS}
   ```

The directory is created if it doesn't exist (`mkdir` with mode 0755). The creation is safe to call multiple times (checks for existence first).

---

## 7. Thread Safety

### 7.1 NCCL's Threading Model

NCCL may call profiler functions from multiple threads concurrently:

1. **Application Threads**: User code calling NCCL collectives (`ncclAllReduce`, etc.)
2. **Proxy Threads**: Internal NCCL threads handling network communication
3. **Network Plugin Threads**: Threads from network plugins (e.g., UCX)

Within a single rank, multiple threads may call `myStartEvent`, `myStopEvent`, and `myRecordEventState` concurrently. The profiler must handle this safely.

Additionally, a single process may have multiple ranks (when using `ncclCommInitAll` or threading patterns), each with its own context but sharing process-wide resources.

### 7.2 Mutexes

The profiler uses six mutexes to protect shared resources:

| Mutex | Scope | Protects |
|-------|-------|----------|
| `logMutex` | Per-context (`MyContext`) | Log file I/O operations |
| `allocMutex` | Per-context (`MyContext`) | Pool allocation/free and blacklist operations |
| `detachMutex` | Process-wide (static) | Detach pool and detach blacklist |
| `detachLogMutex` | Process-wide (static) | PXN log file I/O |
| `processInitMutex` | Process-wide (static) | `processInitCount`, detach pool init/cleanup |
| `contextRegistryMutex` | Process-wide (static) | Context registry hash map |

**Per-Context Mutexes:**

```cpp
struct MyContext {
  pthread_mutex_t logMutex;     // Protects logFile
  pthread_mutex_t allocMutex;   // Protects pool and blacklist
  // ...
};
```

Each context has its own mutexes, so operations on different contexts don't contend.

**Usage Pattern:**

```cpp
// Logging
pthread_mutex_lock(&ctx->logMutex);
fprintf(ctx->logFile, ...);
fflush(ctx->logFile);
pthread_mutex_unlock(&ctx->logMutex);

// Allocation
pthread_mutex_lock(&ctx->allocMutex);
event = ctx->pool.allocate(ctx->blacklist, ctx->logFile);
pthread_mutex_unlock(&ctx->allocMutex);
```

### 7.3 Lock Ordering

To prevent deadlocks, locks are acquired in a consistent order when multiple locks are needed:

1. `processInitMutex` (outermost - protects initialization/cleanup)
2. `contextRegistryMutex` (context validation)
3. `allocMutex` or `detachMutex` (resource allocation)
4. `logMutex` or `detachLogMutex` (innermost - I/O operations)

In practice, most operations only need one or two locks. The ordering matters primarily during initialization and finalization when multiple resources are accessed.

**Critical Sections:**

The profiler minimizes time spent holding locks:
- Allocation is fast (O(1) for both pool and blacklist)
- Log writes are buffered and flushed atomically
- Context validation is a single hash map lookup

---

## 8. Reference

### 8.1 Data Structures

#### BlacklistNode

Doubly-linked list node for LRU tracking within the blacklist.

```cpp
struct BlacklistNode {
  void* address;           // The blacklisted memory address
  int refCount;            // Number of active children referencing this address
  BlacklistNode* prev;     // Previous node (toward head/older)
  BlacklistNode* next;     // Next node (toward tail/newer)
};
```

**Invariants:**
- `refCount >= 1` while in the list (removed when it reaches 0)
- `prev == nullptr` only for head node
- `next == nullptr` only for tail node

#### Blacklist

Hash map + doubly-linked list for O(1) LRU blacklist operations.

```cpp
struct Blacklist {
  std::unordered_map<void*, BlacklistNode*> map;  // O(1) address lookup
  BlacklistNode* head;    // LRU end (oldest, evict from here)
  BlacklistNode* tail;    // MRU end (newest, add/move here)
  size_t cap;             // Maximum entries before forced eviction
};
```

**Key Methods:**
- `contains(addr)`: Check if address is blacklisted (O(1))
- `add(addr, logFile)`: Add new address at tail, evict if at capacity (O(1))
- `incrementRef(addr)`: Increment reference count, move to tail (O(1))
- `decrementRef(addr)`: Decrement reference count, remove if zero (O(1))
- `evictLRU(logFile)`: Remove head node (O(1))
- `moveToTail(node)`: Internal helper to update LRU position (O(1))

#### MyEvent

Stores metadata for a single profiled event.

```cpp
struct MyEvent {
  uint64_t type;          // Event type enum value
  double startTime;       // When the event started (microseconds)
  const char* func;       // Function/operation name (e.g., "AllReduce")
  const char* typeName;   // Type name string (e.g., "ncclProfileColl")
  MyContext* ctx;         // Owning context (nullptr for PXN events)
  void* parentObj;        // Parent event address (for blacklist tracking)
  bool fromPool;          // True if from pool, false if dynamically allocated
  bool fromDetachPool;    // True if from process-wide detach pool
  pid_t originPid;        // PID that originated this event (for PXN tracking)
};
```

**Lifecycle:**
1. Allocated by `EventPool::allocate()` or `allocateFromDetachPool()`
2. Fields populated in `myStartEvent()`
3. Read in `myRecordEventState()` and `myStopEvent()`
4. Freed by `EventPool::free()` or `freeToDetachPool()`

#### EventPool

Chunked memory pool with round-robin allocation.

```cpp
struct EventPool {
  std::vector<MyEvent*> chunks;      // Array of chunk pointers
  std::vector<bool*> inUseChunks;    // Parallel array of in-use flag arrays
  size_t chunkSize;                   // Slots per chunk (fixed after init)
  size_t nextChunkIndex;              // Round-robin: which chunk to try next
  size_t nextSlotInChunk;             // Round-robin: which slot in chunk
};
```

**Key Methods:**
- `init(initialChunkSize)`: Allocate first chunk, set chunk size
- `allocate(blacklist, logFile)`: Find free non-blacklisted slot (O(n) worst case, typically O(1))
- `free(event)`: Mark slot as not in use (O(1))
- `grow(logFile)`: Allocate new chunk (O(1))
- `contains(event)`: Check if pointer belongs to this pool (O(chunks))
- `getIndex(event)`: Get linearized slot index (O(chunks))

#### MyContext

Per-rank profiler state.

```cpp
struct MyContext {
  FILE* logFile;                   // Log file handle
  pthread_mutex_t logMutex;        // Protects logFile I/O
  pthread_mutex_t allocMutex;      // Protects pool and blacklist
  uint64_t commId;                 // Communicator ID
  int rank;                        // Rank within communicator
  int nranks;                      // Total ranks in communicator
  double startTime;                // Initialization timestamp (microseconds)
  int cudaDev;                     // CUDA device ID (-1 if unknown)
  char gpuUuid[37];                // GPU UUID string (empty if unknown)
  EventPool pool;                  // Per-context event pool
  Blacklist blacklist;             // Per-context address blacklist
};
```

### 8.2 API Functions

#### myInit

```cpp
ncclResult_t myInit(void** context, uint64_t commId, int* eActivationMask,
                    const char* commName, int nNodes, int nranks, int rank,
                    ncclDebugLogger_t logfn)
```

**When Called**: During NCCL communicator initialization (`ncclCommInit*` functions).

**Parameters:**
- `context` [out]: Pointer where profiler stores its context
- `commId`: Unique identifier for this communicator
- `eActivationMask` [out]: Bitmask of event types to enable
- `commName`: Optional communicator name (may be null)
- `nNodes`: Number of nodes in communicator
- `nranks`: Total ranks in communicator
- `rank`: This rank's index (0 to nranks-1)
- `logfn`: NCCL debug logger callback (unused in this implementation)

**Actions:**
1. Parse `NCCL_PROFILE_EVENT_MASK` environment variable
2. Create dump directory
3. Initialize process-wide state (first rank only): PID, detach pool, PXN log
4. Allocate and initialize `MyContext`
5. Query CUDA device ID and UUID
6. Initialize mutexes, pool, and blacklist
7. Open per-rank log file and write header
8. Register context in process-wide registry

**Return Values:**
- `ncclSuccess`: Initialization successful
- `ncclSystemError`: Memory allocation failed, directory creation failed, or file open failed

#### myStartEvent

```cpp
ncclResult_t myStartEvent(void* context, void** eHandle, 
                          ncclProfilerEventDescr_v5_t* eDescr)
```

**When Called**: When a profilable event begins.

**Parameters:**
- `context`: Profiler context from `myInit` (may be invalid for PXN events)
- `eHandle` [out]: Pointer where profiler stores event handle
- `eDescr`: Event descriptor with type and type-specific fields

**Actions:**
1. Validate context (check registry, compare PIDs for ProxyOp)
2. Allocate `MyEvent` from appropriate pool (per-context or detach)
3. Populate event fields (type, timestamp, function name, parent pointer)
4. Add parent address to blacklist (if parentObj is set)
5. Extract type-specific information and log START entry

**Return Values:**
- `ncclSuccess`: Always (allocation failures set `eHandle` to nullptr)

#### myStopEvent

```cpp
ncclResult_t myStopEvent(void* eHandle)
```

**When Called**: When an event completes.

**Parameters:**
- `eHandle`: Event handle from `myStartEvent`

**Actions:**
1. Calculate duration (current time - start time)
2. Log STOP entry (to appropriate log file)
3. Decrement parent's blacklist reference count
4. Free event back to pool (or delete if dynamically allocated)

**Return Values:**
- `ncclSuccess`: Always

#### myRecordEventState

```cpp
ncclResult_t myRecordEventState(void* eHandle, 
                                ncclProfilerEventState_v5_t eState,
                                ncclProfilerEventStateArgs_v5_t* eStateArgs)
```

**When Called**: During event execution when state changes occur.

**Parameters:**
- `eHandle`: Event handle from `myStartEvent`
- `eState`: New state enum value
- `eStateArgs`: Optional state-specific arguments (may be null)

**Actions:**
1. Map state enum to human-readable name
2. Log STATE entry with timestamp and state-specific args
3. Handle both normal and PXN events (different log files)

**Return Values:**
- `ncclSuccess`: Always

#### myFinalize

```cpp
ncclResult_t myFinalize(void* context)
```

**When Called**: During NCCL communicator destruction (`ncclCommDestroy`).

**Parameters:**
- `context`: Profiler context from `myInit`

**Actions:**
1. Unregister context from process-wide registry
2. Log finalization footer (duration, final pool/blacklist size)
3. Close log file
4. Destroy mutexes
5. Delete `MyContext` (destructor cleans up pool and blacklist)
6. Decrement process init count; clean up detach pool if last rank

**Return Values:**
- `ncclSuccess`: Always

### 8.3 Event Types

| Type Constant | Value | Description | Key Fields |
|---------------|-------|-------------|------------|
| `ncclProfileGroupApi` | 0x001 | Group API boundaries | `groupDepth`, `graphCaptured` |
| `ncclProfileCollApi` | 0x002 | Collective API call | `func`, `count`, `datatype`, `root`, `stream` |
| `ncclProfileP2pApi` | 0x004 | P2P API call | `func`, `count`, `datatype`, `stream` |
| `ncclProfileKernelLaunch` | 0x008 | CUDA kernel launch | `stream` |
| `ncclProfileColl` | 0x010 | Collective execution | `func`, `algo`, `proto`, `nChannels`, `count` |
| `ncclProfileP2p` | 0x020 | P2P execution | `func`, `peer`, `nChannels`, `count` |
| `ncclProfileProxyOp` | 0x040 | Proxy operation | `channelId`, `peer`, `nSteps`, `pid` |
| `ncclProfileProxyStep` | 0x080 | Proxy step | `step` |
| `ncclProfileProxyCtrl` | 0x100 | Proxy control | (none) |
| `ncclProfileKernelCh` | 0x200 | Kernel channel | `channelId`, `pTimer` |
| `ncclProfileNetPlugin` | 0x400 | Network plugin event | `id`, `data` |
| `ncclProfileGroup` | 0x800 | Group management | (none) |

**Event Hierarchy Examples:**

```
ncclProfileGroupApi
└── ncclProfileCollApi (ncclAllReduce)
    ├── ncclProfileKernelLaunch
    └── ncclProfileColl (AllReduce)
        ├── ncclProfileProxyOp (send)
        │   ├── ncclProfileProxyStep (0)
        │   └── ncclProfileProxyStep (1)
        └── ncclProfileProxyOp (recv)
            └── ...
```

### 8.4 Configuration

**Environment Variables:**

| Variable | Purpose | Default |
|----------|---------|---------|
| `NCCL_PROFILE_EVENT_MASK` | Bitmask of enabled event types | 4095 (0xFFF, all types) |
| `SLURM_JOB_ID` | Used for dump directory naming | (none) |

**Compile-Time Constants:**

| Constant | Value | Purpose |
|----------|-------|---------|
| `INITIAL_POOL_SIZE` | 1024 | Initial slots per per-context pool chunk |
| `INITIAL_DETACH_POOL_SIZE` | 256 | Initial slots per detach pool chunk |
| `BLACKLIST_CAP` | 10,000,000 | Maximum entries in any blacklist |

**Configuration Value Trade-offs:**

- **Pool Size**: Larger pools reduce growth frequency but consume more memory. Smaller pools may grow frequently under high event load.

- **Blacklist Cap**: Higher caps prevent forced eviction (and potential false parent matches) but consume more memory. The 10M default is chosen to handle typical workloads without eviction.

- **Event Mask**: Fewer event types mean less overhead but less visibility. The default (all types) provides complete tracing at the cost of higher overhead.
