/*************************************************************************
 * Minimal NCCL Profiler Plugin Example (C++ version)
 * 
 * This profiler plugin uses memory pools and blacklisting to prevent
 * address reuse that would obfuscate parent-child event relationships.
 * 
 * Key features:
 * - Single process-wide event pool with round-robin allocation to maximize time between address reuse
 * - Process-wide blacklist with LRU eviction to prevent reuse of addresses still referenced as parents
 * - O(1) operations for all blacklist operations via hash map + doubly linked list
 * - PXN (PCI x NVLink) support for cross-process ProxyOp events (same pool as normal events)
 * - Kernel record correlation with nccl events via CUPTI API 
 *
 * Code Organization:
 * 1. Configuration constants
 * 2. Util functions
 * 3. Data structures
 * 4. Process resources requiring thread safety
 * 5. Process pool, blacklist, and PXN handling
 * 6. CUPTI activity record tracking
 * 7. Logging
 * 8. Profiler API implementation
 * 9. Export struct
 ************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cstddef>
#include <unistd.h>
#include <cstdint>
#include <cstdarg>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <cerrno>
#include <pthread.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <cstring>

// Include CUDA runtime for GPU UUID queries
#include <cuda_runtime.h>

// Include CUPTI for correlation ID tracking
#include <cupti.h>
#include <atomic>

// Include the profiler API headers (C headers)
extern "C" {
#include "nccl/common.h"
#include "nccl/err.h"
#include "nccl/profiler.h"
#include "nccl/types.h"
}

// ============================================================================
// CUPTI Error Checking Macro
// ============================================================================
#ifndef CUPTI_API_CALL
#define CUPTI_API_CALL(apiFunctionCall)                                             \
do {                                                                                \
    CUptiResult _status = apiFunctionCall;                                          \
    if (_status != CUPTI_SUCCESS) {                                                 \
        const char *pErrorString;                                                   \
        cuptiGetResultString(_status, &pErrorString);                               \
        fprintf(stderr, "[MinimalProfiler] WARNING: %s:%d: Function %s failed with error(%d): %s\n", \
                __FILE__, __LINE__, #apiFunctionCall, _status,                      \
                pErrorString ? pErrorString : "Unknown error");                     \
    }                                                                               \
} while (0)
#endif

// ============================================================================
// SECTION 1: Configuration Constants
// ============================================================================

// Environment variable names
static const char* ENV_EVENT_MASK = "NCCL_PROFILE_EVENT_MASK";

// Pool sizing (single process-wide pool for all events including PXN)
static const size_t INITIAL_POOL_SIZE = 256;

// Blacklist sizing
static const size_t BLACKLIST_CAP = 10000000;            // 10 million entries max

// CUPTI activity record requested buffer sizing
static const size_t CUPTI_BUFFER_SIZE = 1024 * 1024 * 8; // 8MB buffers

// Forward declarations
static bool initProcessPool();
static bool initJsonlTraceFile(const char* dirname);
static bool initCuptiActivity();
static void writeJsonlCuptiActivityRecord(uint32_t kind, uint64_t startTime, uint64_t endTime,
                                          uint32_t correlationId, uint64_t externalCorrelationId,
                                          uint32_t deviceId, const char* name, void* stream,
                                          size_t size, const std::string& detailsJson);
static void cleanupCuptiActivity();
static void closeSharedJsonlFile();
static std::string escapeJsonStringStd(const char* input);


// ============================================================================
// SECTION 2: Utility Functions
// ============================================================================

// Get current cpu timestamp in microseconds
static double getTime() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

// Get thread ID
static inline pid_t getTid() {
  return static_cast<pid_t>(syscall(SYS_gettid));
}

// Get GPU UUID from current CUDA device
static std::string getGpuUuid() {
  int dev;
  if (cudaGetDevice(&dev) != cudaSuccess) {
    return "";
  }
  
  cudaDeviceProp deviceProp;
  if (cudaGetDeviceProperties(&deviceProp, dev) != cudaSuccess) {
    return "";
  }
  
  // Format UUID as standard string: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
  char uuidStr[37];
  snprintf(uuidStr, sizeof(uuidStr),
           "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
           deviceProp.uuid.bytes[0], deviceProp.uuid.bytes[1],
           deviceProp.uuid.bytes[2], deviceProp.uuid.bytes[3],
           deviceProp.uuid.bytes[4], deviceProp.uuid.bytes[5],
           deviceProp.uuid.bytes[6], deviceProp.uuid.bytes[7],
           deviceProp.uuid.bytes[8], deviceProp.uuid.bytes[9],
           deviceProp.uuid.bytes[10], deviceProp.uuid.bytes[11],
           deviceProp.uuid.bytes[12], deviceProp.uuid.bytes[13],
           deviceProp.uuid.bytes[14], deviceProp.uuid.bytes[15]);
  
  return std::string(uuidStr);
}

// Get human-readable event type name
static const char* getEventTypeName(uint64_t type) {
  switch (type) {
    case ncclProfileGroupApi:     return "ncclProfileGroupApi";
    case ncclProfileCollApi:      return "ncclProfileCollApi";
    case ncclProfileP2pApi:       return "ncclProfileP2pApi";
    case ncclProfileKernelLaunch: return "ncclProfileKernelLaunch";
    case ncclProfileColl:         return "ncclProfileColl";
    case ncclProfileP2p:          return "ncclProfileP2p";
    case ncclProfileProxyOp:      return "ncclProfileProxyOp";
    case ncclProfileProxyStep:    return "ncclProfileProxyStep";
    case ncclProfileProxyCtrl:    return "ncclProfileProxyCtrl";
    case ncclProfileKernelCh:     return "ncclProfileKernelCh";
    case ncclProfileNetPlugin:    return "ncclProfileNetPlugin";
    case ncclProfileGroup:        return "ncclProfileGroup";
    default:                      return "ncclProfileUnknown";
  }
}

// Get human-readable state name
static const char* getStateName(ncclProfilerEventState_v5_t eState) {
  switch (eState) {
    case ncclProfilerProxyOpSendPosted:        return "ProxyOpSendPosted (deprecated)";
    case ncclProfilerProxyOpSendRemFifoWait:   return "ProxyOpSendRemFifoWait (deprecated)";
    case ncclProfilerProxyOpSendTransmitted:   return "ProxyOpSendTransmitted (deprecated)";
    case ncclProfilerProxyOpSendDone:          return "ProxyOpSendDone (deprecated)";
    case ncclProfilerProxyOpRecvPosted:        return "ProxyOpRecvPosted (deprecated)";
    case ncclProfilerProxyOpRecvReceived:      return "ProxyOpRecvReceived (deprecated)";
    case ncclProfilerProxyOpRecvTransmitted:   return "ProxyOpRecvTransmitted (deprecated)";
    case ncclProfilerProxyOpRecvDone:          return "ProxyOpRecvDone (deprecated)";
    case ncclProfilerProxyStepSendGPUWait:     return "ProxyStepSendGPUWait";
    case ncclProfilerProxyStepSendWait:        return "ProxyStepSendWait";
    case ncclProfilerProxyStepRecvWait:        return "ProxyStepRecvWait";
    case ncclProfilerProxyStepRecvFlushWait:   return "ProxyStepRecvFlushWait";
    case ncclProfilerProxyStepRecvGPUWait:     return "ProxyStepRecvGPUWait";
    case ncclProfilerProxyCtrlIdle:            return "ProxyCtrlIdle";
    case ncclProfilerProxyCtrlActive:          return "ProxyCtrlActive";
    case ncclProfilerProxyCtrlSleep:           return "ProxyCtrlSleep";
    case ncclProfilerProxyCtrlWakeup:          return "ProxyCtrlWakeup";
    case ncclProfilerProxyCtrlAppend:          return "ProxyCtrlAppend";
    case ncclProfilerProxyCtrlAppendEnd:       return "ProxyCtrlAppendEnd";
    case ncclProfilerProxyOpInProgress_v4:     return "ProxyOpInProgress";
    case ncclProfilerProxyStepSendPeerWait_v4: return "ProxyStepSendPeerWait";
    case ncclProfilerNetPluginUpdate:          return "NetPluginUpdate";
    case ncclProfilerKernelChStop:             return "KernelChStop";
    case ncclProfilerEndGroupApiStart:         return "EndGroupApiStart";
    case ncclProfilerBeginGroupApiEnd:         return "BeginGroupApiEnd";
    default:                                   return "Unknown";
  }
}

// ============================================================================
// SECTION 3: Data Structures
// ============================================================================

// ----------------------------------------------------------------------------
// Forward declarations for data structures
// ----------------------------------------------------------------------------
struct MyContext;

// ----------------------------------------------------------------------------
// MyEvent: Stores event data + parent reference
// ----------------------------------------------------------------------------
struct MyEvent {
  // Basic event info (set at START)
  uint64_t type;
  const char* typeName;
  MyContext* ctx;
  void* parentObj;          // Store parentObj to decrement blacklist ref on stop
  bool fromPool;            // Indicates whether this event was allocated from pool or dynamically
  bool isPxn;               // Indicates whether this event is pxn
  double startTime;
  pid_t startTid;
  pid_t startPid;
  std::string typeDetails;  // Type-specific details JSON
  
  // Context-dependent fields captured at START (safe for PXN - copied, not pointer)
  char gpuUuid[37];       // GPU UUID (empty for PXN events)
  uint64_t commId;        // Communicator ID (0 for PXN events)
  int rank;               // Rank (-1 for PXN events)
  
  // CUPTI external correlation ID tracking
  // This ID links NCCL events to CUDA kernel activities, enabling correlation
  // between communication operations and the kernels that triggered them.
  uint64_t externalCorrelationId;  // Our custom correlation ID for this event
  bool hasExternalCorrelationId;    // Whether we successfully pushed an ID
};

// ----------------------------------------------------------------------------
// BlacklistNode: Node for LRU linked list in Blacklist
// ----------------------------------------------------------------------------
struct BlacklistNode {
  void* address;           // The blacklisted address
  int refCount;            // Number of active children referencing this address
  BlacklistNode* prev;     // LRU list: previous (older)
  BlacklistNode* next;     // LRU list: next (newer)
};

// ----------------------------------------------------------------------------
// Blacklist: Hash map + doubly linked list for O(1) LRU operations
// Tracks addresses that cannot be reused because they are referenced as parents
// ----------------------------------------------------------------------------
struct Blacklist {
  std::unordered_map<void*, BlacklistNode*> map;  // O(1) lookup by address
  BlacklistNode* head;    // LRU end (oldest, evict from here)
  BlacklistNode* tail;    // MRU end (newest, add/move here)
  size_t cap;             // Maximum entries before forced eviction
  
  Blacklist() : head(nullptr), tail(nullptr), cap(BLACKLIST_CAP) {}
  
  ~Blacklist() {
    BlacklistNode* current = head;
    while (current) {
      BlacklistNode* next = current->next;
      delete current;
      current = next;
    }
  }
  
  // Check if address is blacklisted
  bool contains(void* addr) const {
    return map.find(addr) != map.end();
  }
  
  // Move node to tail (mark as most recently used)
  void moveToTail(BlacklistNode* node) {
    if (node == tail) return;  // Already at tail
    
    // Unlink from current position
    if (node->prev) {
      node->prev->next = node->next;
    } else {
      head = node->next;  // Node was head
    }
    if (node->next) {
      node->next->prev = node->prev;
    }
    
    // Link at tail
    node->prev = tail;
    node->next = nullptr;
    if (tail) {
      tail->next = node;
    }
    tail = node;
    if (!head) {
      head = node;
    }
  }
  
  // Evict LRU entry (called when at cap)
  void evictLRU() {
    if (!head) return;
    
    BlacklistNode* node = head;
    
    // Log warning about forced eviction to stderr
    fprintf(stderr, "[MinimalProfiler] WARNING: Blacklist cap reached (%zu), evicting address %p with refCount=%d. "
              "Potential false parent-child match may occur.\n",
              cap, node->address, node->refCount);
    
    // Unlink head
    head = node->next;
    if (head) {
      head->prev = nullptr;
    } else {
      tail = nullptr;
    }
    
    // Remove from map and delete
    map.erase(node->address);
    delete node;
  }
  
  // Add new address to blacklist (at tail)
  void add(void* addr) {
    // Check if we need to evict
    if (map.size() >= cap) {
      evictLRU();
    }
    
    BlacklistNode* node = new BlacklistNode();
    node->address = addr;
    node->refCount = 1;
    node->prev = tail;
    node->next = nullptr;
    
    if (tail) {
      tail->next = node;
    }
    tail = node;
    if (!head) {
      head = node;
    }
    
    map[addr] = node;
  }
  
  // Increment reference count and move to tail
  void incrementRef(void* addr) {
    auto it = map.find(addr);
    if (it != map.end()) {
      it->second->refCount++;
      moveToTail(it->second);  // Update LRU position
    }
  }
  
  // Decrement reference count, remove if zero
  void decrementRef(void* addr) {
    auto it = map.find(addr);
    if (it != map.end()) {
      it->second->refCount--;
      if (it->second->refCount <= 0) {
        // Remove from list
        BlacklistNode* node = it->second;
        if (node->prev) {
          node->prev->next = node->next;
        } else {
          head = node->next;
        }
        if (node->next) {
          node->next->prev = node->prev;
        } else {
          tail = node->prev;
        }
        
        // Remove from map and delete
        map.erase(it);
        delete node;
      }
    }
  }
};

// ----------------------------------------------------------------------------
// EventPool: Round-robin allocation pool for events
// Uses chunked allocation to avoid address reuse while maintaining cache locality
// ----------------------------------------------------------------------------
struct EventPool {
  // Chunked allocation: events are stored in multiple fixed-size chunks
  // Each chunk is allocated separately and remains valid until pool destruction
  std::vector<MyEvent*> chunks;       // Array of chunk pointers
  std::vector<bool*> inUseChunks;     // Array of in-use flag arrays
  size_t chunkSize;                   // Fixed size per chunk
  size_t nextChunkIndex;              // Which chunk to try next (round-robin)
  size_t nextSlotInChunk;             // Which slot in current chunk (round-robin)
  
  // NOTE: All pool operations must be called with appropriate mutex held for thread safety
  
  EventPool() : chunkSize(0), nextChunkIndex(0), nextSlotInChunk(0) {}
  
  ~EventPool() {
    for (size_t i = 0; i < chunks.size(); i++) {
      delete[] chunks[i];
      delete[] inUseChunks[i];
    }
  }
  
  // Initialize pool with given chunk size
  bool init(size_t initialChunkSize) {
    chunkSize = initialChunkSize;
    nextChunkIndex = 0;
    nextSlotInChunk = 0;
    
    // Allocate first chunk
    MyEvent* firstChunk = new (std::nothrow) MyEvent[chunkSize]();
    bool* firstInUse = new (std::nothrow) bool[chunkSize]();
    
    if (!firstChunk || !firstInUse) {
      delete[] firstChunk;
      delete[] firstInUse;
      return false;
    }
    
    // Initialize all slots as not in use
    for (size_t i = 0; i < chunkSize; i++) {
      firstInUse[i] = false;
    }
    
    chunks.push_back(firstChunk);
    inUseChunks.push_back(firstInUse);
    
    return true;
  }
  
  // Allocate a new chunk and append it (never deletes old chunks)
  bool grow() {
    // Allocate a new chunk (same size as existing chunks)
    MyEvent* newChunk = new (std::nothrow) MyEvent[chunkSize]();
    bool* newInUse = new (std::nothrow) bool[chunkSize]();
    
    if (!newChunk || !newInUse) {
      delete[] newChunk;
      delete[] newInUse;
      fprintf(stderr, "[MinimalProfiler] WARNING: Failed to allocate new chunk of size %zu\n", chunkSize);
      return false;
    }
    
    // Initialize all slots as not in use
    for (size_t i = 0; i < chunkSize; i++) {
      newInUse[i] = false;
    }
    
    chunks.push_back(newChunk);
    inUseChunks.push_back(newInUse);
    
    fprintf(stdout, "[MinimalProfiler] INFO: Event pool grown: added chunk %zu (total chunks: %zu, total slots: %zu)\n",
              chunks.size() - 1, chunks.size(), chunks.size() * chunkSize);
    
    return true;
  }
  
  // Check if an event belongs to this pool
  bool contains(MyEvent* event) const {
    if (!event || chunks.empty() || chunkSize == 0) {
      return false;
    }
    
    // Check if event pointer falls within any chunk's address range
    for (size_t chunkIdx = 0; chunkIdx < chunks.size(); chunkIdx++) {
      MyEvent* chunkStart = chunks[chunkIdx];
      MyEvent* chunkEnd = chunkStart + chunkSize;
      
      if (event >= chunkStart && event < chunkEnd) {
        return true;
      }
    }
    
    return false;
  }
  
  // Get slot index for an event (chunkIndex * chunkSize + slotInChunk)
  size_t getIndex(MyEvent* event) const {
    if (!contains(event)) {
      return SIZE_MAX;  // Invalid index
    }
    
    // Find which chunk contains this event
    for (size_t chunkIdx = 0; chunkIdx < chunks.size(); chunkIdx++) {
      MyEvent* chunkStart = chunks[chunkIdx];
      MyEvent* chunkEnd = chunkStart + chunkSize;
      
      if (event >= chunkStart && event < chunkEnd) {
        size_t slotInChunk = static_cast<size_t>(event - chunkStart);
        return chunkIdx * chunkSize + slotInChunk;
      }
    }
    
    return SIZE_MAX;  // Should not reach here if contains() returned true
  }
  
  // Allocate an event from the pool (round-robin across chunks, avoiding blacklisted addresses)
  MyEvent* allocate(Blacklist& blacklist) {
    // Defensive check: ensure we have at least one chunk
    if (chunks.empty() || chunkSize == 0) {
      fprintf(stderr, "[MinimalProfiler] ERROR: EventPool not initialized (chunks.empty() or chunkSize==0)\n");
      return nullptr;
    }
    
    // Round-robin search across all chunks for a free, non-blacklisted slot
    // Start from nextChunkIndex/nextSlotInChunk and wrap around
    size_t totalSlots = chunks.size() * chunkSize;
    size_t startChunk = nextChunkIndex;
    size_t startSlot = nextSlotInChunk;
    
    for (size_t attempt = 0; attempt < totalSlots; attempt++) {
      size_t chunkIdx = (startChunk + (startSlot + attempt) / chunkSize) % chunks.size();
      size_t slotInChunk = (startSlot + attempt) % chunkSize;
      
      if (!inUseChunks[chunkIdx][slotInChunk]) {
        MyEvent* event = &chunks[chunkIdx][slotInChunk];
        void* addr = static_cast<void*>(event);
        
        if (!blacklist.contains(addr)) {
          inUseChunks[chunkIdx][slotInChunk] = true;
          event->fromPool = true;
          
          // Update round-robin pointers for next allocation
          nextSlotInChunk = (slotInChunk + 1) % chunkSize;
          if (nextSlotInChunk == 0) {
            nextChunkIndex = (chunkIdx + 1) % chunks.size();
          } else {
            nextChunkIndex = chunkIdx;
          }
          
          return event;
        }
      }
    }
    
    // No suitable slot found - all are either in use or blacklisted
    // Try to grow the pool by adding a new chunk
    if (grow()) {
      // New chunk is at the end of the vectors
      size_t newChunkIdx = chunks.size() - 1;
      // Start from slot 0 in the new chunk
      size_t slotInChunk = 0;
      
      MyEvent* event = &chunks[newChunkIdx][slotInChunk];
      inUseChunks[newChunkIdx][slotInChunk] = true;
      event->fromPool = true;
      
      // Update round-robin pointers
      nextSlotInChunk = 1;
      nextChunkIndex = newChunkIdx;
      
      return event;
    }
    
    // Pool growth failed - fall back to dynamic allocation
    fprintf(stderr, "[MinimalProfiler] WARNING: Pool exhausted and growth failed, using dynamic allocation\n");
    
    MyEvent* event = new (std::nothrow) MyEvent();
    if (event) {
      event->fromPool = false;
    }
    return event;
  }
  
  // Free an event back to the pool
  void free(MyEvent* event) {
    if (!contains(event)) {
      // Not from pool, caller handles deletion
      return;
    }
    
    // Find which chunk and slot this event belongs to
    for (size_t chunkIdx = 0; chunkIdx < chunks.size(); chunkIdx++) {
      MyEvent* chunkStart = chunks[chunkIdx];
      MyEvent* chunkEnd = chunkStart + chunkSize;
      
      if (event >= chunkStart && event < chunkEnd) {
        size_t slotInChunk = static_cast<size_t>(event - chunkStart);
        inUseChunks[chunkIdx][slotInChunk] = false;
        // Don't reset other fields - they'll be overwritten on next allocation
        return;
      }
    }
  }
};

// ----------------------------------------------------------------------------
// MyContext: Per-communicator plugin state
// ----------------------------------------------------------------------------
struct MyContext {
  uint64_t commId;
  int rank;
  int nranks;
  double startTime;
  char gpuUuid[37];             // Physical GPU UUID string (36 chars + null terminator)
};

// ============================================================================
// SECTION 4: Process resources
// ============================================================================
// These resources are shared across all communicators (contexts) within a single
// process. They are initialized when the first communicator is created and
// cleaned up when the last communicator is finalized.

static pthread_mutex_t processStateMutex = PTHREAD_MUTEX_INITIALIZER;

// JSONL trace file (each process writes to its own file)
static FILE* sharedJsonlFile = nullptr;      // Per-process trace file
static char sharedJsonlPath[512] = {0};      // Path to this process's JSONL file
static pthread_mutex_t jsonlFileMutex = PTHREAD_MUTEX_INITIALIZER;  // Protects writes within this process

// CUPTI Subscriber for Callback API
static CUpti_SubscriberHandle cuptiSubscriber = nullptr;
// track if CUPTI is initialized or cleaned up
static bool cuptiActivityEnabled = false;

// CUPTI external correlation ID tracking
// These correlation IDs link NCCL events to CUDA kernel activities, enabling post-processing
// tools to identify which kernels were launched during specific communication operations.
static std::atomic<uint64_t> nextCorrelationId(1);  // Global correlation ID counter (thread-safe)

// Timestamp synchronization: Map CUPTI timestamps to CPU time domain
// CUPTI activity records use GPU-synchronized timestamps (nanoseconds) that are in a different
// time domain than CPU timestamps from getTime() (microseconds, CLOCK_MONOTONIC).
// We capture one reference pair to caculate the offset once during CUPTI
// initialization as follows: offset = cpuTime - cuptiTime/1000.0
// when outputting CUPTI records, convert timestamp: cpuTime = cuptiTime/1000.0 + offset
static double cuptiTimestampOffsetUs = 0.0;

// Process-wide event pool and blacklist (shared by all contexts and PXN events)
static EventPool* processPool = nullptr;
static Blacklist* processBlacklist = nullptr;
static pthread_mutex_t poolBlacklistMutex = PTHREAD_MUTEX_INITIALIZER;

// Context registry: tracks all valid context pointers created by this process
// When PXN routing is enabled, NCCL may pass context pointers from other processes
// which are invalid in our address space. We use this registry to validate contexts.
static std::unordered_map<void*, MyContext*> contextRegistry;
static pthread_mutex_t contextRegistryMutex = PTHREAD_MUTEX_INITIALIZER;

static int activeContextCount = 0;           // Number of active communicators in this process

static void initProcessResources(const char* dirname) {
  if (initProcessPool()) {
    fprintf(stdout, "[MinimalProfiler] Process pool initialized\n");
  } else {
    fprintf(stderr, "[MinimalProfiler] WARNING: Failed to initialize process pool\n");
  }

  if (initJsonlTraceFile(dirname)) {
    fprintf(stdout, "[MinimalProfiler] JSONL trace file initialized\n");
  } else {
    fprintf(stderr, "[MinimalProfiler] WARNING: Failed to initialize JSONL file\n");
  }
  
  if (initCuptiActivity()) {
    fprintf(stdout, "[MinimalProfiler] CUPTI Activity API initialized\n");
  } else {
    fprintf(stderr, "[MinimalProfiler] WARNING: CUPTI Activity API failed\n");
  }
}

// Clean up process-wide resources (called when last communicator is finalized)
static void cleanupProcessResources() {
  // Cleanup CUPTI Activity API (flushes activities to JSONL)
  // Must happen before closing the JSONL file
  cleanupCuptiActivity();
  
  // Close the shared JSONL file (after CUPTI activities are written)
  closeSharedJsonlFile();
  pthread_mutex_destroy(&jsonlFileMutex);
  
  // Clean up process-wide pool and blacklist
  delete processPool;
  delete processBlacklist;
  processPool = nullptr;
  processBlacklist = nullptr;
  pthread_mutex_destroy(&poolBlacklistMutex);
  
  pthread_mutex_lock(&contextRegistryMutex);
  contextRegistry.clear();
  pthread_mutex_unlock(&contextRegistryMutex);
  pthread_mutex_destroy(&contextRegistryMutex);
}

// ============================================================================
// Section 5: PXN (PCI x NVLink)
// ============================================================================
// When PXN routing is enabled, some ProxyOp events may be executed by a different
// process than the one that originated them. The parentObj pointer in such events
// points to the originator's address space, which is invalid in the executing process.
// All events (normal and PXN) use the same process-wide pool and blacklist.
// For PXN events we store parentObj as metadata (pointer value) for blacklist tracking
// but never dereference it. We detect PXN events by checking eDescr->proxyOp.pid against getpid().

// Check if a context pointer is valid (belongs to this process)
static bool isValidContext(void* ctx) {
  if (!ctx) return false;
  pthread_mutex_lock(&contextRegistryMutex);
  bool valid = contextRegistry.find(ctx) != contextRegistry.end();
  pthread_mutex_unlock(&contextRegistryMutex);
  return valid;
}

// Register a context as valid for this process
static void registerContext(MyContext* ctx) {
  if (!ctx) return;
  pthread_mutex_lock(&contextRegistryMutex);
  contextRegistry[static_cast<void*>(ctx)] = ctx;
  pthread_mutex_unlock(&contextRegistryMutex);
}

// Unregister a context when it's being destroyed
static void unregisterContext(MyContext* ctx) {
  if (!ctx) return;
  pthread_mutex_lock(&contextRegistryMutex);
  contextRegistry.erase(static_cast<void*>(ctx));
  pthread_mutex_unlock(&contextRegistryMutex);
}

// Initialize the process-wide pool and blacklist (called on first communicator init)
static bool initProcessPool() {
  processPool = new (std::nothrow) EventPool();
  processBlacklist = new (std::nothrow) Blacklist();
  
  if (!processPool || !processBlacklist) {
    delete processPool;
    delete processBlacklist;
    processPool = nullptr;
    processBlacklist = nullptr;
    return false;
  }
  
  if (!processPool->init(INITIAL_POOL_SIZE)) {
    delete processPool;
    delete processBlacklist;
    processPool = nullptr;
    processBlacklist = nullptr;
    return false;
  }
  
  return true;
}

// Allocate an event from the process-wide pool
static MyEvent* allocateFromProcessPool() {
  if (!processPool || !processBlacklist) {
    return nullptr;
  }
  
  pthread_mutex_lock(&poolBlacklistMutex);
  MyEvent* event = processPool->allocate(*processBlacklist);
  pthread_mutex_unlock(&poolBlacklistMutex);
  
  return event;
}

// Free an event back to the process-wide pool (or delete if dynamically allocated)
static void freeToProcessPool(MyEvent* event) {
  if (!processPool) return;
  
  pthread_mutex_lock(&poolBlacklistMutex);
  if (event->fromPool) {
    processPool->free(event);
  } else {
    delete event;
  }
  pthread_mutex_unlock(&poolBlacklistMutex);
}

// Track a parent address in the process-wide blacklist
static void trackParent(void* parentObj) {
  if (!parentObj || !processBlacklist) return;
  
  pthread_mutex_lock(&poolBlacklistMutex);
  if (processBlacklist->contains(parentObj)) {
    processBlacklist->incrementRef(parentObj);
  } else {
    processBlacklist->add(parentObj);
  }
  pthread_mutex_unlock(&poolBlacklistMutex);
}

// Release a parent address from the process-wide blacklist
static void releaseParent(void* parentObj) {
  if (!parentObj || !processBlacklist) return;
  
  pthread_mutex_lock(&poolBlacklistMutex);
  processBlacklist->decrementRef(parentObj);
  pthread_mutex_unlock(&poolBlacklistMutex);
}

// ============================================================================
// SECTION 6: CUPTI Activity API Module for JSONL Activity Record Dumping
// ============================================================================
// This module captures CUDA activity records (kernels, memcpy, etc.) that have
// external correlation IDs matching our NCCL events. By writing these to the
// same JSONL file, post-processing tools can correlate NCCL communication events
// with the CUDA kernels that triggered them.
// ============================================================================

// CUPTI callback functions (modified from code sample `helper_cupti_activity.h`)
// Copyright 2022-2025 NVIDIA Corporation. refer to NVIDIA EULA associated with this code
static void handleDomainStateCallback(CUpti_CallbackId callbackId, const CUpti_StateData *pStateData) {
  switch (callbackId) {
    case CUPTI_CBID_STATE_FATAL_ERROR: {
      // If there is a CUPTI fatal error, it means CUPTI has stopped profiling the application.
      const char *errorString = NULL;
      cuptiGetResultString(pStateData->notification.result, &errorString);

      fprintf(stderr, "\nCUPTI encountered fatal error: %s\n", errorString);
      fprintf(stderr, "Error message: %s\n", pStateData->notification.message);
    }
    default:
      break;
  }
}

static void handleSyncronizationCallbacks(CUpti_CallbackId callbackId, 
                                          const CUpti_SynchronizeData *pSynchronizeData) {
  // Flush the CUPTI activity records buffer on context synchronization
  if (callbackId == CUPTI_CBID_SYNCHRONIZE_CONTEXT_SYNCHRONIZED) {
    CUPTI_API_CALL(cuptiActivityFlushAll(0));
  }
  // Flush the CUPTI activity records buffer on stream synchronization
  else if (callbackId == CUPTI_CBID_SYNCHRONIZE_STREAM_SYNCHRONIZED) {
    uint32_t streamId = 0;
    CUPTI_API_CALL(cuptiGetStreamId(pSynchronizeData->context, pSynchronizeData->stream, &streamId));
    CUPTI_API_CALL(cuptiActivityFlushAll(0));
  }
}

// subscribe this function in profiler/cupti initialization
static void CUPTIAPI cuptiCallbackHandler(void *pUserData, CUpti_CallbackDomain domain,
                                          CUpti_CallbackId callbackId, const void *pCallbackData) {
  CUPTI_API_CALL(cuptiGetLastError());

  (void)pUserData;

  const CUpti_CallbackData *pCallabckInfo = (CUpti_CallbackData *)pCallbackData;

  switch (domain) {
    case CUPTI_CB_DOMAIN_STATE:
      handleDomainStateCallback(callbackId, (CUpti_StateData *)pCallbackData);
      break;
    case CUPTI_CB_DOMAIN_RUNTIME_API:
      switch (callbackId) {
        case CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020:
          if (pCallabckInfo->callbackSite == CUPTI_API_ENTER)  {
            CUPTI_API_CALL(cuptiActivityFlushAll(0));
          }
          break;
        default:
          break;
      }
      break;
    case CUPTI_CB_DOMAIN_SYNCHRONIZE:
      handleSyncronizationCallbacks(callbackId, (CUpti_SynchronizeData *)pCallbackData);
      break;
    default:
      break;
  }
}

// CUPTI callback functions for buffer management
static void CUPTIAPI cuptiBufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  fprintf(stderr, "[CUPTI DEBUG] cuptiBufferRequested called\n");
  
  *buffer = (uint8_t *) malloc(CUPTI_BUFFER_SIZE);
  *size = CUPTI_BUFFER_SIZE;
  *maxNumRecords = 0;  // Let CUPTI decide

}

static void CUPTIAPI cuptiBufferCompleted(CUcontext ctx, uint32_t streamId, 
                                          uint8_t* buffer, size_t size, size_t validSize) {
  fprintf(stderr, "[CUPTI DEBUG] cuptiBufferCompleted called (streamId=%u, size=%zu, validSize=%zu)\n", 
          streamId, size, validSize);
  
  if (validSize > 0 && cuptiActivityEnabled) {
    
    CUpti_Activity* record = nullptr;
    int recordCount = 0;
    int kernelCount = 0, memcpyCount = 0, memsetCount = 0, extCorrCount = 0;
    
    // https://docs.nvidia.com/cupti/12.8/main/main.html#external-correlation
    // If a CUDA API activity record is generated while any CUpti_ExternalCorrelationKind-stack
    // on the same CPU thread is non-empty, one CUpti_ActivityExternalCorrelation record per 
    // CUpti_ExternalCorrelationKind-stack is inserted into the activity buffer before the respective 
    // CUDA API activity record
    
    // Following ncclsee approach: while processing the buffer, track the id from the "previously" 
    // inserted externnal correlation record to match next record in the buffer

    // track id in previous external correlation record
    uint64_t previous_external_id = 0;
    uint32_t previous_correlation_id = 0;
    
    // Process all activity records in the buffer
    do {
      CUptiResult status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status != CUPTI_SUCCESS) break;
      
      recordCount++;
      
      switch (record->kind) {
        case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: {
          // Store for next kernel/activity record
          CUpti_ActivityExternalCorrelation* extCorr = (CUpti_ActivityExternalCorrelation*)record;
          previous_external_id = extCorr->externalId;
          previous_correlation_id = extCorr->correlationId;
          extCorrCount++;

          fprintf(stderr, "[CUPTI DEBUG] ExtCorr: corrId=%u -> extId=%llu\n",
                  previous_correlation_id, (unsigned long long)previous_external_id);
          break;
        }
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
          CUpti_ActivityKernel4* kernel = (CUpti_ActivityKernel4*)record;
          kernelCount++;

          // Check if this kernel matches the previous external correlation record
          // A match means this kernel was launched during an NCCL event we're tracking
          if (previous_correlation_id == kernel->correlationId) {
            // // We only care about NCCL kernels
            // if (strstr(kernel->name, "nccl") == NULL)
            //   continue;
            
            uint64_t extCorrId = previous_external_id;
            fprintf(stderr, "[CUPTI DEBUG] KERNEL MATCHED: %s, corrId=%u, extCorrId=%llu\n",
                    kernel->name, kernel->correlationId, (unsigned long long)extCorrId);

            // Only write kernels that are correlated with NCCL events
            std::string detailsJson = "\"gridX\":" + std::to_string(kernel->gridX) +
                          ",\"gridY\":" + std::to_string(kernel->gridY) +
                          ",\"gridZ\":" + std::to_string(kernel->gridZ) +
                          ",\"blockX\":" + std::to_string(kernel->blockX) +
                          ",\"blockY\":" + std::to_string(kernel->blockY) +
                          ",\"blockZ\":" + std::to_string(kernel->blockZ);
            writeJsonlCuptiActivityRecord(record->kind, kernel->start, kernel->end,
                                          kernel->correlationId, extCorrId,
                                          kernel->deviceId, kernel->name,
                                          (void*)(uintptr_t)kernel->streamId, 0, detailsJson);
          } else {
            fprintf(stderr, "[CUPTI DEBUG] KERNEL (no match, skipping): %s, corrId=%u (prev=%u)\n",
                    kernel->name, kernel->correlationId, previous_correlation_id);
          }
          break;
        }
        case CUPTI_ACTIVITY_KIND_MEMCPY2: {
          CUpti_ActivityMemcpy2* memcpy = (CUpti_ActivityMemcpy2*)record;
          memcpyCount++;

          if (previous_correlation_id == memcpy->correlationId) {
            uint64_t extCorrId = previous_external_id;

            const char* copyKind = "Unknown";
            switch (memcpy->copyKind) {
              case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD: copyKind = "HtoD"; break;
              case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH: copyKind = "DtoH"; break;
              case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA: copyKind = "HtoA"; break;
              case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH: copyKind = "AtoH"; break;
              case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA: copyKind = "AtoA"; break;
              case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD: copyKind = "AtoD"; break;
              case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA: copyKind = "DtoA"; break;
              case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD: copyKind = "DtoD"; break;
              case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH: copyKind = "HtoH"; break;
              case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP: copyKind = "PtoP"; break;
              default: copyKind = "Unknown"; break;
            }

            std::string detailsJson = "\"copyKind\":\"" + escapeJsonStringStd(copyKind) + "\"" +
                          ",\"bytes\":" + std::to_string(memcpy->bytes);
            writeJsonlCuptiActivityRecord(record->kind, memcpy->start, memcpy->end,
                                          memcpy->correlationId, extCorrId,
                                          memcpy->deviceId, copyKind,
                                          (void*)(uintptr_t)memcpy->streamId, memcpy->bytes, detailsJson);
          }
          break;
        }
        case CUPTI_ACTIVITY_KIND_MEMSET: {
          CUpti_ActivityMemset* memset_rec = (CUpti_ActivityMemset*)record;
          memsetCount++;

          if (previous_correlation_id == memset_rec->correlationId) {
            uint64_t extCorrId = previous_external_id;

            std::string detailsJson = "\"bytes\":" + std::to_string(memset_rec->bytes) +
                          ",\"value\":" + std::to_string(memset_rec->value);
            writeJsonlCuptiActivityRecord(record->kind, memset_rec->start, memset_rec->end,
                                          memset_rec->correlationId, extCorrId,
                                          memset_rec->deviceId, "memset",
                                          (void*)(uintptr_t)memset_rec->streamId, memset_rec->bytes, detailsJson);
          }
          break;
        }
        default:
          break;
      }
    } while (1);
    
    fprintf(stderr, "[CUPTI DEBUG] Processed %d records: %d kernels, %d memcpy, %d memset, %d extCorr\n",
            recordCount, kernelCount, memcpyCount, memsetCount, extCorrCount);
  }
  
  free(buffer);
}

static double convertCuptiToCpuTime(uint64_t cuptiTimeNs) {
  return static_cast<double>(cuptiTimeNs) / 1000.0 + cuptiTimestampOffsetUs;
}

// Capture a timestamp reference pair (CPU time, CUPTI time) to calculate offset
// Should be called once during initialization
static void captureCuptiTimestampOffset() {
  double cpuTimeUs = getTime();
  uint64_t cuptiTimeNs = 0;
  CUPTI_API_CALL(cuptiGetTimestamp(&cuptiTimeNs));
  
  if (cuptiTimeNs != 0) {
    double cuptiTimeUs = static_cast<double>(cuptiTimeNs) / 1000.0;
    cuptiTimestampOffsetUs = cpuTimeUs - cuptiTimeUs;
  }
}

// CUPTI init and clean up
static bool initCuptiActivity() {
  // 1. Subscribe to CUPTI callback API
  // 2. Enable specific callbacks
  // 3. Register function callbacks for buffers
  // 4. Enable activity kinds
  // // 5. Additionally push initial correlation ID for pop/push logic
  
  CUptiResult cuptiErr;
  
  fprintf(stderr, "[CUPTI DEBUG] initCuptiActivity called\n");
  
  // Step 1: Subscribe to CUPTI callback API
  cuptiErr = cuptiSubscribe(&cuptiSubscriber, (CUpti_CallbackFunc)cuptiCallbackHandler, nullptr);
  if (cuptiErr != CUPTI_SUCCESS) {
    const char *pErrorString;
    cuptiGetResultString(cuptiErr, &pErrorString);
    fprintf(stderr, "[MinimalProfiler] WARNING: %s:%d: cuptiSubscribe failed with error(%d): %s\n",
            __FILE__, __LINE__, cuptiErr, pErrorString ? pErrorString : "Unknown error");
    return false;
  }

  // Step 2: Enable CUPTI callback on context syncronization, stream syncronization, device reset and fatal errors
  CUPTI_API_CALL(cuptiEnableCallback(1, cuptiSubscriber, CUPTI_CB_DOMAIN_SYNCHRONIZE, CUPTI_CBID_SYNCHRONIZE_CONTEXT_SYNCHRONIZED));
  CUPTI_API_CALL(cuptiEnableCallback(1, cuptiSubscriber, CUPTI_CB_DOMAIN_SYNCHRONIZE, CUPTI_CBID_SYNCHRONIZE_STREAM_SYNCHRONIZED));
  CUPTI_API_CALL(cuptiEnableCallback(1, cuptiSubscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020));
  CUPTI_API_CALL(cuptiEnableCallback(1, cuptiSubscriber, CUPTI_CB_DOMAIN_STATE, CUPTI_CBID_STATE_FATAL_ERROR));
  
  // Step 3: Register buffer callbacks
  cuptiErr = cuptiActivityRegisterCallbacks(cuptiBufferRequested, cuptiBufferCompleted);
  if (cuptiErr != CUPTI_SUCCESS) {
    const char *pErrorString;
    cuptiGetResultString(cuptiErr, &pErrorString);
    fprintf(stderr, "[MinimalProfiler] WARNING: %s:%d: cuptiActivityRegisterCallbacks failed with error(%d): %s\n",
            __FILE__, __LINE__, cuptiErr, pErrorString ? pErrorString : "Unknown error");
    return false;
  }

  // Step 4: Enable activity kinds
  // Enable DRIVER activity kind
  CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  
  // EXTERNAL_CORRELATION - https://docs.nvidia.com/cupti/12.8/main/main.html#external-correlation
  CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
   
  // specific kinds of interest: CONCURRENT_KERNEL, ... - https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#_CPPv418CUpti_ActivityKind
  CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY2));
  CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
  
  cuptiActivityEnabled = true;
  fprintf(stdout, "[MinimalProfiler] CUPTI Activity API initialized\n");
  
  captureCuptiTimestampOffset();
  
  // // Step 5: Push initial correlation ID
  // // This ensures there is a correlation ID on the stack that we can pop initially
  // uint64_t initialCorrId = nextCorrelationId.fetch_add(1);
  // CUPTI_API_CALL(cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2, initialCorrId));
  
  return true;
}

static void cleanupCuptiActivity() {
  fprintf(stderr, "[CUPTI DEBUG] cleanupCuptiActivity called (enabled=%d)\n", cuptiActivityEnabled);
  
  if (!cuptiActivityEnabled) return;
  
  CUPTI_API_CALL(cuptiGetLastError());
  
  CUPTI_API_CALL(cuptiActivityFlushAll(0)); // pass 1?

  if (cuptiSubscriber) {
    CUPTI_API_CALL(cuptiUnsubscribe(cuptiSubscriber));
    cuptiSubscriber = nullptr;
  }
  
  CUPTI_API_CALL(cuptiFinalize());

  cuptiActivityEnabled = false;
  
  fprintf(stderr, "[CUPTI DEBUG] CUPTI cleanup complete\n");
}

// ============================================================================
// Section 7: Logging
// ============================================================================

// ----------------------------------------------------------------------------
// Directory and File Helpers
// ----------------------------------------------------------------------------

// Get dump directory name (SLURM job ID or timestamp)
static const char* getDumpDir(char* dirname, size_t dirname_size) {
  const char* slurm_job_id = getenv("SLURM_JOB_ID");
  if (slurm_job_id && slurm_job_id[0] != '\0') {
    snprintf(dirname, dirname_size, "minimal_profiler_dump-%s", slurm_job_id);
    return dirname;
  }
  
  struct timespec ts;
  if (clock_gettime(CLOCK_REALTIME, &ts) != 0) {
    return nullptr;
  }
  
  struct tm* tm_info = localtime(&ts.tv_sec);
  if (!tm_info) {
    return nullptr;
  }
  
  snprintf(dirname, dirname_size, "minimal_profiler_dump-%04d%02d%02d-%02d%02d%02d",
           tm_info->tm_year + 1900,
           tm_info->tm_mon + 1,
           tm_info->tm_mday,
           tm_info->tm_hour,
           tm_info->tm_min,
           tm_info->tm_sec);
  
  return dirname;
}

// Create dump directory if it doesn't exist
static int ensureDumpDir(const char* dirname) {
  struct stat st = {0};
  
  if (stat(dirname, &st) == 0) {
    if (S_ISDIR(st.st_mode)) {
      return 0;
    } else {
      return -1;
    }
  }
  
  if (mkdir(dirname, 0755) != 0) {
    if (errno == EEXIST) {
      if (stat(dirname, &st) == 0 && S_ISDIR(st.st_mode)) {
        return 0;
      }
    }
    return -1;
  }
  
  return 0;
}

// ----------------------------------------------------------------------------
// JSON/JSONL Utility Functions
// ----------------------------------------------------------------------------

static std::string escapeJsonStringStd(const char* input) {
  if (!input) return "";
  
  std::string result;
  
  // Reserve enough space if every character would need escaping in advance
  result.reserve(strlen(input) * 2);  
  
  for (size_t i = 0; input[i] != '\0'; i++) {
    char c = input[i];
    switch (c) {
      case '"':  result += "\\\""; break;
      case '\\': result += "\\\\"; break;
      case '\n': result += "\\n"; break;
      case '\r': result += "\\r"; break;
      case '\t': result += "\\t"; break;
      default:
        if (c >= 32 && c < 127) {  // Printable ASCII
          result += c;
        }
        // Skip non-printable characters
        break;
    }
  }
  return result;
}

static std::string formatPointerStd(const void* ptr) {
  char buf[32];
  if (ptr) {
    snprintf(buf, sizeof(buf), "0x%lx", reinterpret_cast<unsigned long>(ptr));
  } else {
    snprintf(buf, sizeof(buf), "null");
  }
  return std::string(buf);
}

// Initialize the per-process JSONL trace file (called on first communicator init)
static bool initJsonlTraceFile(const char* dirname) {
  if (ensureDumpDir(dirname) != 0) {
    return false;
  }

  snprintf(sharedJsonlPath, sizeof(sharedJsonlPath), 
            "%s/trace_pid%d.jsonl", dirname, static_cast<int>(getpid()));
  
  // Open in write mode - this file is exclusive to this process, no append/locking needed
  sharedJsonlFile = fopen(sharedJsonlPath, "w");
  if (!sharedJsonlFile) {
    fprintf(stderr, "[MinimalProfiler] WARNING: Failed to open JSONL file: %s\n", sharedJsonlPath);
    return false;
  }
  
  fprintf(stdout, "[MinimalProfiler] JSONL trace file: %s\n", sharedJsonlPath);
  return true;
}

// Close the shared JSONL file (called when last communicator is finalized)
static void closeSharedJsonlFile() {
  if (sharedJsonlFile) {
    fflush(sharedJsonlFile);
    fclose(sharedJsonlFile);
    sharedJsonlFile = nullptr;
  }
}

// ----------------------------------------------------------------------------
// JSONL Writers - Separate Records for Events and States and CUPTI records
// ----------------------------------------------------------------------------
// - "event" records: written at STOP time with start/stop info and details
// - "state" records: written immediately at STATE time with eventAddr reference
// ----------------------------------------------------------------------------

// Write an event record to JSONL (called at STOP time - no state transitions)
static void writeJsonlEventRecord(MyEvent* event) {
  if (!event || !sharedJsonlFile) return;

  double stopTime = getTime();
  double duration = (stopTime - event->startTime);
  pid_t stopPid = getpid();
  pid_t stopTid = getTid();
  
  // Build the event JSON - (event states are separate records)
  std::string json = "{\"recordType\":\"event\"" +
                     std::string(",\"type\":\"") + escapeJsonStringStd(event->typeName ? event->typeName : "Unknown") + "\"";
  
  if (!event->isPxn) {
    // Normal event - include gpuUuid, commId, rank, ctx
    json += ",\"gpuUuid\":\"" + escapeJsonStringStd(event->gpuUuid[0] != '\0' ? event->gpuUuid : "") + "\"" +
            ",\"commId\":" + std::to_string(event->commId) +
            ",\"rank\":" + std::to_string(event->rank);
  }
  
  // Start/stop timestamps
  char buf[64];
  snprintf(buf, sizeof(buf), "%.3f", event->startTime);
  json += ",\"start\":{\"ts\":" + std::string(buf) +
          ",\"pid\":" + std::to_string(static_cast<int>(event->startPid)) +
          ",\"tid\":" + std::to_string(static_cast<int>(event->startTid)) + "}";
  
  snprintf(buf, sizeof(buf), "%.3f", stopTime);
  json += ",\"stop\":{\"ts\":" + std::string(buf) +
          ",\"pid\":" + std::to_string(static_cast<int>(stopPid)) +
          ",\"tid\":" + std::to_string(static_cast<int>(stopTid)) + "}";
  
  snprintf(buf, sizeof(buf), "%.3f", duration);
  json += ",\"duration\":" + std::string(buf);
  
  // Add external correlation ID if present (links to CUPTI activity records)
  if (event->hasExternalCorrelationId) {
    json += ",\"extCorrId\":" + std::to_string(event->externalCorrelationId);
  }
  
  json += ",\"parentObj\":\"" + formatPointerStd(event->parentObj) + "\"" +
          ",\"eventAddr\":\"" + formatPointerStd(static_cast<void*>(event)) + "\"";
  
  if (!event->isPxn && event->ctx) {
    json += ",\"ctx\":\"" + formatPointerStd(static_cast<void*>(event->ctx)) + "\"";
    }
  
  if (event->isPxn) {
    json += ",\"isPxn\":true";
  }
  
  // Add details
  json += ",\"details\":{" + event->typeDetails + "}}\n";
  
  // event state transitions are not added. currently they are separate records

  // Write to per-process file
  pthread_mutex_lock(&jsonlFileMutex);
  if (fileno(sharedJsonlFile) >= 0) {
    int fd = fileno(sharedJsonlFile);
    write(fd, json.c_str(), json.size());
  }
  pthread_mutex_unlock(&jsonlFileMutex);
}

// Write a state transition record to JSONL (called immediately at STATE time)
static void writeJsonlStateRecord(MyEvent* event, double timestamp, const char* stateName,
                                   int stateId, pid_t pid, pid_t tid, const std::string& details) {
  if (!event || !sharedJsonlFile) return;
  
  char buf[64];
  snprintf(buf, sizeof(buf), "%.3f", timestamp);
  
  std::string json = "{\"recordType\":\"state\"" +
                     std::string(",\"eventAddr\":\"") + formatPointerStd(static_cast<void*>(event)) + "\"" +
                     ",\"ts\":" + std::string(buf) +
                     ",\"name\":\"" + escapeJsonStringStd(stateName) + "\"" +
                     ",\"id\":" + std::to_string(stateId) +
                     ",\"pid\":" + std::to_string(static_cast<int>(pid)) +
                     ",\"tid\":" + std::to_string(static_cast<int>(tid));
  
  if (!details.empty()) {
    json += "," + details;
  }
  json += "}\n";
  
  // Write to per-process file
  pthread_mutex_lock(&jsonlFileMutex);
  if (fileno(sharedJsonlFile) >= 0) {
    int fd = fileno(sharedJsonlFile);
    write(fd, json.c_str(), json.size());
  }
  pthread_mutex_unlock(&jsonlFileMutex);
}

// Write a ProfilerLifecycle pseudo-event to JSONL
static void writeJsonlLifecycleEvent(const char* action, double timestamp, 
                                      const char* gpuUuid, uint64_t commId, int rank,
                                      const std::string& detailsJson) {
  if (!sharedJsonlFile) return;
  
  pid_t tid = getTid();
  pid_t pid = getpid();
  
  char buf[64];
  snprintf(buf, sizeof(buf), "%.3f", timestamp);
  
  std::string json = "{\"type\":\"ProfilerLifecycle\",\"func\":\"" + escapeJsonStringStd(action) + "\"" +
                     ",\"gpuUuid\":\"" + escapeJsonStringStd(gpuUuid ? gpuUuid : "") + "\"" +
                     ",\"commId\":" + std::to_string(commId) +
                     ",\"rank\":" + std::to_string(rank) +
                     ",\"start\":{\"ts\":" + std::string(buf) +
                     ",\"pid\":" + std::to_string(static_cast<int>(pid)) +
                     ",\"tid\":" + std::to_string(static_cast<int>(tid)) + "}" +
                     ",\"stop\":{\"ts\":" + std::string(buf) +
                     ",\"pid\":" + std::to_string(static_cast<int>(pid)) +
                     ",\"tid\":" + std::to_string(static_cast<int>(tid)) + "}" +
                     ",\"duration\":0.0" +
                     ",\"details\":{" + detailsJson + "}}\n";
  
  // Write to per-process file
  pthread_mutex_lock(&jsonlFileMutex);
  if (fileno(sharedJsonlFile) >= 0) {
    int fd = fileno(sharedJsonlFile);
    write(fd, json.c_str(), json.size());
  }
  pthread_mutex_unlock(&jsonlFileMutex);
}

// Write a CUPTI activity record to JSONL
static void writeJsonlCuptiActivityRecord(uint32_t kind, uint64_t startTime, uint64_t endTime,
                                          uint32_t correlationId, uint64_t externalCorrelationId,
                                          uint32_t deviceId, const char* name, void* stream,
                                          size_t size, const std::string& detailsJson) {
  if (!sharedJsonlFile) return;
  
  pid_t tid = getTid();
  pid_t pid = getpid();
  std::string gpuUuid = getGpuUuid();
  
  char startBuf[64], endBuf[64], durationBuf[64];
  double startTimeUs = convertCuptiToCpuTime(startTime);
  double endTimeUs = convertCuptiToCpuTime(endTime);
  double durationUs = endTimeUs - startTimeUs;
  
  snprintf(startBuf, sizeof(startBuf), "%.3f", startTimeUs);
  snprintf(endBuf, sizeof(endBuf), "%.3f", endTimeUs);
  snprintf(durationBuf, sizeof(durationBuf), "%.3f", durationUs);
  
  const char* kindName = "Unknown";
  switch (kind) {
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
      kindName = "KERNEL";
      break;
    case CUPTI_ACTIVITY_KIND_MEMCPY2:
      kindName = "MEMCPY";
      break;
    case CUPTI_ACTIVITY_KIND_MEMSET:
      kindName = "MEMSET";
      break;
    default:
      kindName = "OTHER";
      break;
  }
  
  // Build JSONL record
  std::string json = std::string("{\"recordType\":\"cuptiActivity\"") +
                     ",\"type\":\"" + escapeJsonStringStd(kindName) + "\"" +
                     ",\"extCorrId\":" + std::to_string(externalCorrelationId) +
                     ",\"corrId\":" + std::to_string(correlationId) +
                     ",\"gpuUuid\":\"" + escapeJsonStringStd(gpuUuid.c_str()) + "\"" +
                     ",\"deviceId\":" + std::to_string(deviceId) +
                     ",\"start\":{\"ts\":" + std::string(startBuf) +
                     ",\"pid\":" + std::to_string(static_cast<int>(pid)) +
                     ",\"tid\":" + std::to_string(static_cast<int>(tid)) + "}" +
                     ",\"stop\":{\"ts\":" + std::string(endBuf) +
                     ",\"pid\":" + std::to_string(static_cast<int>(pid)) +
                     ",\"tid\":" + std::to_string(static_cast<int>(tid)) + "}" +
                     ",\"duration\":" + std::string(durationBuf);
  
  if (name) {
    json += ",\"name\":\"" + escapeJsonStringStd(name) + "\"";
  }
  
  if (stream) {
    json += ",\"stream\":\"" + formatPointerStd(stream) + "\"";
  }
  
  if (size > 0) {
    json += ",\"size\":" + std::to_string(size);
  }
  
  if (!detailsJson.empty()) {
    json += ",\"details\":{" + detailsJson + "}";
  }
  
  json += "}\n";
  
  // Write to per-process JSONL file (same file as NCCL events)
  pthread_mutex_lock(&jsonlFileMutex);
  if (fileno(sharedJsonlFile) >= 0) {
    int fd = fileno(sharedJsonlFile);
    write(fd, json.c_str(), json.size());
  }
  pthread_mutex_unlock(&jsonlFileMutex);
}

// ============================================================================
// SECTION 8: Profiler API Implementation
// ============================================================================
extern "C" {

// ----------------------------------------------------------------------------
// myInit: Initialize profiler context for a communicator
// ----------------------------------------------------------------------------
ncclResult_t myInit(void** context, uint64_t commId, int* eActivationMask,
                    const char* commName, int nNodes, int nranks, int rank,
                    ncclDebugLogger_t logfn) {
  // Parse event activation mask from environment
  const char* maskStr = getenv(ENV_EVENT_MASK);
  int mask = maskStr ? atoi(maskStr) : 4095;  // Default: enable ALL event types
  *eActivationMask = mask;
  
  pid_t tid = getTid();
  pid_t pid = getpid();
  std::string gpuUuidStr = getGpuUuid();
  
  fprintf(stdout, "[MinimalProfiler] Init called: pid=%d, tid=%d, rank=%d/%d, commId=%lu, eventMask=0x%x\n",
          static_cast<int>(pid), tid, rank, nranks, commId, mask);
  
  // Get dump directory name
  char dirname[256];
  if (!getDumpDir(dirname, sizeof(dirname))) {
    return ncclInternalError;
  }
  
  // Initialize process-wide state on first communicator init
  pthread_mutex_lock(&processStateMutex);
  if (activeContextCount == 0) {
    initProcessResources(dirname);
  }
  activeContextCount++;
  pthread_mutex_unlock(&processStateMutex);
  
  // Initialize context
  MyContext* ctx = new (std::nothrow) MyContext();
  if (!ctx) return ncclInternalError;
  
  ctx->commId = commId;
  ctx->rank = rank;
  ctx->nranks = nranks;
  ctx->startTime = getTime();
  snprintf(ctx->gpuUuid, sizeof(ctx->gpuUuid), "%s", gpuUuidStr.c_str());
  
  registerContext(ctx);  // Register in the process-wide registry for PXN validation
  
  // Write ProfilerInit lifecycle event to JSONL
  std::string detailsJson = "\"nranks\":" + std::to_string(nranks) +
                            ",\"commName\":\"" + escapeJsonStringStd(commName ? commName : "") + "\"" +
                            ",\"eventMask\":" + std::to_string(mask);
  writeJsonlLifecycleEvent("ProfilerInit", ctx->startTime, ctx->gpuUuid, commId, rank, detailsJson);
  
  fprintf(stdout, "[MinimalProfiler] Successfully initialized: pid=%d, tid=%d, rank=%d, commId=%lu, pool_size=%zu\n", 
    static_cast<int>(pid), tid, rank, commId, INITIAL_POOL_SIZE);

  *context = ctx;
  return ncclSuccess;
}

// ----------------------------------------------------------------------------
// myStartEvent: Begin tracking a profiler event
// ----------------------------------------------------------------------------
ncclResult_t myStartEvent(void* context, void** eHandle, 
                          ncclProfilerEventDescr_v5_t* eDescr) {
  double startTime = getTime();
  MyContext* ctx = static_cast<MyContext*>(context);
  MyEvent* event = nullptr;
  
  // ========================================================================
  // PXN (PCI x NVLink) CONTEXT VALIDATION
  // ========================================================================
  // When PXN routing is enabled, NCCL may process events on a different proxy
  // thread than the one that originated them. In this case, 'context' points
  // to memory in the originating process's address space, which is INVALID in
  // our address space.
  //
  // We detect this in two ways:
  // 1. For ProxyOp events: check eDescr->proxyOp.pid against getpid()
  // 2. For ALL events: check if context is in our registry of valid contexts
  //
  // If the context is invalid, we use the process-wide pxn pool instead.
  // ========================================================================
  
  
  bool isPxn;
  if (eDescr->type == ncclProfileProxyOp) {
    isPxn = (eDescr->proxyOp.pid != getpid());
  } else {
    isPxn = !isValidContext(context);
  }
  
  // Allocate event from process-wide pool (same for normal and PXN events)
  event = allocateFromProcessPool();
  if (!event) {
    *eHandle = nullptr;
    return ncclSuccess;  // Pool allocation failed - drop this event gracefully
  }
  // Track parent in process-wide blacklist to prevent address reuse
  if (eDescr->parentObj) {
    trackParent(eDescr->parentObj);
  }

  if (isPxn) {
    event->ctx = nullptr;        // ctx is invalid for PXN events - don't store it
    event->gpuUuid[0] = '\0';    // Not available for PXN
    event->commId = 0;           // Not available for PXN
    event->rank = -1;            // Not available for PXN
  } else {
    event->ctx = ctx;
    snprintf(event->gpuUuid, sizeof(event->gpuUuid), "%s", ctx->gpuUuid);
    event->commId = ctx->commId;
    event->rank = ctx->rank;
  }
  
  // correlate NCCL events with Kernels using CUPTI Activity API (only for non-PXN-events)
  if ((eDescr->type == ncclProfileKernelLaunch) && !isPxn) {
    
    // PUSH a correlation ID for subsequent activities
    uint64_t newCorrId = nextCorrelationId.fetch_add(1);
    CUPTI_API_CALL(cuptiActivityPushExternalCorrelationId(
      CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2,
      newCorrId
    ));
    
    // Store the new correlation ID with this event for later matching
    event->externalCorrelationId = newCorrId;
    event->hasExternalCorrelationId = true;

  } else {
    event->externalCorrelationId = 0;
    event->hasExternalCorrelationId = false;
  }
  
  // set type specific details JSON
  event->typeDetails.clear();
  switch (eDescr->type) {
    case ncclProfileGroupApi:
      event->typeDetails = "\"depth\":" + std::to_string(eDescr->groupApi.groupDepth) +
                            ",\"graphCaptured\":" + (eDescr->groupApi.graphCaptured ? "true" : "false");
      break;
    case ncclProfileCollApi:
      event->typeDetails = "\"func\":\"" + escapeJsonStringStd(eDescr->collApi.func) + "\"" +
                            ",\"count\":" + std::to_string(eDescr->collApi.count) +
                            ",\"dtype\":\"" + escapeJsonStringStd(eDescr->collApi.datatype) + "\"" +
                            ",\"root\":" + std::to_string(eDescr->collApi.root) +
                            ",\"stream\":\"" + formatPointerStd(eDescr->collApi.stream) + "\"" +
                            ",\"graphCaptured\":" + (eDescr->collApi.graphCaptured ? "true" : "false");
      break;
    case ncclProfileP2pApi:
      event->typeDetails = "\"func\":\"" + escapeJsonStringStd(eDescr->p2pApi.func) + "\"" +
                            ",\"count\":" + std::to_string(eDescr->p2pApi.count) +
                            ",\"dtype\":\"" + escapeJsonStringStd(eDescr->p2pApi.datatype) + "\"" +
                            ",\"stream\":\"" + formatPointerStd(eDescr->p2pApi.stream) + "\"" +
                            ",\"graphCaptured\":" + (eDescr->p2pApi.graphCaptured ? "true" : "false");
      break;
    case ncclProfileKernelLaunch:
      event->typeDetails = "\"stream\":\"" + formatPointerStd(eDescr->kernelLaunch.stream) + "\"";
      break;
    case ncclProfileColl:
      event->typeDetails = "\"func\":\"" + escapeJsonStringStd(eDescr->coll.func) + "\"" +
                            ",\"seq\":" + std::to_string(eDescr->coll.seqNumber) +
                            ",\"algo\":\"" + escapeJsonStringStd(eDescr->coll.algo) + "\"" +
                            ",\"proto\":\"" + escapeJsonStringStd(eDescr->coll.proto) + "\"" +
                            ",\"channels\":" + std::to_string(eDescr->coll.nChannels) +
                            ",\"warps\":" + std::to_string(eDescr->coll.nWarps) +
                            ",\"count\":" + std::to_string(eDescr->coll.count) +
                            ",\"root\":" + std::to_string(eDescr->coll.root) +
                            ",\"dtype\":\"" + escapeJsonStringStd(eDescr->coll.datatype) + "\"" +
                            ",\"sendBuff\":\"" + formatPointerStd(eDescr->coll.sendBuff) + "\"" +
                            ",\"recvBuff\":\"" + formatPointerStd(eDescr->coll.recvBuff) + "\"" +
                            ",\"parentGroup\":\"" + formatPointerStd(eDescr->coll.parentGroup) + "\"";
      break;
    case ncclProfileP2p:
      event->typeDetails = "\"func\":\"" + escapeJsonStringStd(eDescr->p2p.func) + "\"" +
                            ",\"peer\":" + std::to_string(eDescr->p2p.peer) +
                            ",\"channels\":" + std::to_string(eDescr->p2p.nChannels) +
                            ",\"count\":" + std::to_string(eDescr->p2p.count) +
                            ",\"dtype\":\"" + escapeJsonStringStd(eDescr->p2p.datatype) + "\"" +
                            ",\"buff\":\"" + formatPointerStd(eDescr->p2p.buff) + "\"" +
                            ",\"parentGroup\":\"" + formatPointerStd(eDescr->p2p.parentGroup) + "\"";
      break;
    case ncclProfileProxyOp:
      event->typeDetails = "\"channel\":" + std::to_string(eDescr->proxyOp.channelId) +
                            ",\"peer\":" + std::to_string(eDescr->proxyOp.peer) +
                            ",\"steps\":" + std::to_string(eDescr->proxyOp.nSteps) +
                            ",\"chunkSize\":" + std::to_string(eDescr->proxyOp.chunkSize) +
                            ",\"isSend\":" + std::string(eDescr->proxyOp.isSend ? "true" : "false") +
                            ",\"eventPid\":" + std::to_string(static_cast<int>(eDescr->proxyOp.pid));
      break;
    case ncclProfileProxyStep:
      event->typeDetails = "\"step\":" + std::to_string(eDescr->proxyStep.step);
      break;
    case ncclProfileProxyCtrl:
      // No extra details
      break;
    case ncclProfileKernelCh:
      event->typeDetails = "\"channel\":" + std::to_string(eDescr->kernelCh.channelId) +
                            ",\"pTimer\":" + std::to_string(eDescr->kernelCh.pTimer);
      break;
    case ncclProfileNetPlugin:
      event->typeDetails = "\"id\":" + std::to_string(static_cast<long>(eDescr->netPlugin.id)) +
                            ",\"data\":\"" + formatPointerStd(eDescr->netPlugin.data) + "\"";
      break;
    case ncclProfileGroup:
      // No extra details
      break;
    default:
      {
        char buf[32];
        snprintf(buf, sizeof(buf), "0x%lx", static_cast<unsigned long>(eDescr->type));
        event->typeDetails = "\"rawType\":\"" + std::string(buf) + "\"";
      }
      break;
  }
  
  // Set basic fields
  event->type = eDescr->type;
  event->parentObj = eDescr->parentObj;
  event->typeName = getEventTypeName(eDescr->type);
  event->isPxn = isPxn;
  event->startTime = startTime;
  event->startTid = getTid();
  event->startPid = getpid();

  // Note: JSONL output is written in myStopEvent as a complete event with stopTime etc.
  
  *eHandle = event;
  return ncclSuccess;
}

// ----------------------------------------------------------------------------
// myStopEvent: End tracking a profiler event
// ----------------------------------------------------------------------------
ncclResult_t myStopEvent(void* eHandle) {
  if (!eHandle) return ncclSuccess;
  
  MyEvent* event = static_cast<MyEvent*>(eHandle);
  
  writeJsonlEventRecord(event);

  // CUPTI Activity API - pop id
  if (event->hasExternalCorrelationId) {
    uint64_t poppedCorrId = 0;
    CUPTI_API_CALL(cuptiActivityPopExternalCorrelationId(
      CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2,
      &poppedCorrId
    ));
  }

  // Release parent from process-wide blacklist and free event to process pool
  if (event->parentObj) {
    releaseParent(event->parentObj);
  }
  freeToProcessPool(event);

  return ncclSuccess;
}

// ----------------------------------------------------------------------------
// myRecordEventState: Record state transitions during event lifecycle
// ----------------------------------------------------------------------------
ncclResult_t myRecordEventState(void* eHandle, 
                                 ncclProfilerEventState_v5_t eState,
                                 ncclProfilerEventStateArgs_v5_t* eStateArgs) {
  if (!eHandle) return ncclSuccess;
  
  // Write state to JSONL
  double currentTime = getTime();
  pid_t tid = getTid();
  pid_t pid = getpid();
  
  std::string details;
  if (eStateArgs) {
    switch (eState) {
      case ncclProfilerProxyStepSendWait:
      case ncclProfilerProxyStepRecvFlushWait:
          details = "\"transSize\":" + std::to_string(eStateArgs->proxyStep.transSize);
        break;
      case ncclProfilerProxyCtrlAppendEnd:
          details = "\"appendedProxyOps\":" + std::to_string(eStateArgs->proxyCtrl.appendedProxyOps);
        break;
      case ncclProfilerKernelChStop:
          details = "\"pTimer\":" + std::to_string(eStateArgs->kernelCh.pTimer);
        break;
      case ncclProfilerNetPluginUpdate:
          details = "\"data\":\"" + formatPointerStd(eStateArgs->netPlugin.data) + "\"";
        break;
      default:
        break;
    }
  }
  writeJsonlStateRecord(static_cast<MyEvent*>(eHandle), currentTime, getStateName(eState), static_cast<int>(eState), pid, tid, details);
  
  return ncclSuccess;
}

// ----------------------------------------------------------------------------
// myFinalize: Clean up profiler context for a communicator
// ----------------------------------------------------------------------------
ncclResult_t myFinalize(void* context) {
  MyContext* ctx = static_cast<MyContext*>(context);
  pid_t tid = getTid();
  double finalizeTime = getTime();
  double totalDuration = finalizeTime - ctx->startTime;
  
  // Write ProfilerFinalize lifecycle event to JSONL (process pool stats)
  char buf[64];
  snprintf(buf, sizeof(buf), "%.3f", totalDuration);
  size_t poolChunks = 0, poolSlots = 0;
  if (processPool) {
    pthread_mutex_lock(&poolBlacklistMutex);
    poolChunks = processPool->chunks.size();
    poolSlots = processPool->chunks.size() * processPool->chunkSize;
    pthread_mutex_unlock(&poolBlacklistMutex);
  }
  std::string detailsJson = "\"totalDuration\":" + std::string(buf) +
                            ",\"poolChunks\":" + std::to_string(poolChunks) +
                            ",\"poolSlots\":" + std::to_string(poolSlots);
  writeJsonlLifecycleEvent("ProfilerFinalize", finalizeTime, ctx->gpuUuid, ctx->commId, ctx->rank, detailsJson);
  
  // Clean up context
  unregisterContext(ctx); // Unregister this context from the process-wide registry
  delete ctx;
  
  // Clean up process-wide resources when last communicator is finalized
  pthread_mutex_lock(&processStateMutex);
  activeContextCount--;
  if (activeContextCount == 0) {
    cleanupProcessResources();
  }
  pthread_mutex_unlock(&processStateMutex);
  
  fprintf(stdout, "[MinimalProfiler] Finalized: tid=%d\n", tid);
  return ncclSuccess;
}

// ============================================================================
// SECTION 9: Export Struct
// ============================================================================
ncclProfiler_v5_t ncclProfiler_v5 = {
  "MyMinimalProfiler",           // Plugin name
  myInit,                        // Initialize function
  myStartEvent,                  // Start event function
  myStopEvent,                   // Stop event function
  myRecordEventState,            // Record state function
  myFinalize,                    // Finalize function
};

} // extern "C"
