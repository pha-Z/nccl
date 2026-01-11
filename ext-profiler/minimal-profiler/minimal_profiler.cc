/*************************************************************************
 * Minimal NCCL Profiler Plugin Example (C++ version)
 * 
 * This profiler plugin uses memory pools and blacklisting to prevent
 * address reuse that would obfuscate parent-child event relationships.
 * 
 * Key features:
 * - Event pool with round-robin allocation to maximize time between address reuse
 * - Blacklist with LRU eviction to prevent reuse of addresses still referenced as parents
 * - O(1) operations for all blacklist operations via hash map + doubly linked list
 * - PXN (PCI x NVLink) support for cross-process ProxyOp events
 * 
 * Code Organization:
 * 1. Configuration constants
 * 2. Data structures
 * 3. PXN Detach Pool module
 * 4. Utility functions
 * 5. Profiler API implementation
 * 6. Export struct
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

// Include the profiler API headers (C headers)
extern "C" {
#include "nccl/common.h"
#include "nccl/err.h"
#include "nccl/profiler.h"
#include "nccl/types.h"
}

// ============================================================================
// SECTION 1: Configuration Constants
// ============================================================================

// Pool sizing
static const size_t INITIAL_POOL_SIZE = 1024;           // Initial event pool size per context
static const size_t INITIAL_DETACH_POOL_SIZE = 256;     // Initial size for PXN detach pool

// Blacklist sizing
static const size_t BLACKLIST_CAP = 10000000;           // 10 million entries max

// Environment variable names
static const char* ENV_EVENT_MASK = "NCCL_PROFILE_EVENT_MASK";

// JSONL tracing configuration
// Note: Each process writes to its own trace file (trace_<jobid>_proc<procid>.jsonl).
// This avoids write interleaving issues on network filesystems (NFS, Lustre, GPFS)
// where O_APPEND atomicity guarantees do not apply across different clients/nodes.

// ============================================================================
// SECTION 2: Data Structures
// ============================================================================

// Forward declarations
struct MyEvent;
struct MyContext;
struct EventPool;
struct Blacklist;

// Forward declare functions used before definition
static void closeSharedJsonlFile();

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
// MyEvent: Stores event data + parent reference for cleanup
// ----------------------------------------------------------------------------
struct MyEvent {
  // Basic event info (set at START)
  uint64_t type;
  double startTime;
  const char* typeName;
  MyContext* ctx;
  void* parentObj;        // Store parentObj to decrement blacklist ref on stop
  bool fromPool;          // Indicates whether this event was allocated from pool or dynamically
  bool fromDetachPool;    // Indicates whether this event is from detach pool (PXN event)
  pid_t originPid;        // PID of the process that originated this event (for PXN tracking)
  
  // Start event details (captured at START for JSONL output)
  pid_t startTid;
  pid_t startPid;
  std::string startDetails;  // Type-specific details JSON from START (dynamic)
  
  // Context-dependent fields captured at START (safe for PXN - copied, not pointer)
  char gpuUuid[64];       // GPU UUID (empty for PXN events)
  uint64_t commId;        // Communicator ID (0 for PXN events)
  int rank;               // Rank (-1 for PXN events)
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
    
    fprintf(stderr, "[MinimalProfiler] INFO: Event pool grown: added chunk %zu (total chunks: %zu, total slots: %zu)\n",
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
  pthread_mutex_t allocMutex;   // Protects pool and blacklist
  uint64_t commId;
  int rank;
  int nranks;
  double startTime;
  char gpuUuid[37];             // Physical GPU UUID string (36 chars + null terminator)
  
  EventPool pool;
  Blacklist blacklist;
};

// ============================================================================
// SECTION 3: Process-Wide Global State
// ============================================================================
// These resources are shared across all communicators (contexts) within a single
// process. They are initialized when the first communicator is created and
// cleaned up when the last communicator is finalized.
//
// PXN (PCI x NVLink) Support:
// When PXN routing is enabled, some ProxyOp events may be executed by a different
// process than the one that originated them. The parentObj pointer in such events
// points to the originator's address space, which is invalid in the executing process.
// These "detached" events are stored in a separate pool (pxnDetachPool) and we store
// parentObj as metadata (pointer value) for tracking but never dereference it.
// We detect PXN events by checking eDescr->proxyOp.pid against profilerPid.
// ============================================================================

// Forward declaration for utility function used in this section
static double getTime();

// Lifecycle tracking for process-wide resources
static pthread_mutex_t processStateMutex = PTHREAD_MUTEX_INITIALIZER;
static int activeContextCount = 0;           // Number of active communicators in this process
static pid_t profilerPid = 0;                // This process's PID (set once at first init)
static uint64_t firstCommId = 0;             // First commId seen (used for file naming)

// PXN Detach Pool: handles cross-process ProxyOp events
static EventPool* pxnDetachPool = nullptr;
static Blacklist* pxnDetachBlacklist = nullptr;
static pthread_mutex_t pxnDetachMutex = PTHREAD_MUTEX_INITIALIZER;

// Per-process JSONL trace file (each process writes to its own file)
static FILE* sharedJsonlFile = nullptr;      // Per-process trace file (write mode, exclusive access)
static pthread_mutex_t jsonlFileMutex = PTHREAD_MUTEX_INITIALIZER;  // Protects writes within this process
static char sharedJsonlPath[512] = {0};      // Path to this process's JSONL file

// Context registry: tracks all valid context pointers created by this process
// When PXN routing is enabled, NCCL may pass context pointers from other processes
// which are invalid in our address space. We use this registry to validate contexts.
static std::unordered_map<void*, MyContext*> contextRegistry;
static pthread_mutex_t contextRegistryMutex = PTHREAD_MUTEX_INITIALIZER;

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

// Initialize the PXN detach pool (called on first communicator init)
static bool initPxnDetachPool() {
  pxnDetachPool = new (std::nothrow) EventPool();
  pxnDetachBlacklist = new (std::nothrow) Blacklist();
  
  if (!pxnDetachPool || !pxnDetachBlacklist) {
    delete pxnDetachPool;
    delete pxnDetachBlacklist;
    pxnDetachPool = nullptr;
    pxnDetachBlacklist = nullptr;
    return false;
  }
  
  if (!pxnDetachPool->init(INITIAL_DETACH_POOL_SIZE)) {
    delete pxnDetachPool;
    delete pxnDetachBlacklist;
    pxnDetachPool = nullptr;
    pxnDetachBlacklist = nullptr;
    return false;
  }
  
  return true;
}

// Allocate an event from the PXN detach pool
static MyEvent* allocateFromPxnPool() {
  if (!pxnDetachPool || !pxnDetachBlacklist) {
    return nullptr;
  }
  
  pthread_mutex_lock(&pxnDetachMutex);
  MyEvent* event = pxnDetachPool->allocate(*pxnDetachBlacklist);
  pthread_mutex_unlock(&pxnDetachMutex);
  
  if (event) {
    event->fromDetachPool = true;
    event->fromPool = true;
  }
  
  return event;
}

// Free an event back to the PXN detach pool
static void freeToPxnPool(MyEvent* event) {
  if (!pxnDetachPool) return;
  
  pthread_mutex_lock(&pxnDetachMutex);
  pxnDetachPool->free(event);
  pthread_mutex_unlock(&pxnDetachMutex);
}

// Track a parent address in the PXN blacklist
static void trackPxnParent(void* parentObj) {
  if (!parentObj || !pxnDetachBlacklist) return;
  
  pthread_mutex_lock(&pxnDetachMutex);
  if (pxnDetachBlacklist->contains(parentObj)) {
    pxnDetachBlacklist->incrementRef(parentObj);
  } else {
    pxnDetachBlacklist->add(parentObj);
  }
  pthread_mutex_unlock(&pxnDetachMutex);
}

// Release a parent address from the PXN blacklist
static void releasePxnParent(void* parentObj) {
  if (!parentObj || !pxnDetachBlacklist) return;
  
  pthread_mutex_lock(&pxnDetachMutex);
  pxnDetachBlacklist->decrementRef(parentObj);
  pthread_mutex_unlock(&pxnDetachMutex);
}

// Clean up process-wide resources (called when last communicator is finalized)
static void cleanupProcessResources() {
  // Close the shared JSONL file
  closeSharedJsonlFile();
  pthread_mutex_destroy(&jsonlFileMutex);
  
  delete pxnDetachPool;
  delete pxnDetachBlacklist;
  pxnDetachPool = nullptr;
  pxnDetachBlacklist = nullptr;
  profilerPid = 0;
  pthread_mutex_destroy(&pxnDetachMutex);
  
  // Clean up the context registry
  pthread_mutex_lock(&contextRegistryMutex);
  contextRegistry.clear();
  pthread_mutex_unlock(&contextRegistryMutex);
  pthread_mutex_destroy(&contextRegistryMutex);
}

// ============================================================================
// SECTION 4: Utility Functions
// ============================================================================

// ----------------------------------------------------------------------------
// Time and Thread Helpers
// ----------------------------------------------------------------------------

// Get current timestamp in microseconds
static double getTime() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

// Get thread ID (Linux-specific)
static inline pid_t getTid() {
  return static_cast<pid_t>(syscall(SYS_gettid));
}

// ----------------------------------------------------------------------------
// JSON/JSONL Utility Functions
// ----------------------------------------------------------------------------

// Escape a JSON string and return as std::string (dynamic, no size limit)
static std::string escapeJsonStringStd(const char* input) {
  if (!input) return "";
  
  std::string result;
  result.reserve(strlen(input) * 2);  // Reserve space for worst case
  
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

// Format a pointer as std::string
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
// Each process writes to its own file to avoid interleaving issues on network filesystems
// (NFS, Lustre, GPFS do not honor O_APPEND atomicity guarantees)
static bool initSharedJsonlFile(const char* dirname) {
  const char* slurm_job_id = getenv("SLURM_JOB_ID");
  pid_t pid = getpid();
  
  if (slurm_job_id && slurm_job_id[0] != '\0') {
    // Use SLURM job ID for grouping, but actual PID for process identification
    snprintf(sharedJsonlPath, sizeof(sharedJsonlPath), 
             "%s/trace_%s_pid%d.jsonl", dirname, slurm_job_id, static_cast<int>(pid));
  } else {
    // No SLURM - use timestamp + PID for unique naming
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    snprintf(sharedJsonlPath, sizeof(sharedJsonlPath), 
             "%s/trace_%ld_pid%d.jsonl", dirname, ts.tv_sec, static_cast<int>(pid));
  }
  
  // Open in write mode - this file is exclusive to this process, no append/locking needed
  sharedJsonlFile = fopen(sharedJsonlPath, "w");
  if (!sharedJsonlFile) {
    fprintf(stderr, "[MinimalProfiler] WARNING: Failed to open JSONL file: %s\n", sharedJsonlPath);
    return false;
  }
  
  fprintf(stderr, "[MinimalProfiler] JSONL trace file: %s\n", sharedJsonlPath);
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
// JSONL Writers - Separate Records for Events and States
// ----------------------------------------------------------------------------
// Each JSONL line is small and atomic (<3KB). No buffering needed.
// - "event" records: written at STOP time with start/stop info and details
// - "state" records: written immediately at STATE time with eventAddr reference
// The visualizer reconstructs full events by matching states to their eventAddr.
// ----------------------------------------------------------------------------

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
  
  // Write directly to per-process file (no cross-process coordination needed)
  pthread_mutex_lock(&jsonlFileMutex);
  if (sharedJsonlFile) {
    int fd = fileno(sharedJsonlFile);
    if (fd >= 0) {
      ssize_t written = write(fd, json.c_str(), json.size());
      (void)written;
    }
  }
  pthread_mutex_unlock(&jsonlFileMutex);
}

// Write an event record to JSONL (called at STOP time - no embedded states)
static void writeJsonlEventRecord(MyEvent* event, double endTime, double duration, pid_t stopTid) {
  if (!event || !sharedJsonlFile) return;
  
  // Build the event JSON - (event states are separate records)
  std::string json = "{\"recordType\":\"event\"" +
                     std::string(",\"type\":\"") + escapeJsonStringStd(event->typeName ? event->typeName : "Unknown") + "\"";
  
  if (!event->fromDetachPool) {
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
  
  snprintf(buf, sizeof(buf), "%.3f", endTime);
  json += ",\"stop\":{\"ts\":" + std::string(buf) +
          ",\"pid\":" + std::to_string(static_cast<int>(getpid())) +
          ",\"tid\":" + std::to_string(static_cast<int>(stopTid)) + "}";
  
  snprintf(buf, sizeof(buf), "%.3f", duration);
  json += ",\"duration\":" + std::string(buf);
  
  json += ",\"myPid\":" + std::to_string(static_cast<int>(profilerPid));
  
  if (event->fromDetachPool) {
    json += ",\"originPid\":" + std::to_string(static_cast<int>(event->originPid));
  }
  
  json += ",\"parentObj\":\"" + formatPointerStd(event->parentObj) + "\"" +
          ",\"eventAddr\":\"" + formatPointerStd(static_cast<void*>(event)) + "\"";
  
  if (!event->fromDetachPool && event->ctx) {
    json += ",\"ctx\":\"" + formatPointerStd(static_cast<void*>(event->ctx)) + "\"";
  }
  
  if (event->fromDetachPool) {
    json += ",\"isPxn\":true";
  }
  
  // Add details (no states array - states are separate records)
  json += ",\"details\":{" + event->startDetails + "}}\n";
  
  // Write directly to per-process file (no cross-process coordination needed)
  pthread_mutex_lock(&jsonlFileMutex);
  if (sharedJsonlFile) {
    int fd = fileno(sharedJsonlFile);
    if (fd >= 0) {
      ssize_t written = write(fd, json.c_str(), json.size());
      (void)written;
    }
  }
  pthread_mutex_unlock(&jsonlFileMutex);
}

// Write a ProfilerLifecycle pseudo-event to JSONL
// Uses dynamic std::string - no fixed buffer size limits
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
                     ",\"myPid\":" + std::to_string(static_cast<int>(profilerPid)) +
                     ",\"states\":[]" +
                     ",\"details\":{" + detailsJson + "}}\n";
  
  // Write directly to per-process file (no cross-process coordination needed)
  pthread_mutex_lock(&jsonlFileMutex);
  if (sharedJsonlFile) {
    int fd = fileno(sharedJsonlFile);
    if (fd >= 0) {
      ssize_t written = write(fd, json.c_str(), json.size());
      (void)written;
    }
  }
  pthread_mutex_unlock(&jsonlFileMutex);
}

// ----------------------------------------------------------------------------
// Type and State Name Mappers
// ----------------------------------------------------------------------------

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
// SECTION 5: Profiler API Implementation
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
  
  fprintf(stderr, "[MinimalProfiler] Init called: tid=%d, rank=%d/%d, commId=%lu, eventMask=0x%x\n",
          tid, rank, nranks, commId, mask);
  
  // Get dump directory name (needed for JSONL trace file)
  char dirname[256];
  if (!getDumpDir(dirname, sizeof(dirname))) {
    return ncclSystemError;
  }
  
  // Create dump directory (safe to call multiple times)
  if (ensureDumpDir(dirname) != 0) {
    return ncclSystemError;
  }
  
  // Initialize process-wide state on first communicator init
  pthread_mutex_lock(&processStateMutex);
  if (activeContextCount == 0) {
    profilerPid = getpid();
    firstCommId = commId;  // Store commId for this profiler instance
    
    if (initPxnDetachPool()) {
      fprintf(stderr, "[MinimalProfiler] Init: profilerPid=%d, firstCommId=%lu\n",
              static_cast<int>(profilerPid), firstCommId);
    } else {
      fprintf(stderr, "[MinimalProfiler] WARNING: Failed to initialize detach pool\n");
    }
    
    // Initialize per-process JSONL trace file
    if (initSharedJsonlFile(dirname)) {
      fprintf(stderr, "[MinimalProfiler] JSONL trace file initialized\n");
    } else {
      fprintf(stderr, "[MinimalProfiler] WARNING: Failed to initialize JSONL file\n");
    }
  }
  activeContextCount++;
  pthread_mutex_unlock(&processStateMutex);
  
  // Allocate context
  MyContext* ctx = new (std::nothrow) MyContext();
  if (!ctx) return ncclSystemError;
  
  ctx->commId = commId;
  ctx->rank = rank;
  ctx->nranks = nranks;
  ctx->startTime = getTime();
  ctx->gpuUuid[0] = '\0';  // Initialize to empty string
  
  // Get GPU UUID from current CUDA device
  int dev;
  if (cudaGetDevice(&dev) == cudaSuccess) {
    cudaDeviceProp deviceProp;
    if (cudaGetDeviceProperties(&deviceProp, dev) == cudaSuccess) {
      // Format UUID as standard string: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
      snprintf(ctx->gpuUuid, sizeof(ctx->gpuUuid),
               "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
               deviceProp.uuid.bytes[0], deviceProp.uuid.bytes[1],
               deviceProp.uuid.bytes[2], deviceProp.uuid.bytes[3],
               deviceProp.uuid.bytes[4], deviceProp.uuid.bytes[5],
               deviceProp.uuid.bytes[6], deviceProp.uuid.bytes[7],
               deviceProp.uuid.bytes[8], deviceProp.uuid.bytes[9],
               deviceProp.uuid.bytes[10], deviceProp.uuid.bytes[11],
               deviceProp.uuid.bytes[12], deviceProp.uuid.bytes[13],
               deviceProp.uuid.bytes[14], deviceProp.uuid.bytes[15]);
    }
  }
  
  // Initialize mutex for pool/blacklist access
  if (pthread_mutex_init(&ctx->allocMutex, nullptr) != 0) {
    delete ctx;
    return ncclSystemError;
  }
  
  // Initialize event pool
  if (!ctx->pool.init(INITIAL_POOL_SIZE)) {
    pthread_mutex_destroy(&ctx->allocMutex);
    delete ctx;
    return ncclSystemError;
  }
  
  fprintf(stderr, "[MinimalProfiler] Successfully initialized: tid=%d, rank=%d, commId=%lu, pool_size=%zu\n", 
          tid, rank, commId, INITIAL_POOL_SIZE);
  
  // Register this context in the process-wide registry for PXN validation
  registerContext(ctx);
  
  // Write ProfilerInit lifecycle event to JSONL
  {
    std::string detailsJson = "\"nranks\":" + std::to_string(nranks) +
                              ",\"commName\":\"" + escapeJsonStringStd(commName ? commName : "") + "\"" +
                              ",\"eventMask\":" + std::to_string(mask);
    writeJsonlLifecycleEvent("ProfilerInit", ctx->startTime, ctx->gpuUuid, commId, rank, detailsJson);
  }
  
  *context = ctx;
  return ncclSuccess;
}

// ----------------------------------------------------------------------------
// myStartEvent: Begin tracking a profiler event
// ----------------------------------------------------------------------------
ncclResult_t myStartEvent(void* context, void** eHandle, 
                          ncclProfilerEventDescr_v5_t* eDescr) {
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
  // 1. For ProxyOp events: check eDescr->proxyOp.pid against our profilerPid
  // 2. For ALL events: check if context is in our registry of valid contexts
  //
  // If the context is invalid, we treat the event as "detached" and use the
  // process-wide detach pool instead.
  // ========================================================================
  
  // First, check the context registry to see if this is a valid context
  // This catches ALL event types with invalid contexts (not just ProxyOp)
  bool contextValid = isValidContext(context);
  
  // For ProxyOp events, we can also check the pid field for more information
  pid_t eventPid = 0;
  if (eDescr->type == ncclProfileProxyOp) {
    eventPid = eDescr->proxyOp.pid;
    // If pid doesn't match, context is definitely from another process
    if (eventPid != profilerPid) {
      contextValid = false;
    }
  }
  
  // Allocate event from appropriate pool based on context validity
  if (!contextValid) {
    // Detached PXN event - ctx is invalid, use process-wide detach pool
    event = allocateFromPxnPool();
    if (!event) {
      *eHandle = nullptr;
      return ncclSuccess;  // Pool allocation failed - drop this event gracefully
    }
    
    // Set up basic event fields for detached event
    event->type = eDescr->type;
    event->ctx = nullptr;  // ctx is invalid for PXN events - don't store it
    event->startTime = getTime();
    event->typeName = getEventTypeName(eDescr->type);
    event->parentObj = eDescr->parentObj;
    event->originPid = eventPid;  // Will be 0 for non-ProxyOp events
    
    // Initialize new fields for JSONL output (PXN events have no ctx access)
    event->startTid = getTid();
    event->startPid = getpid();
    event->gpuUuid[0] = '\0';  // Not available for PXN
    event->commId = 0;          // Not available for PXN
    event->rank = -1;           // Not available for PXN
    event->startDetails.clear(); // Will be populated in switch below
    
    // Track parentObj in detach blacklist to prevent address reuse
    trackPxnParent(eDescr->parentObj);
  } else {
    // Normal event - context is valid, use per-context pool
    pthread_mutex_lock(&ctx->allocMutex);
    event = ctx->pool.allocate(ctx->blacklist);
    pthread_mutex_unlock(&ctx->allocMutex);
    
    if (!event) {
      *eHandle = nullptr;
      return ncclSuccess;  // Fail gracefully
    }
    
    // Set up basic event fields for normal event
    event->type = eDescr->type;
    event->ctx = ctx;
    event->startTime = getTime();
    event->typeName = getEventTypeName(eDescr->type);
    event->parentObj = eDescr->parentObj;
    event->fromDetachPool = false;
    event->originPid = getpid();
    
    // Initialize new fields for JSONL output - capture ctx-dependent fields NOW
    event->startTid = getTid();
    event->startPid = getpid();
    strncpy(event->gpuUuid, ctx->gpuUuid, sizeof(event->gpuUuid) - 1);
    event->gpuUuid[sizeof(event->gpuUuid) - 1] = '\0';
    event->commId = ctx->commId;
    event->rank = ctx->rank;
    event->startDetails.clear(); // Will be populated in switch below
    
    // Handle parent reference (add to blacklist) - only for local events
    if (eDescr->parentObj) {
      pthread_mutex_lock(&ctx->allocMutex);
      if (ctx->blacklist.contains(eDescr->parentObj)) {
        ctx->blacklist.incrementRef(eDescr->parentObj);
      } else {
        ctx->blacklist.add(eDescr->parentObj);
      }
      pthread_mutex_unlock(&ctx->allocMutex);
    }
  }
  
  // Build startDetails JSON for later JSONL output (stored in event, written on STOP)
  // Uses std::string - no fixed buffer size limits
  switch (eDescr->type) {
    case ncclProfileGroupApi:
      event->startDetails = "\"depth\":" + std::to_string(eDescr->groupApi.groupDepth) +
                            ",\"graphCaptured\":" + (eDescr->groupApi.graphCaptured ? "true" : "false");
      break;
    case ncclProfileCollApi:
      event->startDetails = "\"func\":\"" + escapeJsonStringStd(eDescr->collApi.func) + "\"" +
                            ",\"count\":" + std::to_string(eDescr->collApi.count) +
                            ",\"dtype\":\"" + escapeJsonStringStd(eDescr->collApi.datatype) + "\"" +
                            ",\"root\":" + std::to_string(eDescr->collApi.root) +
                            ",\"stream\":\"" + formatPointerStd(eDescr->collApi.stream) + "\"" +
                            ",\"graphCaptured\":" + (eDescr->collApi.graphCaptured ? "true" : "false");
      break;
    case ncclProfileP2pApi:
      event->startDetails = "\"func\":\"" + escapeJsonStringStd(eDescr->p2pApi.func) + "\"" +
                            ",\"count\":" + std::to_string(eDescr->p2pApi.count) +
                            ",\"dtype\":\"" + escapeJsonStringStd(eDescr->p2pApi.datatype) + "\"" +
                            ",\"stream\":\"" + formatPointerStd(eDescr->p2pApi.stream) + "\"" +
                            ",\"graphCaptured\":" + (eDescr->p2pApi.graphCaptured ? "true" : "false");
      break;
    case ncclProfileKernelLaunch:
      event->startDetails = "\"stream\":\"" + formatPointerStd(eDescr->kernelLaunch.stream) + "\"";
      break;
    case ncclProfileColl:
      event->startDetails = "\"func\":\"" + escapeJsonStringStd(eDescr->coll.func) + "\"" +
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
      event->startDetails = "\"func\":\"" + escapeJsonStringStd(eDescr->p2p.func) + "\"" +
                            ",\"peer\":" + std::to_string(eDescr->p2p.peer) +
                            ",\"channels\":" + std::to_string(eDescr->p2p.nChannels) +
                            ",\"count\":" + std::to_string(eDescr->p2p.count) +
                            ",\"dtype\":\"" + escapeJsonStringStd(eDescr->p2p.datatype) + "\"" +
                            ",\"buff\":\"" + formatPointerStd(eDescr->p2p.buff) + "\"" +
                            ",\"parentGroup\":\"" + formatPointerStd(eDescr->p2p.parentGroup) + "\"";
      break;
    case ncclProfileProxyOp:
      event->startDetails = "\"channel\":" + std::to_string(eDescr->proxyOp.channelId) +
                            ",\"peer\":" + std::to_string(eDescr->proxyOp.peer) +
                            ",\"steps\":" + std::to_string(eDescr->proxyOp.nSteps) +
                            ",\"chunkSize\":" + std::to_string(eDescr->proxyOp.chunkSize) +
                            ",\"isSend\":" + std::string(eDescr->proxyOp.isSend ? "true" : "false") +
                            ",\"eventPid\":" + std::to_string(static_cast<int>(eDescr->proxyOp.pid));
      break;
    case ncclProfileProxyStep:
      event->startDetails = "\"step\":" + std::to_string(eDescr->proxyStep.step);
      break;
    case ncclProfileProxyCtrl:
      // No extra details
      break;
    case ncclProfileKernelCh:
      event->startDetails = "\"channel\":" + std::to_string(eDescr->kernelCh.channelId) +
                            ",\"pTimer\":" + std::to_string(eDescr->kernelCh.pTimer);
      break;
    case ncclProfileNetPlugin:
      event->startDetails = "\"id\":" + std::to_string(static_cast<long>(eDescr->netPlugin.id)) +
                            ",\"data\":\"" + formatPointerStd(eDescr->netPlugin.data) + "\"";
      break;
    case ncclProfileGroup:
      // No extra details
      break;
    default:
      {
        char buf[32];
        snprintf(buf, sizeof(buf), "0x%lx", static_cast<unsigned long>(eDescr->type));
        event->startDetails = "\"rawType\":\"" + std::string(buf) + "\"";
      }
      break;
  }
  // Note: JSONL output is now written in myStopEvent as a complete event
  
  *eHandle = event;
  return ncclSuccess;
}

// ----------------------------------------------------------------------------
// myStopEvent: End tracking a profiler event
// ----------------------------------------------------------------------------
ncclResult_t myStopEvent(void* eHandle) {
  if (!eHandle) return ncclSuccess;
  
  MyEvent* event = static_cast<MyEvent*>(eHandle);
  MyContext* ctx = event->ctx;
  double endTime = getTime();
  double duration = (endTime - event->startTime);
  pid_t tid = getTid();
  
  // Handle detached PXN ProxyOp events from the detach pool
  if (event->fromDetachPool) {
    // Write JSONL event (PXN)
    writeJsonlEventRecord(event, endTime, duration, tid);
    
    // Release parent reference and free to detach pool
    releasePxnParent(event->parentObj);
    freeToPxnPool(event);
    
    return ncclSuccess;
  }
  
  // Normal event (from per-context pool) - ctx must be valid
  if (!ctx) {
    return ncclSuccess;  // Drop event if ctx is somehow NULL
  }
  
  // Write JSONL event (normal)
  writeJsonlEventRecord(event, endTime, duration, tid);
  
  // Decrement parent reference in blacklist
  if (event->parentObj) {
    pthread_mutex_lock(&ctx->allocMutex);
    ctx->blacklist.decrementRef(event->parentObj);
    pthread_mutex_unlock(&ctx->allocMutex);
  }
  
  // Free the event
  pthread_mutex_lock(&ctx->allocMutex);
  if (event->fromPool) {
    ctx->pool.free(event);
  } else {
    delete event;
  }
  pthread_mutex_unlock(&ctx->allocMutex);
  
  return ncclSuccess;
}

// ----------------------------------------------------------------------------
// myRecordEventState: Record state transitions during event lifecycle
// ----------------------------------------------------------------------------
ncclResult_t myRecordEventState(void* eHandle, 
                                 ncclProfilerEventState_v5_t eState,
                                 ncclProfilerEventStateArgs_v5_t* eStateArgs) {
  if (!eHandle) return ncclSuccess;
  
  MyEvent* event = static_cast<MyEvent*>(eHandle);
  MyContext* ctx = event->ctx;
  double currentTime = getTime();
  pid_t tid = getTid();
  pid_t pid = getpid();
  
  const char* stateName = getStateName(eState);
  
  // Handle PXN events (ctx is nullptr)
  if (event->fromDetachPool) {
    // Write state to JSONL immediately
    std::string details;
    if (eStateArgs) {
      switch (eState) {
        case ncclProfilerProxyStepSendWait:
        case ncclProfilerProxyStepRecvFlushWait:
          details = "\"transSize\":" + std::to_string(eStateArgs->proxyStep.transSize);
          break;
        default:
          break;
      }
    }
    writeJsonlStateRecord(event, currentTime, stateName, static_cast<int>(eState), pid, tid, details);
    return ncclSuccess;
  }
  
  // Normal event - ctx must be valid
  if (!ctx) {
    return ncclSuccess;  // Drop event if ctx is NULL
  }
  
  // Write state to JSONL immediately
  {
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
    writeJsonlStateRecord(event, currentTime, stateName, static_cast<int>(eState), pid, tid, details);
  }
  
  return ncclSuccess;
}

// ----------------------------------------------------------------------------
// myFinalize: Clean up profiler context for a communicator
// ----------------------------------------------------------------------------
ncclResult_t myFinalize(void* context) {
  MyContext* ctx = static_cast<MyContext*>(context);
  pid_t tid = getTid();
  double endTime = getTime();
  double totalDuration = endTime - ctx->startTime;
  
  // Write ProfilerFinalize lifecycle event to JSONL BEFORE unregistering/cleanup
  {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.3f", totalDuration);
    std::string detailsJson = "\"totalDuration\":" + std::string(buf) +
                              ",\"poolChunks\":" + std::to_string(ctx->pool.chunks.size()) +
                              ",\"poolSlots\":" + std::to_string(ctx->pool.chunks.size() * ctx->pool.chunkSize) +
                              ",\"blacklistSize\":" + std::to_string(ctx->blacklist.map.size());
    writeJsonlLifecycleEvent("ProfilerFinalize", endTime, ctx->gpuUuid, ctx->commId, ctx->rank, detailsJson);
  }
  
  // Unregister this context from the process-wide registry
  // Do this FIRST before any other cleanup to prevent race conditions
  // where another thread might try to validate this context
  unregisterContext(ctx);
  
  // Destroy mutex
  pthread_mutex_destroy(&ctx->allocMutex);
  
  fprintf(stderr, "[MinimalProfiler] Finalized: tid=%d, rank=%d, commId=%lu, final_pool_chunks=%zu, final_pool_slots=%zu, final_blacklist_size=%zu\n", 
          tid, ctx->rank, ctx->commId, ctx->pool.chunks.size(), 
          ctx->pool.chunks.size() * ctx->pool.chunkSize, ctx->blacklist.map.size());
  
  delete ctx;
  
  // Clean up detach pool when last communicator is finalized
  pthread_mutex_lock(&processStateMutex);
  activeContextCount--;
  if (activeContextCount == 0) {
    cleanupProcessResources();
    fprintf(stderr, "[MinimalProfiler] Cleanup: detach pool and blacklist freed\n");
  }
  pthread_mutex_unlock(&processStateMutex);
  
  return ncclSuccess;
}

// ============================================================================
// SECTION 6: Export Struct
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
