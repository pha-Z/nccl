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

// Include CUDA runtime for device ID queries
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
// Note: We use dynamic std::string and std::vector for JSON construction,
// so there are no fixed buffer size limits. Events can have any number of
// state transitions and any size of details.

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
  void evictLRU(FILE* logFile = nullptr) {
    if (!head) return;
    
    BlacklistNode* node = head;
    
    // Log warning about forced eviction
    if (logFile) {
      fprintf(logFile, "[WARNING] Blacklist cap reached (%zu), evicting address %p with refCount=%d. "
              "Potential false parent-child match may occur.\n",
              cap, node->address, node->refCount);
    }
    
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
  void add(void* addr, FILE* logFile = nullptr) {
    // Check if we need to evict
    if (map.size() >= cap) {
      evictLRU(logFile);
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
// StateTransition: Stores a single state transition during an event's lifecycle
// Uses dynamic std::string for details - no size limit
// ----------------------------------------------------------------------------
struct StateTransition {
  double timestamp;
  std::string stateName;
  int stateId;
  pid_t tid;
  pid_t pid;
  std::string details;  // State-specific details as JSON fragment (dynamic)
};

// ----------------------------------------------------------------------------
// MyEvent: Stores event data + parent reference for cleanup
// Extended to support buffering complete events for JSONL output
// Uses dynamic containers - no fixed size limits
// ----------------------------------------------------------------------------
struct MyEvent {
  // Basic event info (set at START)
  uint64_t type;
  double startTime;
  const char* func;
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
  
  // State transitions (accumulated during event lifetime) - dynamic vector, no limit
  std::vector<StateTransition> states;
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
  bool grow(FILE* logFile = nullptr) {
    // Allocate a new chunk (same size as existing chunks)
    MyEvent* newChunk = new (std::nothrow) MyEvent[chunkSize]();
    bool* newInUse = new (std::nothrow) bool[chunkSize]();
    
    if (!newChunk || !newInUse) {
      delete[] newChunk;
      delete[] newInUse;
      if (logFile) {
        fprintf(logFile, "[WARNING] Failed to allocate new chunk of size %zu\n", chunkSize);
      }
      return false;
    }
    
    // Initialize all slots as not in use
    for (size_t i = 0; i < chunkSize; i++) {
      newInUse[i] = false;
    }
    
    chunks.push_back(newChunk);
    inUseChunks.push_back(newInUse);
    
    if (logFile) {
      fprintf(logFile, "[INFO] Event pool grown: added chunk %zu (total chunks: %zu, total slots: %zu)\n",
              chunks.size() - 1, chunks.size(), chunks.size() * chunkSize);
    }
    
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
  MyEvent* allocate(Blacklist& blacklist, FILE* logFile = nullptr) {
    // Defensive check: ensure we have at least one chunk
    if (chunks.empty() || chunkSize == 0) {
      if (logFile) {
        fprintf(logFile, "[ERROR] EventPool not initialized (chunks.empty() or chunkSize==0)\n");
      }
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
    if (grow(logFile)) {
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
    if (logFile) {
      fprintf(logFile, "[WARNING] Pool exhausted and growth failed, using dynamic allocation\n");
    }
    
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
// ----------------------------------------------------------------------------
// MyContext: Per-communicator plugin state
// ----------------------------------------------------------------------------
struct MyContext {
  FILE* logFile;
  pthread_mutex_t logMutex;     // Protects logFile I/O
  pthread_mutex_t allocMutex;   // Protects pool and blacklist
  uint64_t commId;
  int rank;
  int nranks;
  double startTime;
  int cudaDev;                  // Physical GPU device ID (set once during init, never changes)
  char gpuUuid[37];             // Physical GPU UUID string (36 chars + null terminator)
  
  EventPool pool;
  Blacklist blacklist;
};

// ============================================================================
// SECTION 3: PXN Detach Pool Module
// ============================================================================
// When PXN (PCI x NVLink) routing is enabled, some ProxyOp events may be
// executed by a different process than the one that originated them. The
// parentObj pointer in such events points to the originator's address space,
// which is invalid in the executing process.
//
// These events are "detached" because we cannot resolve their parent pointer.
// We check eDescr->proxyOp.pid against our local myPid - if they differ, the
// event originated from another process and its parentObj cannot be dereferenced.
//
// Detached events are stored in a separate pool shared across all contexts in
// this process. We store parentObj as metadata (pointer value) for tracking
// purposes but never dereference it.
// ============================================================================

// Forward declaration for utility function used in this section
static double getTime();

// Global state for PXN detach pool (shared across all contexts in this process)
static pthread_mutex_t processInitMutex = PTHREAD_MUTEX_INITIALIZER;
static int processInitCount = 0;             // Number of initialized communicators in this process
static pid_t myPid = 0;                      // This process's PID (profiler plugin's process)
static uint64_t myCommId = 0;                // commId for this profiler instance
static EventPool* detachPool = nullptr;      // Pool for detached PXN ProxyOp events
static Blacklist* detachBlacklist = nullptr; // Blacklist for detach pool
static pthread_mutex_t detachMutex = PTHREAD_MUTEX_INITIALIZER;  // Protects detach pool and blacklist
static FILE* detachLogFile = nullptr;        // Process-wide log file for PXN events
static pthread_mutex_t detachLogMutex = PTHREAD_MUTEX_INITIALIZER;  // Protects detach log file
static double detachStartTime = 0;           // Start time for PXN event timestamps

// Global shared JSONL trace file (all ranks write to same file using O_APPEND)
static FILE* sharedJsonlFile = nullptr;      // Shared trace file (opened with append mode)
static pthread_mutex_t jsonlFileMutex = PTHREAD_MUTEX_INITIALIZER;  // Protects sharedJsonlFile writes
static char sharedJsonlPath[512] = {0};      // Path to shared JSONL file

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

// Initialize the detach pool (called on first communicator init)
static bool initDetachPool() {
  detachPool = new (std::nothrow) EventPool();
  detachBlacklist = new (std::nothrow) Blacklist();
  
  if (!detachPool || !detachBlacklist) {
    delete detachPool;
    delete detachBlacklist;
    detachPool = nullptr;
    detachBlacklist = nullptr;
    return false;
  }
  
  if (!detachPool->init(INITIAL_DETACH_POOL_SIZE)) {
    delete detachPool;
    delete detachBlacklist;
    detachPool = nullptr;
    detachBlacklist = nullptr;
    return false;
  }
  
  return true;
}

// Initialize the PXN log file (called on first communicator init)
static bool initPxnLogFile(const char* dirname, uint64_t commId) {
  // Create process-wide log file for PXN events with commId in filename
  char pxnLogPath[512];
  snprintf(pxnLogPath, sizeof(pxnLogPath), "%s/minimal_profiler_pxn_events_pid%d_comm%lu.log", 
           dirname, static_cast<int>(myPid), commId);
  detachLogFile = fopen(pxnLogPath, "w");
  if (detachLogFile) {
    detachStartTime = getTime();
    fprintf(detachLogFile, "=== PXN Events Log (Process PID: %d, CommId: %lu) ===\n", 
            static_cast<int>(myPid), commId);
    fprintf(detachLogFile, "CommId: %lu\n", commId);
    fprintf(detachLogFile, "Start time: %.0f us\n", detachStartTime);
    fprintf(detachLogFile, "This file contains ProxyOp events from other processes (PXN routing)\n");
    fprintf(detachLogFile, "================================================\n\n");
    fflush(detachLogFile);
  }
  // Note: We continue even if log file creation fails - events just won't be logged
  
  return detachLogFile != nullptr;
}

// Allocate an event from the detach pool
static MyEvent* allocateFromDetachPool(FILE* logFile = nullptr) {
  if (!detachPool || !detachBlacklist) {
    return nullptr;
  }
  
  pthread_mutex_lock(&detachMutex);
  MyEvent* event = detachPool->allocate(*detachBlacklist, logFile);
  pthread_mutex_unlock(&detachMutex);
  
  if (event) {
    event->fromDetachPool = true;
    event->fromPool = true;
  }
  
  return event;
}

// Free an event back to the detach pool
static void freeToDetachPool(MyEvent* event) {
  if (!detachPool) return;
  
  pthread_mutex_lock(&detachMutex);
  detachPool->free(event);
  pthread_mutex_unlock(&detachMutex);
}

// Track a parent address in the detach blacklist
static void trackDetachParent(void* parentObj, FILE* logFile = nullptr) {
  if (!parentObj || !detachBlacklist) return;
  
  pthread_mutex_lock(&detachMutex);
  if (detachBlacklist->contains(parentObj)) {
    detachBlacklist->incrementRef(parentObj);
  } else {
    detachBlacklist->add(parentObj, logFile);
  }
  pthread_mutex_unlock(&detachMutex);
}

// Release a parent address from the detach blacklist
static void releaseDetachParent(void* parentObj) {
  if (!parentObj || !detachBlacklist) return;
  
  pthread_mutex_lock(&detachMutex);
  detachBlacklist->decrementRef(parentObj);
  pthread_mutex_unlock(&detachMutex);
}

// Clean up the detach pool and log file (called when last communicator is finalized)
static void cleanupDetachPool() {
  // Close the PXN log file
  if (detachLogFile) {
    fprintf(detachLogFile, "\n=== PXN Events Log Finalized ===\n");
    fclose(detachLogFile);
    detachLogFile = nullptr;
  }
  
  // Close the shared JSONL file
  closeSharedJsonlFile();
  pthread_mutex_destroy(&jsonlFileMutex);
  
  delete detachPool;
  delete detachBlacklist;
  detachPool = nullptr;
  detachBlacklist = nullptr;
  myPid = 0;
  detachStartTime = 0;
  pthread_mutex_destroy(&detachMutex);
  pthread_mutex_destroy(&detachLogMutex);
  
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

// Initialize the shared JSONL trace file (called on first communicator init)
static bool initSharedJsonlFile(const char* dirname) {
  const char* slurm_job_id = getenv("SLURM_JOB_ID");
  
  if (slurm_job_id && slurm_job_id[0] != '\0') {
    snprintf(sharedJsonlPath, sizeof(sharedJsonlPath), 
             "%s/trace_%s.jsonl", dirname, slurm_job_id);
  } else {
    // Fallback to timestamp-based naming
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    snprintf(sharedJsonlPath, sizeof(sharedJsonlPath), 
             "%s/trace_%ld.jsonl", dirname, ts.tv_sec);
  }
  
  // Open in append mode (O_APPEND) - POSIX guarantees atomic appends for complete lines
  sharedJsonlFile = fopen(sharedJsonlPath, "a");
  if (!sharedJsonlFile) {
    fprintf(stderr, "[MinimalProfiler] WARNING: Failed to open shared JSONL file: %s\n", sharedJsonlPath);
    return false;
  }
  
  fprintf(stderr, "[MinimalProfiler] Shared JSONL trace file: %s\n", sharedJsonlPath);
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
// Logging Helper Functions
// ----------------------------------------------------------------------------

// Helper function to log START events (handles both normal and detached PXN events)
static void logStartEvent(MyEvent* event, void* parentObj, const char* fmt, ...) {
  if (!event) return;
  
  pid_t pid = getpid();
  pid_t tid = getTid();
  
  // Check if this is a detached PXN event
  if (event->fromDetachPool) {
    // Use process-wide detach log file
    if (!detachLogFile) return;
    
    pthread_mutex_lock(&detachLogMutex);
    if (*fmt != '\0') {
      va_list args;
      va_start(args, fmt);
      fprintf(detachLogFile, "[%.3f] START %s (pid=%d, tid=%d, PXN from pid=%d, ", 
              event->startTime - detachStartTime, event->typeName, 
              static_cast<int>(pid), tid, static_cast<int>(event->originPid));
      vfprintf(detachLogFile, fmt, args);
      va_end(args);
      fprintf(detachLogFile, ", parentObj=%p, event=%p)\n",
              parentObj, static_cast<void*>(event));
    } else {
      fprintf(detachLogFile, "[%.3f] START %s (pid=%d, tid=%d, PXN from pid=%d, parentObj=%p, event=%p)\n",
              event->startTime - detachStartTime, event->typeName, 
              static_cast<int>(pid), tid, static_cast<int>(event->originPid),
              parentObj, static_cast<void*>(event));
    }
    fflush(detachLogFile);
    pthread_mutex_unlock(&detachLogMutex);
  } else {
    // Normal event - use context's log file
    MyContext* ctx = event->ctx;
    if (!ctx || !ctx->logFile) return;
    
    pthread_mutex_lock(&ctx->logMutex);
    if (*fmt != '\0') {
      va_list args;
      va_start(args, fmt);
      fprintf(ctx->logFile, "[%.3f] START %s (pid=%d, tid=%d, gpu_uuid=%s, ", 
              event->startTime, event->typeName, static_cast<int>(pid), tid, 
              ctx->gpuUuid[0] != '\0' ? ctx->gpuUuid : "unknown");
      vfprintf(ctx->logFile, fmt, args);
      va_end(args);
      fprintf(ctx->logFile, ", parentObj=%p, event=%p, ctx=%p)\n",
              parentObj, static_cast<void*>(event), static_cast<void*>(ctx));
    } else {
      fprintf(ctx->logFile, "[%.3f] START %s (pid=%d, tid=%d, gpu_uuid=%s, parentObj=%p, event=%p, ctx=%p)\n",
              event->startTime, event->typeName, static_cast<int>(pid), tid, 
              ctx->gpuUuid[0] != '\0' ? ctx->gpuUuid : "unknown",
              parentObj, static_cast<void*>(event), static_cast<void*>(ctx));
    }
    fflush(ctx->logFile);
    pthread_mutex_unlock(&ctx->logMutex);
  }
}

// ----------------------------------------------------------------------------
// JSONL Complete Event Writer
// Writes a single complete event with all accumulated states (called at STOP)
// Uses dynamic std::string - no fixed buffer size limits
// ----------------------------------------------------------------------------

static void writeJsonlCompleteEvent(MyEvent* event, double endTime, double duration, pid_t stopTid) {
  if (!event || !sharedJsonlFile) return;
  
  // Build the complete event JSON using std::string (dynamic, no size limits)
  std::string json = "{\"type\":\"" + escapeJsonStringStd(event->typeName ? event->typeName : "Unknown") + "\"" +
                     ",\"func\":\"" + escapeJsonStringStd(event->func ? event->func : "Unknown") + "\"";
  
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
  
  json += ",\"myPid\":" + std::to_string(static_cast<int>(myPid));
  
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
  
  // Build states array - dynamic, no limit on number of states
  json += ",\"states\":[";
  for (size_t i = 0; i < event->states.size(); i++) {
    const StateTransition& st = event->states[i];
    if (i > 0) json += ",";
    
    snprintf(buf, sizeof(buf), "%.3f", st.timestamp);
    json += "{\"ts\":" + std::string(buf) +
            ",\"name\":\"" + escapeJsonStringStd(st.stateName.c_str()) + "\"" +
            ",\"id\":" + std::to_string(st.stateId) +
            ",\"pid\":" + std::to_string(static_cast<int>(st.pid)) +
            ",\"tid\":" + std::to_string(static_cast<int>(st.tid));
    
    if (!st.details.empty()) {
      json += "," + st.details;
    }
    json += "}";
  }
  json += "]";
  
  // Add details
  json += ",\"details\":{" + event->startDetails + "}}";
  
  // Write the complete JSON line
  pthread_mutex_lock(&jsonlFileMutex);
  fprintf(sharedJsonlFile, "%s\n", json.c_str());
  fflush(sharedJsonlFile);
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
                     ",\"myPid\":" + std::to_string(static_cast<int>(myPid)) +
                     ",\"states\":[]" +
                     ",\"details\":{" + detailsJson + "}}";
  
  pthread_mutex_lock(&jsonlFileMutex);
  fprintf(sharedJsonlFile, "%s\n", json.c_str());
  fflush(sharedJsonlFile);
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
  
  // Get dump directory name (needed for both per-context logs and PXN log)
  char dirname[256];
  if (!getDumpDir(dirname, sizeof(dirname))) {
    return ncclSystemError;
  }
  
  // Create dump directory (safe to call multiple times)
  if (ensureDumpDir(dirname) != 0) {
    return ncclSystemError;
  }
  
  // Initialize process-wide state on first communicator init
  pthread_mutex_lock(&processInitMutex);
  if (processInitCount == 0) {
    myPid = getpid();
    myCommId = commId;  // Store commId for this profiler instance
    
    if (initDetachPool()) {
      fprintf(stderr, "[MinimalProfiler] Init: myPid=%d, myCommId=%lu\n",
              static_cast<int>(myPid), myCommId);
    } else {
      fprintf(stderr, "[MinimalProfiler] WARNING: Failed to initialize detach pool\n");
    }
    if (initPxnLogFile(dirname, commId)) {
      fprintf(stderr, "[MinimalProfiler] PXN log file initialized\n");
    } else {
      fprintf(stderr, "[MinimalProfiler] WARNING: Failed to initialize PXN log file\n");
    }
    
    // Initialize shared JSONL trace file (all ranks write to same file)
    if (initSharedJsonlFile(dirname)) {
      fprintf(stderr, "[MinimalProfiler] Shared JSONL trace file initialized\n");
    } else {
      fprintf(stderr, "[MinimalProfiler] WARNING: Failed to initialize shared JSONL file\n");
    }
  }
  processInitCount++;
  pthread_mutex_unlock(&processInitMutex);
  
  // Allocate context
  MyContext* ctx = new (std::nothrow) MyContext();
  if (!ctx) return ncclSystemError;
  
  ctx->commId = commId;
  ctx->rank = rank;
  ctx->nranks = nranks;
  ctx->startTime = getTime();
  ctx->logFile = nullptr;
  ctx->cudaDev = -1;  // Initialize to invalid value
  ctx->gpuUuid[0] = '\0';  // Initialize to empty string
  
  // Get the CUDA device ID for this communicator (set once, never changes)
  cudaError_t cudaErr = cudaGetDevice(&ctx->cudaDev);
  if (cudaErr != cudaSuccess) {
    // If CUDA query fails, log warning but continue (device will be -1)
    fprintf(stderr, "[MinimalProfiler] WARNING: Failed to get CUDA device: %s\n", 
            cudaGetErrorString(cudaErr));
    ctx->cudaDev = -1;
  } else {
    // Get device properties to extract UUID (more reliable than device ID)
    cudaDeviceProp deviceProp;
    cudaErr = cudaGetDeviceProperties(&deviceProp, ctx->cudaDev);
    if (cudaErr == cudaSuccess) {
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
    } else {
      fprintf(stderr, "[MinimalProfiler] WARNING: Failed to get device properties: %s\n", 
              cudaGetErrorString(cudaErr));
    }
  }
  
  // Initialize mutexes
  if (pthread_mutex_init(&ctx->logMutex, nullptr) != 0) {
    delete ctx;
    return ncclSystemError;
  }
  if (pthread_mutex_init(&ctx->allocMutex, nullptr) != 0) {
    pthread_mutex_destroy(&ctx->logMutex);
    delete ctx;
    return ncclSystemError;
  }
  
  // Initialize event pool
  if (!ctx->pool.init(INITIAL_POOL_SIZE)) {
    pthread_mutex_destroy(&ctx->logMutex);
    pthread_mutex_destroy(&ctx->allocMutex);
    delete ctx;
    return ncclSystemError;
  }
  
  // Open log file (dump directory already created above)
  char filename[512];
  snprintf(filename, sizeof(filename), "%s/minimal_profiler_rank%d_comm%lu.log", 
           dirname, rank, commId);
  ctx->logFile = fopen(filename, "w");
  if (!ctx->logFile) {
    pthread_mutex_destroy(&ctx->logMutex);
    pthread_mutex_destroy(&ctx->allocMutex);
    delete ctx;
    return ncclSystemError;
  }
  
  // Write header to log file
  pthread_mutex_lock(&ctx->logMutex);
  fprintf(ctx->logFile, "=== Profiler initialized ===\n");
  fprintf(ctx->logFile, "Process ID (pid): %d\n", static_cast<int>(myPid));
  fprintf(ctx->logFile, "Init thread ID (tid): %d\n", tid);
  fprintf(ctx->logFile, "Start time: %.0f us\n", ctx->startTime);
  fprintf(ctx->logFile, "Dump directory: %s\n", dirname);
  fprintf(ctx->logFile, "Event activation mask: 0x%x\n", mask);
  fprintf(ctx->logFile, "Event pool size: %zu\n", INITIAL_POOL_SIZE);
  fprintf(ctx->logFile, "Blacklist cap: %zu\n", BLACKLIST_CAP);
  fprintf(ctx->logFile, "Physical GPU device ID: %d\n", ctx->cudaDev);
  fprintf(ctx->logFile, "Physical GPU UUID: %s\n", ctx->gpuUuid[0] != '\0' ? ctx->gpuUuid : "(unknown)");
  fprintf(ctx->logFile, "Rank: %d/%d, CommId: %lu, CommName: %s, ctx=%p\n", 
           rank, nranks, commId, commName ? commName : "(null)", static_cast<void*>(ctx));
  fflush(ctx->logFile);
  pthread_mutex_unlock(&ctx->logMutex);
  
  fprintf(stderr, "[MinimalProfiler] Successfully initialized: tid=%d, log=%s, pool_size=%zu\n", 
          tid, filename, INITIAL_POOL_SIZE);
  
  // Register this context in the process-wide registry for PXN validation
  registerContext(ctx);
  
  // Write ProfilerInit lifecycle event to JSONL
  {
    std::string detailsJson = "\"nranks\":" + std::to_string(nranks) +
                              ",\"commName\":\"" + escapeJsonStringStd(commName ? commName : "") + "\"" +
                              ",\"eventMask\":" + std::to_string(mask) +
                              ",\"cudaDev\":" + std::to_string(ctx->cudaDev);
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
  // 1. For ProxyOp events: check eDescr->proxyOp.pid against our myPid
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
    if (eventPid != myPid) {
      contextValid = false;
    }
  }
  
  // Allocate event from appropriate pool based on context validity
  if (!contextValid) {
    // Detached PXN event - ctx is invalid, use process-wide detach pool
    event = allocateFromDetachPool(detachLogFile);
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
    event->states.clear();      // Ensure empty vector
    event->startDetails.clear(); // Will be populated in switch below
    
    // Track parentObj in detach blacklist to prevent address reuse
    trackDetachParent(eDescr->parentObj, detachLogFile);
  } else {
    // Normal event - context is valid, use per-context pool
    pthread_mutex_lock(&ctx->allocMutex);
    event = ctx->pool.allocate(ctx->blacklist, ctx->logFile);
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
    event->states.clear();       // Ensure empty vector
    event->startDetails.clear(); // Will be populated in switch below
    
    // Handle parent reference (add to blacklist) - only for local events
    if (eDescr->parentObj) {
      pthread_mutex_lock(&ctx->allocMutex);
      if (ctx->blacklist.contains(eDescr->parentObj)) {
        ctx->blacklist.incrementRef(eDescr->parentObj);
      } else {
        ctx->blacklist.add(eDescr->parentObj, ctx->logFile);
      }
      pthread_mutex_unlock(&ctx->allocMutex);
    }
  }
  
  // Extract information based on event type and log
  switch (eDescr->type) {
    case ncclProfileGroupApi:
      event->func = "GroupAPI";
      logStartEvent(event, eDescr->parentObj, "depth=%d, graphCaptured=%d",
                    eDescr->groupApi.groupDepth, eDescr->groupApi.graphCaptured ? 1 : 0);
      break;
      
    case ncclProfileCollApi:
      event->func = eDescr->collApi.func;
      logStartEvent(event, eDescr->parentObj, "func=%s, count=%zu, dtype=%s, root=%d, stream=%p, graphCaptured=%d",
                    eDescr->collApi.func, eDescr->collApi.count, eDescr->collApi.datatype,
                    eDescr->collApi.root, eDescr->collApi.stream, eDescr->collApi.graphCaptured ? 1 : 0);
      break;
      
    case ncclProfileP2pApi:
      event->func = eDescr->p2pApi.func;
      logStartEvent(event, eDescr->parentObj, "func=%s, count=%zu, dtype=%s, stream=%p, graphCaptured=%d",
                    eDescr->p2pApi.func, eDescr->p2pApi.count, eDescr->p2pApi.datatype,
                    eDescr->p2pApi.stream, eDescr->p2pApi.graphCaptured ? 1 : 0);
      break;
      
    case ncclProfileKernelLaunch:
      event->func = "KernelLaunch";
      logStartEvent(event, eDescr->parentObj, "stream=%p", eDescr->kernelLaunch.stream);
      break;
    
    case ncclProfileColl:
      event->func = eDescr->coll.func;
      logStartEvent(event, eDescr->parentObj, "func=%s, seq=%lu, algo=%s, proto=%s, channels=%d, warps=%d, count=%zu, root=%d, dtype=%s, sendBuff=%p, recvBuff=%p, parentGroup=%p",
                    eDescr->coll.func, eDescr->coll.seqNumber, eDescr->coll.algo, eDescr->coll.proto,
                    eDescr->coll.nChannels, eDescr->coll.nWarps, eDescr->coll.count, eDescr->coll.root,
                    eDescr->coll.datatype, eDescr->coll.sendBuff, eDescr->coll.recvBuff, eDescr->coll.parentGroup);
      break;
      
    case ncclProfileP2p:
      event->func = eDescr->p2p.func;
      logStartEvent(event, eDescr->parentObj, "func=%s, peer=%d, channels=%d, count=%zu, dtype=%s, buff=%p, parentGroup=%p",
                    eDescr->p2p.func, eDescr->p2p.peer, eDescr->p2p.nChannels, eDescr->p2p.count,
                    eDescr->p2p.datatype, eDescr->p2p.buff, eDescr->p2p.parentGroup);
      break;
      
    case ncclProfileProxyOp:
      event->func = "ProxyOp";
      logStartEvent(event, eDescr->parentObj, "channel=%d, peer=%d, steps=%d, chunkSize=%d, isSend=%d, eventPid=%d",
                    eDescr->proxyOp.channelId, eDescr->proxyOp.peer, eDescr->proxyOp.nSteps,
                    eDescr->proxyOp.chunkSize, eDescr->proxyOp.isSend, static_cast<int>(eDescr->proxyOp.pid));
      break;
      
    case ncclProfileProxyStep:
      event->func = "ProxyStep";
      logStartEvent(event, eDescr->parentObj, "step=%d", eDescr->proxyStep.step);
      break;
      
    case ncclProfileProxyCtrl:
      event->func = "ProxyCtrl";
      logStartEvent(event, eDescr->parentObj, "");
      break;
      
    case ncclProfileKernelCh:
      event->func = "KernelCh";
      logStartEvent(event, eDescr->parentObj, "channel=%d, pTimer=%lu", eDescr->kernelCh.channelId, eDescr->kernelCh.pTimer);
      break;
      
    case ncclProfileNetPlugin:
      event->func = "NetPlugin";
      logStartEvent(event, eDescr->parentObj, "id=%ld, data=%p", static_cast<long>(eDescr->netPlugin.id), eDescr->netPlugin.data);
      break;
      
    case ncclProfileGroup:
      event->func = "Group";
      logStartEvent(event, eDescr->parentObj, "");
      break;
      
    default:
      event->func = "Unknown";
      event->typeName = "ncclProfileUnknown";
      logStartEvent(event, eDescr->parentObj, "type=0x%lx", static_cast<unsigned long>(eDescr->type));
      break;
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
  pid_t pid = getpid();
  
  // Handle detached PXN ProxyOp events from the detach pool
  // Note: ctx is nullptr for PXN events - use process-wide detachLogFile
  if (event->fromDetachPool) {
    // Log to process-wide PXN log file
    if (detachLogFile) {
      pthread_mutex_lock(&detachLogMutex);
      fprintf(detachLogFile, "[%.3f] STOP %s::%s (pid=%d, tid=%d, PXN event origin pid=%d, duration=%.3f us, parentObj=%p, event=%p)\n",
              endTime - detachStartTime,
              event->typeName ? event->typeName : "ncclProfileUnknown",
              event->func ? event->func : "Unknown",
              static_cast<int>(pid),
              tid,
              static_cast<int>(event->originPid),
              duration,
              event->parentObj,
              static_cast<void*>(event));
      fflush(detachLogFile);
      pthread_mutex_unlock(&detachLogMutex);
    }
    
    // Write complete JSONL event (PXN)
    writeJsonlCompleteEvent(event, endTime, duration, tid);
    
    // Release parent reference and free to detach pool
    releaseDetachParent(event->parentObj);
    freeToDetachPool(event);
    
    return ncclSuccess;
  }
  
  // Normal event (from per-context pool) - ctx must be valid
  if (!ctx) {
    return ncclSuccess;  // Drop event if ctx is somehow NULL
  }
  
  pthread_mutex_lock(&ctx->logMutex);
  fprintf(ctx->logFile, "[%.3f] STOP %s::%s (pid=%d, tid=%d, gpu_uuid=%s, duration=%.3f us, event=%p, ctx=%p)\n",
          endTime,
          event->typeName ? event->typeName : "ncclProfileUnknown",
          event->func ? event->func : "Unknown",
          static_cast<int>(pid),
          tid,
          ctx->gpuUuid[0] != '\0' ? ctx->gpuUuid : "unknown",
          duration,
          static_cast<void*>(event),
          static_cast<void*>(ctx));
  pthread_mutex_unlock(&ctx->logMutex);
  
  // Write complete JSONL event (normal)
  writeJsonlCompleteEvent(event, endTime, duration, tid);
  
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
  
  // Handle PXN events (ctx is nullptr) - use process-wide detach log
  if (event->fromDetachPool) {
    if (detachLogFile) {
      pthread_mutex_lock(&detachLogMutex);
      fprintf(detachLogFile, "[%.3f] STATE %s::%s -> %s (pid=%d, tid=%d, PXN from pid=%d, stateId=%d",
              currentTime - detachStartTime,
              event->typeName ? event->typeName : "ncclProfileUnknown",
              event->func ? event->func : "Unknown",
              stateName,
              static_cast<int>(pid),
              tid,
              static_cast<int>(event->originPid),
              static_cast<int>(eState));
      
      // Log additional state args
      if (eStateArgs) {
        switch (eState) {
          case ncclProfilerProxyStepSendWait:
          case ncclProfilerProxyStepRecvFlushWait:
            fprintf(detachLogFile, ", transSize=%zu", eStateArgs->proxyStep.transSize);
            break;
          case ncclProfilerProxyOpInProgress_v4:
            // ProxyOp progress state for PXN events
            break;
          default:
            break;
        }
      }
      
      fprintf(detachLogFile, ", event=%p)\n", static_cast<void*>(event));
      fflush(detachLogFile);
      pthread_mutex_unlock(&detachLogMutex);
    }
    
    // Accumulate state in event buffer (for JSONL output on STOP)
    // Using dynamic vector - no limit on number of states
    {
      StateTransition st;
      st.timestamp = currentTime;
      st.stateName = stateName ? stateName : "Unknown";
      st.stateId = static_cast<int>(eState);
      st.tid = tid;
      st.pid = pid;
      
      if (eStateArgs) {
        switch (eState) {
          case ncclProfilerProxyStepSendWait:
          case ncclProfilerProxyStepRecvFlushWait:
            st.details = "\"transSize\":" + std::to_string(eStateArgs->proxyStep.transSize);
            break;
          default:
            break;
        }
      }
      event->states.push_back(std::move(st));
    }
    
    return ncclSuccess;
  }
  
  // Normal event - ctx must be valid
  if (!ctx) {
    return ncclSuccess;  // Drop event if ctx is NULL
  }
  
  pthread_mutex_lock(&ctx->logMutex);
  fprintf(ctx->logFile, "[%.3f] STATE %s::%s -> %s (pid=%d, tid=%d, gpu_uuid=%s, stateId=%d",
          currentTime,
          event->typeName ? event->typeName : "ncclProfileUnknown",
          event->func ? event->func : "Unknown",
          stateName,
          static_cast<int>(pid),
          tid,
          ctx->gpuUuid[0] != '\0' ? ctx->gpuUuid : "unknown",
          static_cast<int>(eState));
  
  // Log additional state args based on state type
  if (eStateArgs) {
    switch (eState) {
      case ncclProfilerProxyStepSendWait:
      case ncclProfilerProxyStepRecvFlushWait:
        fprintf(ctx->logFile, ", transSize=%zu", eStateArgs->proxyStep.transSize);
        break;
      case ncclProfilerProxyCtrlAppendEnd:
        fprintf(ctx->logFile, ", appendedProxyOps=%d", eStateArgs->proxyCtrl.appendedProxyOps);
        break;
      case ncclProfilerKernelChStop:
        fprintf(ctx->logFile, ", pTimer=%lu", eStateArgs->kernelCh.pTimer);
        break;
      case ncclProfilerNetPluginUpdate:
        fprintf(ctx->logFile, ", data=%p", eStateArgs->netPlugin.data);
        break;
      default:
        break;
    }
  }
  
  fprintf(ctx->logFile, ", event=%p, ctx=%p)\n", static_cast<void*>(event), static_cast<void*>(ctx));
  pthread_mutex_unlock(&ctx->logMutex);
  
  // Accumulate state in event buffer (for JSONL output on STOP)
  // Using dynamic vector - no limit on number of states
  {
    StateTransition st;
    st.timestamp = currentTime;
    st.stateName = stateName ? stateName : "Unknown";
    st.stateId = static_cast<int>(eState);
    st.tid = tid;
    st.pid = pid;
    
    if (eStateArgs) {
      switch (eState) {
        case ncclProfilerProxyStepSendWait:
        case ncclProfilerProxyStepRecvFlushWait:
          st.details = "\"transSize\":" + std::to_string(eStateArgs->proxyStep.transSize);
          break;
        case ncclProfilerProxyCtrlAppendEnd:
          st.details = "\"appendedProxyOps\":" + std::to_string(eStateArgs->proxyCtrl.appendedProxyOps);
          break;
        case ncclProfilerKernelChStop:
          st.details = "\"pTimer\":" + std::to_string(eStateArgs->kernelCh.pTimer);
          break;
        case ncclProfilerNetPluginUpdate:
          st.details = "\"data\":\"" + formatPointerStd(eStateArgs->netPlugin.data) + "\"";
          break;
        default:
          break;
      }
    }
    event->states.push_back(std::move(st));
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
  
  if (ctx->logFile) {
    pthread_mutex_lock(&ctx->logMutex);
    fprintf(ctx->logFile, "[%.3f] === Profiler finalized (tid=%d, totalDuration=%.3f us = %.6f s, ctx=%p) ===\n", 
            endTime, tid, totalDuration, totalDuration / 1e6, static_cast<void*>(ctx));
    fprintf(ctx->logFile, "Final pool size: %zu chunks, %zu total slots\n", 
            ctx->pool.chunks.size(), ctx->pool.chunks.size() * ctx->pool.chunkSize);
    fprintf(ctx->logFile, "Final blacklist size: %zu\n", ctx->blacklist.map.size());
    fflush(ctx->logFile);
    fclose(ctx->logFile);
    pthread_mutex_unlock(&ctx->logMutex);
  }
  
  // Destroy mutexes
  pthread_mutex_destroy(&ctx->logMutex);
  pthread_mutex_destroy(&ctx->allocMutex);
  
  fprintf(stderr, "[MinimalProfiler] Finalized: tid=%d, rank=%d, commId=%lu, final_pool_chunks=%zu, final_pool_slots=%zu, final_blacklist_size=%zu\n", 
          tid, ctx->rank, ctx->commId, ctx->pool.chunks.size(), 
          ctx->pool.chunks.size() * ctx->pool.chunkSize, ctx->blacklist.map.size());
  
  delete ctx;
  
  // Clean up detach pool when last communicator is finalized
  pthread_mutex_lock(&processInitMutex);
  processInitCount--;
  if (processInitCount == 0) {
    cleanupDetachPool();
    fprintf(stderr, "[MinimalProfiler] Cleanup: detach pool and blacklist freed\n");
  }
  pthread_mutex_unlock(&processInitMutex);
  
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
