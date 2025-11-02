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

