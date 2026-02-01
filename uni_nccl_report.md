# NCCL Profiler Plugin API - eine Machbarkeitsstudie

TODO
- Table of Contents
  - 0. Abstract - gpu communication profiling/tracing motivieren
  - 0. Introduction - vorausgesetztes Verständis soweit nötig für bestimmte Konzepte erwähnen oder erläutern (zb MPI, SLURM, NCCL)
  - 1. Die Profiler API
    - 1.1 Einbindung des Plugins in NCCL.
      + first draft 
      - TODO final draft
    - 1.2 "rohe" API definition, kurz und knapp ohne viel erläuterung.
      + first draft 
      - TODO final draft
    - 1.3 Der Codeflow: application NCCL User API calls -> profiler API calls
      - TODO first draft 
      - TODO final draft
    - 1.4 Der Codeflow++ genauere betrachtung in bestimmten situationen für verständis:
      - ncclGroup, multi gpu streams, 
      - multi threaded application
      - in multi-node environment ansicht mit mehreren prozessen und profiler instances 
  - 2. Was einem die Profiler Plugin API (nicht) ermöglicht
    - Beispielhaft logging, running metrics, cupti, ...
  - 3. Warum (nicht) die Profiler Plugin API in Erwähnung ziehen? 
    - Experimente/Benchmarking auswerten
    - Genauigkeit & consistency der timings der api calls diskutieren?
    - Besondere Vor- & Nachteile hervorheben
  - 4. Conclusion - Nützlichkeit für P-Score Messsystem

- main content chunks/concepts to use through out the TOC to keep it all simple and chill:
  - use simple code example walkthrough to illustrate what happens when 
  - illustrate in swim lane diagrams
    - the behavior of user API call -> nccl profiler init/finalize
    - the behavior of user API call -> nccl profiler start/stop/recordEventState
  - benchmarking, measurements
  - conclusion

## Abstract
- AI - big use case for HPC
- Expensive workloads! Desire to understand and optimize application performance
- big part of AI workloads is GPU communication 
- as they often span across many GPUs
- NCCL - the library that implements communication routines for NVIDIA GPUs
- provides an interface to plugin a custom profiler into NCCL to extract performance data

## Introduction
TODO Mention
- MPI concepts
- NCCL concepts
- SLURM concepts
as needed

NCCL Concepts

Before taking a closer look at the Profiler Plugin API it is helpful to first understand what NCCL does internally when an application calls the NCCL User API.

a typical nccl application program follows this basic code structure

```c
// create nccl communicators
createNcclComm();

// allocate memory for computation and communication
prepareDeviceForWork(); 

// do computation and commmunication
callNcclCollectives();
// ...

// finalize and clean up nccl communicators
cleanupNccl();
```

During nccl communicator creation, NCCL will internaly spawn a thread called ProxyService. This thread will lazily start another thread called ProxyProgress, which will handle network requests for GPU communication during collective and p2p operations. 


```plaintext
Thread Creation Flow
─────────────────────────────────────────────────────────────────────────────────
User API                          Internal Flow
─────────────────────────────────────────────────────────────────────────────────
ncclCommInitRank()         ─┐
ncclCommInitAll()           │
ncclCommInitRankConfig()    ├───► ncclCommInitRankDev()  ──┐
ncclCommInitRankScalable() ─┘                              │
ncclCommSplit()            ─┐                              │
ncclCommShrink()           ─┴───► ncclCommInitChildComm() ─┤
ncclCommGrow()             ─┐     ┌────────────────────────┘
                            │     ▼
                            └───► ncclCommInitRankFunc()
                                  │
                                  ▼                        
                                  initTransportsRank()
                                  │
                                  └─► ncclProxyCreate(comm)
                                      │
                                      ▼
                                      if (proxyState->refCount == 1)
                                          pthread_create(&comm->proxyState->thread, NULL, ncclProxyService, ...)
                                          │
                                          ▼
                                          [New Thread] ncclProxyService() 
                                          │
                                          ├──► proxyProgressAsync(...)
                                          │    │
                                          │    └──► proxyConnInit(...)
                                          │         │
                                          │         └──► proxyProgressInit(proxyState)
                                          │              │
                                          │              └──► ncclProxyProgressCreate(proxyState)
                                          │                   │
                                          │                   └──► if (!state->thread)
                                          │                          pthread_create(&state->thread, NULL, ncclProxyProgress, proxyState)
                                          │                          │
                                          │                          ▼
                                          │                          [New Thread] ncclProxyProgress() 
                                          │                             
                                          └──► proxyServiceInitOp(...)
                                               └──► proxyProgressAsync(...)  (same path as above)
```

The if guards `if (proxyState->refCount == 1)` and `if (!state->thread)` ensure that these threads are created once per a shared resource (struct `ncclSharedResources`). Below is a snippet of the relevant structs

**/src/include/comm.h**
```c
struct ncclSharedResources {
  struct ncclComm* owner; /* communicator which creates this shared res. */
  struct ncclProxyState* proxyState;
  // other fields
}
```

**/src/include/proxy.h**
```c
struct ncclProxyState {
  int refCount;
  pthread_t thread;
  // other fields
}
```

By default every nccl communicator has their own shared resource.
In case the application calls `ncclCommSplit()` or `ncclCommShrink()`, where the original communicator was initialized using a `ncclConfig_t` struct with fields `splitShare` or `shrinkShare` set to 1, the newly created nccl communicator will instead share the shared resource (and the proxy threads) with the originating parent communicator. 

"

```c
  /* proxyState is shared among parent comm and split comms. comm->proxyState->thread is
   * pthread_join()'d by commFree() in init.cc when the refCount reduces down to 0. */
```

" (quote from **/src/proxy.cc**) 


Later, whenever the application calls the NCCL User API, NCCL internally will decide on what network operations to do and posts them to a pool. The ProxyProgress thread will read these operations from the pool and progress them.

The following paths reach `ncclProxyPost()`, where Ops will be posted to a pool and the proxy progress thread will be signalled.

```plaintext
Flow from User API Calls to ncclProxyPost()
─────────────────────────────────────────────────────────────────────────────────
---------------------------------------------------------------------------------
User API
---------------------------------------------------------------------------------
ncclCommInitAll()          ─┐    ncclAllGather()      ─┐      
ncclCommInitRankConfig()    │    ncclAlltoAll()        │      
ncclCommInitRankScalable()  │    ncclAllReduce()       │      
ncclCommFinalize()          │    ncclBroadcast()       │      
ncclCommDestroy()           │    ncclGather()          │      
ncclCommRevoke()            │    ncclReduce()          │      
ncclCommAbort()             │    ncclReduceScatter()   │      
ncclCommSplit()             │    ncclScatter()         │      
ncclCommShrink()            │    ncclSend()            │      
ncclCommGrow()              │    ncclRecv()           ─┤      
ncclDevCommCreate()         │                          │      
ncclCommWindowRegister()    │                          │
ncclGroupSimulateEnd()     ─┤                          │
                            │                          │ 
---------------------------------------------------------------------------------
Internal Flow
---------------------------------------------------------------------------------
                            │                          │ 
                            │                          ▼      
                            │                   ncclEnqueueCheck()
                            ├──────────────────────────┘
                            ▼
                            ncclGroupEndInternal()
                            ├────┐
                            │    ▼
                            │    groupLaunchNonBlocking()
                            ├────┘
                            ▼
                            groupLaunch()
                            │
                            ▼
                            doLaunches()
                            ├──► ncclLaunchPrepare() ────────────┐
                            └──► ncclLaunchKernelAfter_NoCuda()  ▼
                                 │                               ncclLaunchPrepare()
                                 │                               │
                                 │                               ▼
                                 │                               cudaLaunchHostFunc()
                                 │                               │
                                 │                               ▼
                                 │                               hostStreamPlanCallback()
                                 ├───────────────────────────────┘
                                 ▼
                                 hostStreamPlanTask()
                                 ├──► uploadProxyOps() ─┐
                                 └──► ncclProxyStart()  ▼
                                      │                 ncclProxySaveOp()
                                      │                 │
                                      │                 ▼
                                      │                 SaveProxy()
                                      │                 │
                                      │                 ▼
                                      │                 ncclLocalOpAppend()
                                      ├─────────────────┘
                                      ▼
                                      ncclProxyPost() (proxy.cc)
                                      ├──► [Posts Ops to pool]
                                      └──► [Signals Proxy Progress Thread]
```

The pproxy rogress thread reads from this pool when calling `ncclProxyGetPostedOps()` and progresses them.

**/src/proxy.cc**
```plaintext
ncclProxyProgress() progressing loop
─────────────────────────────────────────────────────────────────────────────────
ncclProxyProgress(proxyState)
└──► do {
          ├──►progressOps(proxyState, ...)
          │   └──► while (op) {
          │             op->progress(proxyState, op);
          │             op = op->next;
          │        }
          │   
          └──►ncclProxyGetPostedOps()
              └──► [reads Ops or thread will wait]
     } while (...)
```

Understanding this behaviour is useful when taking look at the profiler plugin API for network related activity, explained in the next section.

## The Profiler API
### 1.1 How nccl detects the profiler plugin
When a nccl communicator is created, NCCL looks for a shared library which represents the profiler plugin, checking for the environment variable NCCL_PROFILER_PLUGIN: `profilerName = ncclGetEnv("NCCL_PROFILER_PLUGIN")`. 

It then calls `handle* = dlopen(name, RTLD_NOW | RTLD_LOCAL)` 
and `ncclProfiler_v5 = (ncclProfiler_v5_t*)dlsym(handle, "ncclProfiler_v5")` to load the library immediately with local symbol visibility.

"

- If NCCL_PROFILER_PLUGIN is set, attempt loading the library with name specified by NCCL_PROFILER_PLUGIN;
- If NCCL_PROFILER_PLUGIN is set and previous failed, attempt loading libnccl-profiler-<NCCL_PROFILER_PLUGIN>.so;
- If NCCL_PROFILER_PLUGIN is not set, attempt loading libnccl-profiler.so;
If no plugin was found (neither user defined nor default), do not enable profiling.
- If NCCL_PROFILER_PLUGIN is set to `STATIC_PLUGIN`, the plugin symbols are searched in the program binary.

" (quoted from docs: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-profiler-plugin)

The following depicts the flow from the user API call to loading the profiler plugin:

```plaintext
User API                          Internal Flow
─────────────────────────────────────────────────────────────────────────────────
ncclCommInitRank()         ─┐
ncclCommInitAll()           │
ncclCommInitRankConfig()    ├───► ncclCommInitRankDev()  ──┐
ncclCommInitRankScalable() ─┘                              │
ncclCommSplit()            ─┐                              │
ncclCommShrink()           ─┴───► ncclCommInitChildComm() ─┤
ncclCommGrow()             ─┐     ┌────────────────────────┘
                            │     ▼
                            └───► ncclCommInitRankFunc()
                                  │
                                  ▼                        
                                  initTransportsRank()
                                  ├──► ncclProfilerPluginInit(comm)─┐
                                  └──► ncclProxyCreate(comm)        │
                                       │                            ▼
                                       ▼                            ncclProxyCreate(comm)
                                       ...                          │
                                                                    ▼
                                                                    ncclProfilerPluginInit(comm)
                                                                    ├──► ncclProfilerPluginLoad()
                                                                    │    ├──► ncclGetEnv("NCCL_PROFILER_PLUGIN")
                                                                    │    ├──► ncclOpenProfilerPluginLib()
                                                                    │    │    └──► dlopen(name, RTLD_NOW | RTLD_LOCAL)
                                                                    │    ├──► getNcclProfiler_v5()
                                                                    │    │    └──► (ncclProfiler_v5_t*)dlsym(lib, "ncclProfiler_v5");
                                                                    │    └──► [try fallback to older api version]
                                                                    └──► profiler->init()
```

As shown, the profiler is loaded when creating a communicator (this happens before the proxy thread creation).
The plugin loading mechanism expects the struct variable name to follow the naming convention `ncclProfiler_v{versionNum}`, which also indicates which API version is implemented.

The profiler API has changed multiple times with newer NCCL releases. However, new releases seem to have limited backwards compatibility with older plugins (TODO requires fact check. iirc there was some verison breaking feature that didnt allow backwards compatibilty to v2? but current nccl relrease features the structs for older versions @/src/include/plugin/profiler and has a fallback mechanism to older api versions @/src/plugin/profiler.cc
i think the problem is that with every new api version release, older profiler definitions under @/src/plugin/profiler/profiler_v{x}.cc files must be updated to handle/void/hide the new api version features. But looking at the commit history this doesnt seem to happen consistently. if they are not hidden or made backwards compatible, a plugin implemented against an old api version may be handed features from the new api version, which it does not how to handle (when faithfully implementing the old api)).

The exact implementation details for the loading mechanism can be found at **/src/plugin/plugin_open.cc** and **/src/plugin/profiler.cc**

### 1.2 The profiler API definition
The plugin must implement a profiler API specified by NCCL, in the form of exposing a struct. This struct should contain pointers to all the functions required by the API.
Given the plugin loading mechanism, a plugin may expose multiple versioned structs, allowing for backwards compatibility with older NCCL releases.

```c
ncclProfiler_v5_t ncclProfiler_v5 = {
  const char* name;
  ncclResult_t (*init)(...);              // NCCL calls this right after loading the plugin
  ncclResult_t (*startEvent)(...);        // called at the start of several kinds of operations or activities
  ncclResult_t (*stopEvent)(...);         // called at the end of these operations or activites
  ncclResult_t (*recordEventState)(...);  // called to record the state of certain operations or activities
  ncclResult_t (*finalize)(...);          // called before unloading the plugin
};
```

The full profiler API can be found under **/src/include/plugin/profiler/profiler_v{versionNum}.cc**. As of NCCL release v2.29.1, version 6 is the latest API version. Five functions must be implemented. 
Internally NCCL wraps calls to the profiler API in custom functions. Inside the file **/src/include/profiler.h** they are forward declared and also nicely organized.

The NCCL code internally implements callbacks to the profiler API at different levels to capture start/stop of groups, collective and point-to-point operations, as well as proxy, kernel and network activity.
As the API function names suggest, this will allow the profiler to track these operations and activities as events.

Below the API functions will be explained and where NCCL triggers callbacks to them:

#### init
`init` initializes the profiler plugin. NCCL passes various arguments: 
```c
ncclResult_t init(
  void** context,          // out param - opaque profiler context object for separating profiler behavior across communicators
  uint64_t commId,         // communicator id
  int* eActivationMask,    // out param - bitmask that sets which events are tracked the plugin
  const char* commName,    // user assigned communicator name
  int nNodes,              // number of nodes in communicator
  int nranks,              // number of ranks in communicator
  int rank,                // rank identifier in communicator
  ncclDebugLogger_t logfn  // logger function
);
```
the `init()` function is called immediately upon successful plugin load in `ncclProfilerPluginLoad()`, which is also depicted in the diagram above. It is worth mentioning that this is called everytime a communicator is created (unlike the proxy threads).

If the init function does not return `ncclSuccess`, NCCL disables the plugin.

"

As soon as NCCL finds the plugin and the correct ncclProfiler symbol, it calls its init function. This allows the plugin to initialize its internal context, used during profiling of NCCL events. 

" (quote from **/ext-profiler/README.md**)

Notably, `void** context` is an opaque handle that the plugin developer may point to any custom context object. This pointer is then again passed as argument in the other API calls `startEvent` and `finalize`. This context object is separate across communicators.

the plugin developer should point `int* eActivationMask` to a bitmask, which tells NCCL which type of events that the profiler plugin wants to track. Each bit corresponds to a different event type. The mappings can be found at **/src/include/plugin/nccl_profiler.h**. internally this bitmask is initialized with `0`, which means no events will be tracked by default. Setting it to `4095` will track all events.

TODO what to do with `ncclDebugLogger_t logfn` ?

#### startEvent
`startEvent` is called when NCCL begins certain operations. NCCL passes various arguments:
```c
ncclResult_t startEvent(
  void* context,                       // opaque profiler context object
  void** eHandle,                      // out param - event handle for event
  ncclProfilerEventDescr_v5_t* eDescr  // pointer to ncclProfilerEventDescr_t object
);
```
As of release v2.29.1 NCCL does not care about the return value of this function.

The plugin developer may point `void** eHandle` to a custom event object. this pointer is then again passed as argument in the other API calls `stopEvent` and `recordEventState` to refer to the same event object. `void* context` passes the context object for this communicator in the profiler

`ncclProfilerEventDescr_v5_t* eDescr` is a struct that contains all the information that nccl populates to describe the started event. Exact details can be found under the profiler API **/src/include/plugin/profiler/**. Notably, the event descriptor struct contains a field `void* parentObj`, which is the `eHandle` to a parent event of this event (`null` if it has no parent). The meaning of parent events can be explained as following:

All User API calls to Collective or P2P operations will start a Group API event, and when networking is required (in multi-node environments) ProxyCtrl Events may be emitted. Depending on the `eActivationMask` bitmask returned in the `init()` function, further (child) events will be emitted in deeper sections of the nccl code base. It can be thought of as an event hierarchy with several depth levels:

"
```
Group API event
   |
   +- Collective API event
   |  |
   |  +- Collective event
   |     |
   |     +- ProxyOp event
   |     |  |
   |     |  +- ProxyStep event
   |     |     |
   |     |     +- NetPlugin event
   |     |
   |     +- KernelCh event
   |
   +- Point-to-point API event
   |  |
   |  +- Point-to-point event
   |     |
   |     +- ProxyOp event
   |     |  |
   |     |  +- ProxyStep event
   |     |     |
   |     |     +- NetPlugin event
   |     |
   |     +- KernelCh event
   |
   +- Kernel Launch event

ProxyCtrl event
```
" (quote from **/ext-profiler/README.md**)

If the profiler enables tracking for event types lower in the hierarchy, NCCL will also track their parent event types. This also allows for meaningful `eHandle`s as values for the `parentObj` field inside the `eDescr`.

The below diagrams shows where NCCL emits `startEvent`s and `stopEvent`s. 

```plaintext
Flow from NCCL API Calls to profiler events
─────────────────────────────────────────────────────────────────────────────────
---------------------------------------------------------------------------------
User API
---------------------------------------------------------------------------------
ncclCommInitAll()          ─┐    ncclAllGather()      ─┐      
ncclCommInitRankConfig()    │    ncclAlltoAll()        │      
ncclCommInitRankScalable()  │    ncclAllReduce()       │      
ncclCommFinalize()          │    ncclBroadcast()       │      
ncclCommDestroy()           │    ncclGather()          │      
ncclCommRevoke()            │    ncclReduce()          │      
ncclCommAbort()             │    ncclReduceScatter()   │      
ncclCommSplit()             │    ncclScatter()         │      
ncclCommShrink()            │    ncclSend()            │      
ncclCommGrow()              │    ncclRecv()           ─┤      
ncclDevCommCreate()         │                          │      
ncclCommWindowRegister()    │                          │
ncclGroupSimulateEnd()     ─┤                          │
                            │                          │ 
---------------------------------------------------------------------------------
Internal Flow
---------------------------------------------------------------------------------
                            │                          │ 
                            │                          ▼      
                            │                   ncclEnqueueCheck()
                            │                          ├──► taskAppend()
                            ├──────────────────────────┘    ├──► collTaskAppend()
                            │                               │    └──► Emits: GroupApi Event (start), CollApi Event
                            │                               └──► p2pTaskAppend()
                            ▼                                    └──► Emits: GroupApi Event (start), P2pApi Event
                            ncclGroupEndInternal()
                            ├──► Emits: GroupApi Event (stop, at end of function call if eHandle exists)
                            ├───────────┐
                            │           ▼
                            │           groupLaunchNonBlocking()
                            ├───────────┘
                            ▼
                            groupLaunch()
                            │
                            ▼
                            doLaunches()
                            ├──► ncclLaunchPrepare() ────────────┐
                            ├──► ncclLaunchKernel()              │
                            │    └──► Emits: KernelLaunch Event  │
                            └──► ncclLaunchKernelAfter_NoCuda()  ▼
                                 │                               ncclLaunchPrepare()
                                 │                               │
                                 │                               ▼
                                 │                               cudaLaunchHostFunc()
                                 │                               │
                                 │                               ▼
                                 │                               hostStreamPlanCallback()
                                 ├───────────────────────────────┘
                                 ▼
                                 hostStreamPlanTask()
                                 ├──► Emits: Group Event, Coll Event, P2p Event
                                 ├──► uploadProxyOps() ─┐
                                 └──► ncclProxyStart()  ▼
                                      │                 ...
                                      ▼
                                      ...
```
The implementation can be found at **/src/init.cc** and **/src/plugin/profiler.cc**.

The ProxyProgress Thread is also emitting `startEvent`s and `stopEvent`s while progressing Ops:

```plaintext
ncclProxyProgress() Event Emission Flow
─────────────────────────────────────────────────────────────────────────────────
ncclProxyProgress(proxyState) [Proxy Progress Thread Loop]
└──► do {
          ├──► progressOps(proxyState, opStart=state->active, ...)
          │    │
          │    └──► while (op) {
          │              op->progress(proxyState, op);
          │              └──► [Transport-specific progress function]
          │                   (net.cc, coll_net.cc, p2p.cc, shm.cc)
          │                   ├──► Emits: ProxyStep events
          │                   ├──► Emits: KernelCh events
          │                   └──► Emits: Network plugin specific events
          │              op = op->next;
          │         }
          │
          ├──► [Thread Idle/Active State Transitions]
          │    └──► Emits: ProxyCtrl events
          │
          └──► ncclProxyGetPostedOps(proxyState)
               ├──► [Thread Sleep/Wakeup State Transitions]
               │      ├──► [Thread sleeps, waits for signal]
               │      └──► Emits: ProxyCtrl events
               ├──► [Update Op pool] 
               ├──► Emits: ProxyCtrl events
               └──► ProxyAppend()
                    └──► ncclProxyOpToArgs()
                         └──► Emits: ProxyOp events
     } while (...)
```

`op->progress()` progresses transport specific ops. this is just a function pointer type defined at **/src/include/proxy.h** (confusingly the variable is called `op`, although its type is `ncclProxyArgs` and NOT `ncclProxyOp`).

```c
typedef ncclResult_t (*proxyProgressFunc_t)(struct ncclProxyState*, struct ncclProxyArgs*);

struct ncclProxyArgs {
  proxyProgressFunc_t progress;
  struct ncclProxyArgs* next;
  /* other fields */
}
```

Which specific function this calls depends on the Op. This also decides on which profiler events are being started/stopped (ProxyStep Event, KernelCh Event or Network plugin specific events). 
Check the different ProxyProgress functions under **/src/transport/net.cc**, **/src/transport/coll_net.cc**, **/src/transport/p2p.cc**, **/src/transport/shm.cc** and **/src/transport/shm.cc**.


Unless the user sets the environment variable `NCCL_PXN_DISABLE` to 0 (defaults to 1) to disable the PXN feature (PCI x NVLink - inter-node communication using a non-local NIC, using NVLink and an intermediate GPU), this causes some proxy operations to be processed in a proxy thread of a remote process, different to the one that originally generated the operation. When this happens, dereferencing the `parentObj` event in the `eDescr` is not possible and the event hierarchy reported above may not be utilized. The event descriptor for `ProxyOp` events includes the PID of the originator. The profiler plugin can match this against the local PID (which is the PID of the processing proxy thread calling this startEvent) to safely dereference the `parentObj` event. 

Unfortunately, the ProxyOp child event `ProxyStep` does not provide this field. However when NCCL calls `startEvent()`, it also passes the belonging `context` object of the event in a similar manner, making dereferencing unsafe because of the PXN feature. The profiler plugin developer may decide to internally track which context objects have been created on this process during (potentially multiple thread-level) `init()` calls. If the passed `context` object during `startEvent()` is not present in the profilers tracked contexts, then this also indicates that the event did not originate from this process because of the PXN feature. 

#### stopEvent
`stopEvent` tells the plugin that the event has stopped.
```c
ncclResult_t stopEvent(void* eHandle);  // handle to event object
```
As of release v2.29.1 NCCL does not care about the return value of this function.

`stopEvent` is called in the same functions that call `startEvent`, except for the GroupApi Event (see above diagram).

#### recordEventState
Some event types may be updated by NCCL through calls to `recordEventState`, which can update the state, along with event attributes. These type of events can go through several states during their lifecycle. The list of supported states for the updatable events can be found under the profiler API **/src/include/plugin/profiler/profiler_v{versionNum}.h**
```c
ncclResult_t recordEventState(
  void* eHandle,                               // handle to event object created through startEvent
  ncclProfilerEventState_v5_t eState,          // event state transition
  ncclProfilerEventStateArgs_v5_t* eStateArgs  // optional argument used to capture event attribute updates associated with the state transition
);
```
As of release v2.29.1 NCCL does not care about the return value of this function.

`recordEventState` is called in the same functions that call `startEvent`.

#### finalize
After a user API call is made to free resources associated with the communicator object the profiler API's `finalize()` function is called in `ncclProfilerPluginFinalize()`. Afterwards the profiler is also unloaded by eventually calling `dlclose(handle)` inside `ncclProfilerPluginUnload()`.
```c
ncclResult_t finalize(void* context);  // opaque profiler context object
```


```plaintext
User API                          Internal Flow
─────────────────────────────────────────────────────────────────────
ncclCommAbort()    ─┐
ncclCommDestroy()  ─┴───────────► commReclaim()
                                  │
                                  ▼
                                  ncclProfilerPluginFinalize()
                                  ├──► ncclProfiler->finalize()
                                  └──► ncclProfilerPluginUnload()
```
See implementation details at **/src/init.cc**, **/src/plugin/profiler.cc** and **/src/plugin/plugin_open.cc**.


#### name
Besides these functions, the profiler plugin struct also contains a `name` field.

"

The name field should point to a character string with the name of the profiler plugin. This will be used for all logging, especially when NCCL_DEBUG=INFO is set.

" (quote from **/ext-profiler/README.md**)

#### TODO copy engine based events?

### Callback perspective from viewpoint of multi-node cluster
Besides application side NCCL API calls leading to profiler API calls, NCCL itself gets initialized multiple times when communicators are created. The setting of a multi-node environment influences the total number of callbacks to the profiler. To better understand the profiling behaviour, the different settings below will be explained

TODO the next two subpoints below is one suggestion on how to think about different scenarios (number of nccl launches, profiler launches, per node, per thread...).
not sure how much sense they make or if some other way of differentiation is better. work through the individual questions/points/todos within those subpoints and it will become clear what makes most sense!
#### callback behaviour under different communicator initialization strategies
TODO does `one_device_per_process_mpi` and `one_device_per_thread` behave the same? does any of the these init strats matter to behaviour of profiler?
TODO i think from nccl persepective it doesnt care - it just uses ncclUniqueId down to a thread-level granularity to identify communicator groups?
##### strategy `one_device_per_process_mpi`
##### strategy `one_device_per_thread` 
##### strategy using custom network socket code instead of MPI
TODO

#### callback behaviour when setting specific node configurations
n nodes, m gpus/per node and p tasks on m gpus allow many theoretical cases for node configurations by combination:
- 1 node, n nodes
- 1 gpu/node, m gpus/node
- p tasks/node, where
  - tasks/node < gpus/node, 
  - tasks/node = gpus/node, 
  - tasks/node > gpus/node
  
TODO make this a table + add excalidraw pictograms?

##### tasks/node < gpus/node
###### tasks/node < gpus/node, 1 gpus/node, 1 node
This case would imply 0 tasks so it is ignored.
###### tasks/node < gpus/node, m gpus/node, 1 node
This can come in 2 flavours:
####### 1 task/node
this will behave like `one_device_per_thread`
####### n tasks/node
TODO check slurm 2540242
TODO if this works, might be be similar to `one_device_per_thread` 
###### tasks/node < gpus/node, 1 gpus/node, n nodes
This case requires tasks to access additional gpus from other nodes.
TODO i think this is allowed by nccl.
TODO i think this is impossible to set up with MPI?
TODO how to test this node configuration?
###### tasks/node < gpus/node, m gpus/node, n nodes
This can come in 2 flavours:
####### 1 task/node
TODO not tested yet
####### n tasks/node
this seems to crash nccl during collective operations
potential fixes:
- TODO check slurm results for 2540242, 2540274, 2540073

##### tasks/node = gpus/node
this will behave like `one_device_per_process_mpi`. the number of nodes and gpus/node has no influence.
###### tasks/node = gpus/node, 1 gpus/node, 1 node
this case has 1 gpu in total and will not require any gpu communication so it is ignored.
###### tasks/node = gpus/node, m gpus/node, 1 node
this will behave like `one_device_per_process_mpi`
###### tasks/node = gpus/node, 1 gpus/node, n nodes
this will behave like `one_device_per_process_mpi`
###### tasks/node = gpus/node, m gpus/node, n nodes
this will behave like `one_device_per_process_mpi`


##### tasks/node > gpus/node
TODO i think these cases should have equal behavior with the number of nodes and gpus/node having any influece.
###### tasks/node > gpus/node, 1 gpus/node, 1 node
###### tasks/node > gpus/node, m gpus/node, 1 node
###### tasks/node > gpus/node, 1 gpus/node, n nodes
###### tasks/node > gpus/node, m gpus/node, n nodes
this seems to crash nccl during communicator initialization
potential fixes:
- TODO untested 


## 2. What can you do with it

"
Due to the asynchronous nature of NCCL operations, events associated to collective and point-to-point operations are not easy to delimit precisely. StopEvent for collectives simply indicates to the profiler that the collective has been enqueued.
Without both proxy and/or kernel activity it is impossible for the profiler to figure out when a collective operation completes. The profiler can leverage proxy and/or kernel event information, if these are enabled, to estimate when the collective ends.
" (slightly reformulated section from **/ext-profiler/README.md**)

### Logging

- logging function from `init` TODO
- code snippet custom logging infra, timestamping

### Tracking & running metrics

- code snippet showing where to CRUD custom context object
- code snippet showing where to CRUD custom event object

### Kernel tracing with CUPTI

- cupti ext id mechanism briefly explained
- code snippet where to init/cleanup cupti & use cupti mechanism
- changing profiling behaviour during runtime TODO check example_profiler (and inspector?)

## 3. Why would you use it? pros & cons

- customizable
- might require maintenance / active development since NCCL is actively developed

- overhead
  - nvidia advertises their `inspector` implementation as efficient enough for "always-on" in production

- NCCL_DEBUG env var https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-debug 
  - nccl already comes with debug logging which can be set to various levels of granularity
  - INFO - Prints debug information
  - TRACE - Prints replayable trace information on every call.
  - v2.2.12 https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-debug-file
  - v2.3.4 https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-debug-subsys
  - v2.26 https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-debug-timestamp-format
  - v2.26 https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-debug-timestamp-levels

## known limitations https://github.com/NVIDIA/nccl/tree/master/ext-profiler/README.md
Kernel events instrumentation leverages counters exposed by the kernel to the host and the proxy progress thread. Thus, the proxy progress thread infrastructure is shared between the network and the profiler. If the proxy is serving network requests the kernel profiling probing can be delayed, causing loss of accuracy. Similarly, if the CPU is under heavy load and the scheduling of the proxy progress thread is delayed, a similar loss of accuracy can be encountered.

To mitigate this effect, with version 4 of the profiler NCCL uses a per-channel ring buffer of 64 elements. Every counter is complemented by two timestamps (ptimers) supplied by the NCCL kernel (one for start and one for stop of the operation in the kernel). NCCL propagates these timestamps to the profiler plugin that it can convert them to CPU time domain.

## Comparison to MPI TODO ???
MPI:
- centered around cpu processes communicating
- ranks/tasks per cpu process
- rank = CPU process
  
NCCL:
- centered around GPUs communicating. cpu processes/threads exist for communication orchestration
- ranks/tasks possibly per cpu thread
- ranks/tasks possibly assigned to 1 GPU (or many GPUs? TODO)
- rank = GPU device

# TODO
link code snippets to source code files+lines

do pytorch (+jax+tensorflow) use this profiler plugin api?