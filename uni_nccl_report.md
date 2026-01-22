# NCCL Profiler Plugin API - eine Machbarkeitsstudie

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

Through following sections the feasability of the NCCL profiler plugin API will be made clear:
1. How does it work? 
2. What can you do with it?
3. Why would you use it? pros & cons

## 1. How it works
### 1.1 How nccl detects the profiler plugin
NCCL looks for a shared library which represents the profiler plugin, checking for the environment variable NCCL_PROFILER_PLUGIN: `profilerName = ncclGetEnv("NCCL_PROFILER_PLUGIN")`. 

It then calls `dlopen(name, RTLD_NOW | RTLD_LOCAL)` to load the library immediately with local symbol visibility

"

- If NCCL_PROFILER_PLUGIN is set, attempt loading the library with name specified by NCCL_PROFILER_PLUGIN;
- If NCCL_PROFILER_PLUGIN is set and previous failed, attempt loading libnccl-profiler-<NCCL_PROFILER_PLUGIN>.so;
- If NCCL_PROFILER_PLUGIN is not set, attempt loading libnccl-profiler.so;
If no plugin was found (neither user defined nor default), do not enable profiling.
- If NCCL_PROFILER_PLUGIN is set to `STATIC_PLUGIN`, the plugin symbols are searched in the program binary.

" (quoted from docs: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-profiler-plugin)

The exact implementation details can be found at **/src/plugin/plugin_open.cc** and **/src/plugin/profiler.cc**

### 1.2 The profiler API
The plugin must implement a profiler API specified by NCCL, in the form of exposing a struct. This struct should contain pointers to all the functions required by the API
```c
typedef struct {
  const char* name;
  ncclResult_t (*init)(...);
  ncclResult_t (*startEvent)(...);
  ncclResult_t (*stopEvent)(...);
  ncclResult_t (*recordEventState)(...);
  ncclResult_t (*finalize)(...);
} ncclProfiler_v5_t;
```
The struct type itself also indicates which API version is implemented.
The profiler API has changed multiple times with newer NCCL releases. New releases seem to have limited backwards compatibility with older plugins (TODO requires fact check. iirc there was some verison breaking feature that didnt allow backwards compatibilty to v2? but new nccl relreases certainly still feature the structs for older versions @/src/include/plugin/profiler).
A plugin may expose multiple versioned structs to allow for backwards compatibility with older NCCL releases.

"

The NCCL code is instrumented with profiler callbacks at different levels to capture start/stop of groups, collective and point-to-point operations, as well as proxy, kernel and network activity

" (quote from **/ext-profiler/README.md**)

As the API function names suggest, this will allow the profiler to track these activities as events.

The exact profiler API can be found under **/src/include/plugin/profiler/**. As of NCCL release v2.29.1, five functions must be implemented.

`init` initializes the profiler plugin. NCCL passes various arguments as shown below. 
TODO the exact points where NCCL calls the API will be discussed in section #### 1.3 TODO make this a link
```c
ncclResult_t init(
  void** context,         // opaque profiler context object for separating profiler behavior across communicators
  uint64_t commId,         // communicator id
  int* eActivationMask,    // bitmask that sets which events are tracked the plugin
  const char* commName,    // user assigned communicator name
  int nNodes,              // number of nodes in communicator
  int nranks,              // number of ranks in communicator
  int rank,                // rank identifier in communicator
  ncclDebugLogger_t logfn  // logger function
);
```
Notably, `void** context` is an opaque handle that the plugin developer should point to any custom context object. This pointer is then again passed as argument in the other API calls `startEvent` and `finalize`. This context object is separate across communicators.

the plugin developer should point `int* eActivationMask` to a bitmask, which tells nccl which type of events that the profiler plugin wants to track. Each bit corresponds to a different event type. The mappings can be found at **/src/include/plugin/nccl_profiler.h**. internally this bitmask is initialized with `0`, which means no events will be tracked by default. Setting it to `4095` will track all events.

TODO what to do with `ncclDebugLogger_t logfn` ?


`startEvent` is called at various points in NCCL. it tells the plugin that a new event has started.
TODO the exact points where NCCL calls the API will be discussed in section #### 1.3 TODO make this a link

TODO "
As soon as NCCL finds the plugin and the correct ncclProfiler symbol, it calls its init function. This allows the plugin to initialize its internal context, used during profiling of NCCL events. If the init function does not return ncclSuccess, NCCL disables the plugin.
" (quote from **/ext-profiler/README.md**)

```c
ncclResult_t startEvent(
  void* context,                       // opaque profiler context object
  void** eHandle,                      // event handle for event
  ncclProfilerEventDescr_v5_t* eDescr  // pointer to ncclProfilerEventDescr_t object
);
```
The plugin developer should point `void** eHandle` to a custom event object. this pointer is then again passed as argument in the other API calls `stopEvent` and `recordEventState` to refer to the same event object. `void* context` passes the context object for this communicator in the profiler

`ncclProfilerEventDescr_v5_t* eDescr` is a struct that contains all the information that nccl populates to describe the started event. Exact details can be found under the profiler API **/src/include/plugin/profiler/**. Notably, the event descriptor struct contains a field `void* parentObj`, which is the event handle to the parent event of this event (`null` if it has no parent). The different types of events follow an event hierarchy, with parent events preceding the child event.

"

NCCL core events (reported above) are organized into a hierarchy as reported below:
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

As a consequence of the `eDescr` having the `parentObj` field, event types of parent events will also be tracked if the profiler enables tracking for event types of events that are children of those.

Unless the user sets the environment variable `NCCL_PXN_DISABLE` to 0 (defaults to 1) to disable the PXN feature (PCI x NVLink - inter-node communication using a non-local NIC, using NVLink and an intermediate GPU), this causes some proxy operations to be processed in a remote proxy thread that differs from the one that generated the operation. When this happens, dereferencing the parent event, provided by NCCL in the event descriptor during `startEvent()`, is not possible and the event hierarchy reported above may not be utilized. The event descriptor for `ProxyOp` events includes the PID of the originator, which the profiler plugin can match against the local PID to check if the remote proxy thread is in the same address space of the proxy thread originating the operation to dereference the parent event. 

Unfortunately, the ProxyOp' child event `ProxyStep` do not provide this field. However `startEvent()` also passes the `context` of the event, where a similar dynamic is present and it sometimes should not be dereferenced because the PXN feature. The profiler plugin may internally track which context objects have been created on this process over (potentially multiple thread-level) `init()` calls. If the passed `context` object during `startEvent()` is not present in the profilers tracked contexts, then this indicates that the event did not originate from this process because of the PXN feature. 
TODO See #### 1.3 Where NCCL triggers profiler API callbacks for more information on when `init()` is called TODO make this a link


`stopEvent` tells the plugin that the event has stopped.
TODO the exact points where NCCL calls the API will be discussed in section #### 1.3 TODO make this a link
```c
ncclResult_t stopEvent(void* eHandle);  // handle to event object
```

Some event types may be updated through calls to `recordEventState`, which can update the state, along with event attributes. These type of events can go through several states during their lifecycle. The list of supported states for the updatable events can be found under the profiler API **/src/include/plugin/profiler/**
The exact points where NCCL calls these functions WILL NOT be discussed in this report TODO likely true, but should come back here in case I do
```c
ncclResult_t recordEventState(
  void* eHandle,                               // handle to event object created through startEvent
  ncclProfilerEventState_v5_t eState,          // event state transition
  ncclProfilerEventStateArgs_v5_t* eStateArgs  // optional argument used to capture event attribute updates associated with the state transition
);
```

When the profiler for a communicator is no longer needed, a call to `finalize()` is made and should destroy the profiler context and free up resources.
TODO See #### 1.3 Where NCCL triggers profiler API callbacks for more information on when `finalize()` is called TODO make this a link
```c
ncclResult_t finalize(void* context);  // opaque profiler context object
```

besides these functions, the profiler plugin struct also contains a `name` field.

"

The name field should point to a character string with the name of the profiler plugin. This will be used for all logging, especially when NCCL_DEBUG=INFO is set.

" (quote from **/ext-profiler/README.md**)

### 1.3 Where nccl triggers profiler API callbacks

Below, the callbacks to the profiler API will be explained. They can be categorized in following groups:
- callbacks for profiler initialization and finalization
- callbacks to the profiler API for events directly related to the application calling the NCCL API (e.g. `ncclAllReduce()`)
- callbacks to the profiler API for network events triggered by the proxy progress thread processing network requests
- TODO add copy engine based events section?

- callbacks to the profiler API for events related to the application calling the NCCL API (e.g. `ncclAllReduce()`)

#### Callbacks for profiler initialization and finalization

The profiler API's `init()` function is called in `ncclProfilerPluginInit()` during the initialization of nccl. This function call first triggers the profiler loading mechanism described in section 1.2 TODO make this a link. If the loading succeeds, the profiler plugins's `init()` function is called. 

NCCL calls `ncclProfilerPluginInit()` during its initialization after a user API call to create ranks for a new communicator. As of NCCL release 2.29.1, the following
depicts the flow from user API call to NCCL calling ncclProfilerPluginInit():

```plaintext
User API                          Internal Flow
─────────────────────────────────────────────────────────────────────────────────
ncclCommInitRank()          ─┐
ncclCommInitAll()            │
ncclCommInitRankConfig()     ├──► ncclCommInitRankDev() ────┐
ncclCommInitRankScalable()  ─┘                              │
                                                            │
ncclCommSplit()             ─┐                              │
ncclCommShrink()            ─┴──► ncclCommInitChildComm()───┤
                                                            │
ncclCommGrow()              ────────────────────────────────┤
                                                            │
                                                            ▼
                                                    ncclCommInitRankFunc()
                                                            │
                                                            ▼
                                                    initTransportsRank()
                                                            │
                                                            ▼
                                                    ncclProfilerPluginInit(comm)
                                                            │
                                                            ▼
                                                    ncclProfilerPluginLoad()
                                                    profiler->init()
```

The implementation can be found at **/src/init.cc** and **/src/plugin/profiler.cc**.

The profiler API's `finalize()` function is called in `ncclProfilerPluginFinalize()` 
after a user API call is made to free resources associated with the communicator object. Afterwards the profiler is also unloaded by eventually calling `dlclose(handle);`

```plaintext
User API                          Internal Flow
─────────────────────────────────────────────────────────────────────
ncclCommAbort()    ─┐
ncclCommDestroy()  ─┴───────────► commReclaim() ──┐
                                                  │
                                                  ▼
                                      ncclProfilerPluginFinalize()
                                                  │
                                                  ▼
                                      ncclProfiler->finalize()
                                      ncclProfilerPluginUnload()

```
See implementation details at **/src/init.cc**, **/src/plugin/profiler.cc** and **/src/plugin/plugin_open.cc**.

#### Callbacks for events related to NCCL API calls

"

NCCL profile API events are generated when the API calls are made, right after NCCL checks for graph capture information. They parent collective, point-to-point and kernel launch events and persist across multiple operations in a group.

" (quote from from **/ext-profiler/README.md**)

TODO: @cursor_nccl_profiling_event_flow.md

#### proxy progress thread - callbacks when processing network requests

TODO: @cursor_nccl_profiling_event_flow.md

during nccl initialization, after calling `ncclProfilerPluginInit()`, `ncclProxyCreate(comm)` is called. This creates a new thread `ncclProxyService`. If needed this Proxy Service will launch another thread `ncclProxyProgress`. Proxy Progress is created only once by checking `!state->thread` which is a process wide shared resource.

"

```c
  /* proxyState is shared among parent comm and split comms. comm->proxyState->thread is
   * pthread_join()'d by commFree() in init.cc when the refCount reduces down to 0. */
```

" (quote from **/src/proxy.cc**) 


Callbacks to the profiler API for network progress will happen from this Proxy Progress Thread.

The code that causes this chain is featured below
as pseudo code snippets:


**/src/init.cc**
```c
initTransportsRank() {
  ncclProfilerPluginInit(comm);
  ncclProxyCreate(comm);
    pthread_create(&comm->proxyState->thread, NULL, ncclProxyService, comm->proxyState)
}
```

**/src/proxy.cc**
```c
void* ncclProxyService(void* _args) {
  proxyProgressAsync(..., proxyState, ...);

  /* different code path */ 

  proxyServiceInitOp(..., proxyState, ...);
    proxyProgressAsync(..., proxyState, ...);
}

static ncclResult_t proxyProgressAsync(..., struct ncclProxyState* proxyState, ...) {
  proxyConnInit(..., proxyState, ...) {
    proxyProgressInit(proxyState) {
      ncclProxyProgressCreate(proxyState) {
        struct ncclProxyProgressState* state = &proxyState->progressState;
        if (!state->thread) pthread_create(&state->thread, NULL, ncclProxyProgress, proxyState);     
      }
    }
  }
}

ncclProxyProgress(proxyState){
  // Emits ProxyStep events
  progressOps(proxyState, ..., opStart=state->active, ...) {
      struct ncclProxyArgs* op = opStart;
      while (op) {
          op->progress(proxyState, op)
          op = op->next;
      }
  }

  // Emits ProxyCtrl events when transitioning between idle/active
  ncclProfilerStartProxyCtrlEvent(...);
  if (lastIdle == 0 && idle == 1) ncclProfilerRecordProxyCtrlEventState(..., ncclProfilerProxyCtrlIdle);
  if (lastIdle == 1 && idle == 0) ncclProfilerRecordProxyCtrlEventState(..., ncclProfilerProxyCtrlActive);
  ncclProfilerStopProxyCtrlEvent(...);

  ncclProxyGetPostedOps(proxyState) {
    struct ncclProxyProgressState* state = &proxyState->progressState;
    struct ncclProxyOpsPool* pool = state->opsPool;
    
    // Emits ProxyCtrl events when transitioning between sleep/wakeup
    if (state->active == NULL) {
      pthread_mutex_lock(&pool->mutex);
      if (pool->nextOps == -1 && !state->stop) {
        ncclProfilerStartProxyCtrlEvent();
        
        ncclProfilerRecordProxyCtrlEventState(..., ncclProfilerProxyCtrlSleep);
        pthread_cond_wait(&pool->cond, &pool->mutex);
        ncclProfilerRecordProxyCtrlEventState(..., ncclProfilerProxyCtrlWakeup);
        
        ncclProfilerStopProxyCtrlEvent();
      }
    }

    // update pool
    state->nextOps = pool->nextOps;
    pool->nextOps = pool->nextOpsEnd = -1;
    pthread_mutex_unlock(&pool->mutex);

    // Emits ProxyCtrl events when appending proxyOps
    ncclProfilerStartProxyCtrlEvent();
    ncclProfilerRecordProxyCtrlEventState(..., ncclProfilerProxyCtrlAppend);
    ProxyAppend() {
        
      ncclProxyOpToArgs()
      // Emits ProxyOp events
      if (args->pattern != ncclPatternProfiler) ncclProfilerStartProxyOpEvent();
    }
  }
}
```

`op->progress()` progresses transport specific ops. this is just a function pointer type (similar to interfaces in OOP) defined at **/src/include/proxy.h**

```c
typedef ncclResult_t (*proxyProgressFunc_t)(struct ncclProxyState*, struct ncclProxyArgs*);
proxy

struct ncclProxyArgs {
  proxyProgressFunc_t progress;
  
  struct ncclProxyArgs* next;

/* other fields */
}
```
(confusingly the variable is called `op`, although its type is `ncclProxyArgs` and NOT `ncclProxyOp`).

Which specific function this calls depends on the op. check the different ProxyProgress functions under 
**/src/transport/net.cc**, **/src/transport/coll_net.cc**, **/src/transport/p2p.cc**, **/src/transport/shm.cc** and **/src/transport/shm.cc**.

In these specific functions, calls to the profiler API are eventually made. These notify the profiler plugin about the following type of events
- `ProxyStep`
- `KernelCh`
- Network plugin specific events

===========

The following paths eventually reach `ncclProxyPost()` where Ops will be posted to a pool and proxy progress thread will be signalled. The thread reads from this pool when calling `ncclProxyGetPostedOps()`

```c
any of these NCCL API calls (
ncclCommInitAll()  // init.cc
ncclCommInitRankConfig()  // init.cc
ncclCommInitRankScalable()  // init.cc
ncclCommFinalize()  // init.cc
ncclCommDestroy()  // init.cc
ncclCommRevoke()  // init.cc
ncclCommAbort()  // init.cc
ncclCommSplit()  // init.cc
ncclCommShrink()  // init.cc
ncclCommGrow()  // init.cc
ncclDevCommCreate()  // dev_runtime.cc
ncclCommWindowRegister()  // dev_runtime.cc
ncclGroupSimulateEnd()  // group.cc
ncclGroupEnd()  // group.cc
) {
  ncclGroupEndInternal()
}
```

```c
any of these NCCL API calls (
ncclAllGather() // collectives.cc
ncclAlltoAll() // collectives.cc
ncclAllReduce() // collectives.cc
ncclBroadcast() // collectives.cc
ncclGather() // collectives.cc
ncclReduce() // collectives.cc
ncclReduceScatter() // collectives.cc
ncclScatter() // collectives.cc
ncclSend() // collectives.cc
ncclRecv() // collectives.cc
) {
  ncclEnqueueCheck() { // enqueue.cc
    ncclGroupEndInternal()
  }
}
```

```c
ncclGroupEndInternal() {
  if (...) {
    groupLaunchNonBlocking()
      groupLaunch(&groupJob->base, NULL)
  } else {
    groupLaunch(&groupJob->base, internalSimInfoPtr)
  }
}
```

```c
groupLaunch(&groupJob->base, internalSimInfoPtr) {
  doLaunches(groupCommHeadMain[ncclGroupTaskTypeCollective])
}
```

```c
doLaunches(struct ncclComm* head) {
  plan = comm->planner.unlaunchedPlansHead;
  ncclLaunchKernelAfter_NoCuda(comm, plan) {
    hostStreamPlanTask(comm, plan)
  }

  /* different code path */ 

  ncclLaunchPrepare(comm) {
    plan = ncclMemoryPoolAlloc<struct ncclKernelPlan>(...);
    cudaLaunchHostFunc(hostStream, hostStreamPlanCallback, plan) {
      hostStreamPlanCallback(void *plan_) {
        hostStreamPlanTask(plan->comm, plan)
      }
    }
  }
}
```
    
```c
hostStreamPlanTask(comm, plan) {
  uploadProxyOps(comm, plan) {
    ncclProxySaveOp() {
      SaveProxy() {
        ncclLocalOpAppend() {
          if (...) {
            ncclProxyPost()
          }
        }
      }
    }
  }
  ncclProxyStart(comm) {
    ncclProxyPost()
  }
}
```

```c
ncclResult_t ncclProxyPost(struct ncclProxyOpsPool* pool, int nextOps, int nextOpsEnd) {
  pthread_mutex_lock(&pool->mutex);
  if (...) {
    pool->nextOps = nextOps;
    pthread_cond_signal(&pool->cond);
  } 
  pool->nextOpsEnd = nextOpsEnd;
  pthread_mutex_unlock(&pool->mutex);
}
```

#### TODO include copyengine events (v2.29.1)?


#### TODO section intro title?
In a multi node HPC environment, a typical case for NCCL applications, besides the behaviour of NCCL making callbacks to the profiler, NCCL itself gets instantiated multiple times, which affects the total number of callbacks to the profiler. To better understand the profiling behaviour, the different settings below will each demonstrate the behaviour of callbacks to the profiler plugin.

TODO the next two subpoints below is one suggestion on how to think about different scenarios (number of nccl launches, profiler launches, per node, per thread...).
not sure how much sense they make or if some other way of differentiation is better. work through the individual questions/points/todos within those subpoints and it will become clear what makes most sense!
#### callback behaviour under different communicator initialization strategies
TODO does `one_device_per_process_mpi` and `one_device_per_thread` behave the same? does any of the these init strats matter to behaviour of profiler?
TODO i think from nccl persepective it doesnt care - it just uses ncclUniqueId down to a thread-level granularity to identify communicator groups?
##### strategy `one_device_per_process_mpi`
##### strategy `one_device_per_thread` 
##### strategy using custom network socket code instead of MPI
TODO

#### callback behaviour under specific node configurations
n nodes, m gpus/per node and p tasks on m gpus allow many theoretical cases for node configurations by combination:
- 1 node, n nodes
- 1 gpu/node, m gpus/node
- p tasks/node, where tasks/node < gpus/node, tasks/node = gpus/node, tasks/node > gpus/node
  
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



## 3. Why would you use it? pros & cons

## known limitations https://github.com/NVIDIA/nccl/tree/master/ext-profiler/README.md
Kernel events instrumentation leverages counters exposed by the kernel to the host and the proxy progress thread. Thus, the proxy progress thread infrastructure is shared between the network and the profiler. If the proxy is serving network requests the kernel profiling probing can be delayed, causing loss of accuracy. Similarly, if the CPU is under heavy load and the scheduling of the proxy progress thread is delayed, a similar loss of accuracy can be encountered.

To mitigate this effect, with version 4 of the profiler NCCL uses a per-channel ring buffer of 64 elements. Every counter is complemented by two timestamps (ptimers) supplied by the NCCL kernel (one for start and one for stop of the operation in the kernel). NCCL propagates these timestamps to the profiler plugin that it can convert them to CPU time domain.

## Comparison to MPI TODO ???

# TODO
link code snippets to source code files+lines

do pytorch (+jax+tensorflow) use this profiler plugin api?