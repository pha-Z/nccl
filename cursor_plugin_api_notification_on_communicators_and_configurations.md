# Plugin API, communicators, and configurations
_Exported on 12/5/2025 at 17:50:56 GMT+1 from Cursor (2.0.77)_

---

**User**

does the plugin api say anything about communicator creation? does the profiler plugin know when a communicator is created and destroyed? there are also multiple possible setups (1 comm per device, n devices 1 process, 1 device per thread)... whats the impact?

---

**Cursor**


Reviewing the plugin API and inspector plugin code for communicator creation/destruction hooks and how different setups are handled.



Checking the profiler API header and how NCCL calls the plugin functions:



Checking how NCCL calls the plugin init and finalize functions:



Checking how the inspector plugin handles multiple communicators:



Reviewing how the inspector handles multiple communicators:



## Plugin API and Communicator Lifecycle

Yes, the plugin API tracks communicator creation and destruction:

1. `init` is called once per communicator when it's created:
   - Called from `ncclProfilerPluginInit()` in `src/plugin/profiler.cc` (line 187)
   - Invoked during communicator initialization in `src/init.cc` (line 1251)
   - Receives: `commId` (commHash), `commName`, `nNodes`, `nranks`, `rank`
   - Returns a `context` pointer for per-communicator state

2. `finalize` is called once per communicator when it's destroyed:
   - Called from `ncclProfilerPluginFinalize()` in `src/plugin/profiler.cc` (line 201)
   - Invoked during `commFree()` in `src/init.cc` (line 336)
   - Receives the same `context` pointer from `init`

## Different Communicator Setups

The plugin API does not distinguish between setups. It sees individual communicators:

1. Multiple devices, single process (`ncclCommInitAll`):
   - One `init` call per communicator (one per device)
   - All in the same process, potentially concurrent

2. One device per thread (pthread):
   - One `init` call per communicator (one per thread)
   - Thread-safe handling required

3. One device per process (MPI):
   - One `init` call per communicator (one per process)
   - Each process has its own plugin instance

## Impact and Requirements

1. Multiple communicators: The plugin must handle multiple communicators concurrently. The inspector plugin maintains a global list (`g_state.liveComms`) to track them.

2. Thread safety: If multiple communicators can be active simultaneously (e.g., multiple devices in one process), the plugin must be thread-safe. The inspector uses locks (see `inspectorAddComm`/`inspectorDelComm` in `inspector.cc`).

3. Per-communicator state: Use the `context` pointer to store per-communicator data. The inspector stores `inspectorCommInfo` per communicator.

4. No setup awareness: The plugin does not know the setup (process/thread/device mapping). It only sees individual communicators with their `commHash`, `rank`, `nranks`, etc.

The API is designed to be setup-agnostic: each communicator gets its own `init`/`finalize` pair, and the plugin manages them accordingly.

