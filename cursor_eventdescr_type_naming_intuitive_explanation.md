# EventDescr type naming intuition
_Exported on 12/12/2025 at 10:11:07 GMT+1 from Cursor (2.0.77)_

---

**User**

can you explain to me why the different type of eventDescr types are called as they are? i dont find all of them intuitive.

---

**Cursor**


Searching for where `eventDescr` types are defined and used.



Searching for where the event type constants are defined:



Reading the header file with the event type definitions and searching for documentation explaining the naming:



Reviewing usage to clarify the distinctions:



## Event type naming

### API events (with "Api" suffix) vs execution events (without "Api")

- `ncclProfileCollApi`, `ncclProfileP2pApi`, `ncclProfileGroupApi`: Generated when the API is called (host-side, at function entry). These track API call boundaries.
- `ncclProfileColl`, `ncclProfileP2p`: Generated during actual execution (the work being done). These track the real operation.

The "Api" suffix marks the API boundary; the non-Api versions mark execution.

### Special cases

1. `ncclProfileGroup` vs `ncclProfileGroupApi`:
   - `ncclProfileGroupApi`: When `ncclGroupStart()`/`ncclGroupEnd()` are called
   - `ncclProfileGroup`: The group execution container (holds multiple operations)

2. `ncclProfileProxyOp` vs `ncclProfileProxyStep`:
   - `ncclProfileProxyOp`: Summary of proxy activity for a channel (coarser-grained, one per channel)
   - `ncclProfileProxyStep`: Individual network transfer step (finer-grained, multiple steps per ProxyOp)

3. `ncclProfileProxyCtrl`:
   - "Ctrl" = Control. Tracks the proxy progress thread when it's not processing network requests (idle, sleeping, waking, appending work).

4. `ncclProfileKernelCh`:
   - "Ch" = Channel. Tracks GPU kernel activity per channel (device-side execution).

5. `ncclProfileKernelLaunch`:
   - Tracks when NCCL enqueues a CUDA kernel (host-side enqueue), not the actual execution. Execution is tracked by `KernelCh`.

6. `ncclProfileNetPlugin`:
   - Events defined by network plugins (e.g., InfiniBand QP events, socket events).

### Hierarchy

```
GroupApi (API call)
├── CollApi (API call)
│   └── Coll (execution)
│       ├── ProxyOp (network summary)
│       │   └── ProxyStep (individual network transfer)
│       │       └── NetPlugin (network plugin events)
│       └── KernelCh (GPU kernel per channel)
└── P2pApi (API call)
    └── P2p (execution)
        ├── ProxyOp
        └── KernelCh
```

The naming reflects the distinction between API boundaries and execution, and between summary-level and step-level events.

```plaintext
GroupApi (API call)
├── CollApi (API call)
│   └── Coll (execution)
│       ├── ProxyOp (network summary)
│       │   └── ProxyStep (individual network transfer)
│       │       └── NetPlugin (network plugin events)
│       └── KernelCh (GPU kernel per channel)
└── P2pApi (API call)
    └── P2p (execution)
        ├── ProxyOp
        └── KernelCh
```

---

**User**

explain to me the term Proxy in this context. what does proxy activity generally describe/entail?

---

**Cursor**


Searching for documentation and code explaining the NCCL proxy and its role.



Searching for more details on what the proxy thread does:



Reviewing what the proxy progress function does:



## What is the "Proxy" in NCCL?

The proxy is a dedicated CPU thread that handles network communication on behalf of the GPU. It acts as an intermediary between GPU kernels and the network hardware.

### Why does NCCL need a proxy?

1. GPU kernels can't directly access network hardware (InfiniBand, Ethernet, etc.).
2. Network operations are asynchronous and need continuous progress.
3. The proxy thread offloads network work from the GPU, allowing kernels to focus on computation.

### What does proxy activity entail?

The proxy progress thread (`ncclProxyProgress`) continuously:

1. **Network send operations:**
   - Waits for data from GPU memory
   - Posts buffers to the network interface
   - Transmits data over the network
   - Tracks completion

2. **Network receive operations:**
   - Receives data from the network
   - Waits for GPU to be ready
   - Flushes received data to GPU memory
   - Tracks completion

3. **Connection management:**
   - Sets up connections between ranks
   - Manages shared memory buffers
   - Handles connection state transitions

4. **Work queue management:**
   - Appends new operations from the main thread
   - Processes queued network operations
   - Tracks operation state (posted, transmitted, done)

### The proxy thread lifecycle

```
Main Thread (GPU)              Proxy Thread (CPU)
     |                              |
     |-- Enqueue operation -------->|
     |                              |-- Process network work
     |                              |   - Wait for GPU data
     |                              |   - Send/Receive over network
     |                              |   - Flush to GPU
     |<-- Operation complete -------|
```

### Proxy event types in profiling

- **`ncclProfileProxyOp`**: Summary of proxy work for a channel (one per channel, coarser-grained)
- **`ncclProfileProxyStep`**: Individual network transfer step (multiple per ProxyOp, finer-grained)
  - Steps include: waiting for GPU data, sending over network, receiving from network, flushing to GPU
- **`ncclProfileProxyCtrl`**: Proxy thread state when not processing network requests
  - States: idle, active, sleeping, waking, appending work

### Example flow

For a collective like `ncclAllReduce`:

1. GPU kernel processes data locally
2. Proxy thread receives work items
3. For each channel:
   - **ProxyOp** starts (summary for the channel)
   - Multiple **ProxyStep** events occur:
     - Wait for GPU to produce data
     - Send data over network
     - Receive data from network
     - Flush data to GPU
   - **ProxyOp** completes
4. GPU kernel processes the received data

### Why "Proxy"?

It proxies network operations for the GPU: the GPU enqueues work, and the proxy thread executes the network I/O asynchronously, allowing the GPU to continue computing.

In summary: the proxy is NCCL's CPU-side network communication engine that handles all network I/O on behalf of the GPU, enabling asynchronous, overlapped communication and computation.

```plaintext
Main Thread (GPU)              Proxy Thread (CPU)
     |                              |
     |-- Enqueue operation -------->|
     |                              |-- Process network work
     |                              |   - Wait for GPU data
     |                              |   - Send/Receive over network
     |                              |   - Flush to GPU
     |<-- Operation complete -------|
```

