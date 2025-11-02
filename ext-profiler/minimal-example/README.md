# Minimal Profiler Plugin Example

This is a simplified profiler plugin that demonstrates the essential structure needed to create your own plugin.

## What This Example Does

This minimal plugin:
- Logs all NCCL events to a text file (one file per rank)
- Shows start/stop times and durations
- Demonstrates the 5 required functions
- Is easy to understand and modify

## Building

1. **Copy the NCCL headers** from `../example/nccl/` to a `nccl/` subdirectory:
   ```bash
   mkdir -p nccl
   cp ../example/nccl/*.h nccl/
   ```

2. **Compile:**
   ```bash
   gcc -fPIC -shared -o libnccl-profiler-minimal.so my_profiler.c \
       -I./nccl -I${CUDA_HOME}/include
   ```

   Or with C++ (if you need C++ features):
   ```bash
   g++ -fPIC -shared -o libnccl-profiler-minimal.so my_profiler.c \
       -I./nccl -I${CUDA_HOME}/include
   ```

3. **Create symlink (optional):**
   ```bash
   ln -s libnccl-profiler-minimal.so libnccl-profiler.so
   ```

## Usage

1. **Set environment variable:**
   ```bash
   export NCCL_PROFILER_PLUGIN=minimal
   # Or set to full path:
   # export NCCL_PROFILER_PLUGIN=/path/to/libnccl-profiler-minimal.so
   ```

2. **Set event mask (optional):**
   ```bash
   # Enable collectives and P2P (default)
   export NCCL_PROFILE_EVENT_MASK=258  # ncclProfileColl | ncclProfileP2p
   
   # Enable all events
   export NCCL_PROFILE_EVENT_MASK=4095
   ```

3. **Run your NCCL application** - it will automatically load the plugin

4. **Check output files:**
   ```bash
   cat my_profiler_rank0_comm*.log
   ```

## Example Output

```
=== Profiler initialized ===
Rank: 0/2, CommId: 123456789, CommName: (null)
[0.000] START GroupAPI (depth=2)
[0.001] START Collective: ncclAllReduce(count=1048576, dtype=ncclFloat32, root=-1)
[0.002] START Coll Exec: ncclAllReduce(seq=0, algo=RING, proto=SIMPLE, channels=2)
[1.234] START ProxyOp: channel=0, peer=1, steps=4, send=1
[1.250] STOP ProxyOp (duration=0.016 ms)
[5.678] STOP Coll Exec: ncclAllReduce (duration=5.676 ms)
[5.679] STOP Collective: ncclAllReduce (duration=5.678 ms)
[5.680] STOP GroupAPI (duration=5.680 ms)
=== Profiler finalized ===
```

## Customizing

To customize this plugin:

1. **Change logging format:** Modify the `fprintf` statements in `myStartEvent` and `myStopEvent`

2. **Add statistics:** Keep counters in `MyContext` and aggregate in `finalize`

3. **Write to different format:** Instead of text, write JSON, CSV, or binary format

4. **Filter events:** Check event type and only log what you care about

5. **Add memory pools:** For better performance, pre-allocate events (see example plugin)

## Next Steps

Once you understand this minimal example:
- Study the full example in `../example/` for advanced features
- Read `../WRITING_YOUR_PLUGIN.md` for detailed explanations
- Check `../README.md` for API documentation

