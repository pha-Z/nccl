# Profiler Pool Sizing for Long-Running Applications

## The Problem

The **Example Profiler Plugin** uses fixed-size ring buffers (pools) to store events. The default pool sizes are quite small (256 events per pool), which can be insufficient for long-running HPC applications.

### How It Works

1. **Ring Buffer Design**: Pools act as ring buffers - when full, new events overwrite old ones
2. **Silent Overflow**: When a pool overflows, the profiler silently ignores new events (returns `NULL` handle)
3. **Final Dump**: Only the most recent `poolSize` events are dumped at finalize

### Default Pool Sizes

- `NCCL_PROFILE_GROUP_API_POOL_SIZE`: 256
- `NCCL_PROFILE_COLL_API_POOL_SIZE`: 256
- `NCCL_PROFILE_COLL_POOL_SIZE`: 256
- `NCCL_PROFILE_P2P_API_POOL_SIZE`: 256
- `NCCL_PROFILE_P2P_POOL_SIZE`: 256
- `NCCL_PROFILE_KERNEL_LAUNCH_POOL_SIZE`: 256
- `NCCL_PROFILE_PROXY_CTRL_POOL_SIZE`: 16
- `NCCL_PROFILE_PROXY_DETACH_POOL_SIZE`: 256

**Problem**: For a training run with 10,000 iterations × 4 collectives per iteration = 40,000 collectives, but only the last 256 will be captured!

## Solutions

### Option 1: Increase Pool Sizes (Example Plugin)

Configure larger pool sizes via environment variables:

```bash
# For long-running applications, increase pool sizes significantly
export NCCL_PROFILE_COLL_API_POOL_SIZE=10000
export NCCL_PROFILE_COLL_POOL_SIZE=10000
export NCCL_PROFILE_GROUP_API_POOL_SIZE=5000
export NCCL_PROFILE_P2P_API_POOL_SIZE=5000
export NCCL_PROFILE_P2P_POOL_SIZE=5000
export NCCL_PROFILE_KERNEL_LAUNCH_POOL_SIZE=5000
export NCCL_PROFILE_PROXY_CTRL_POOL_SIZE=1000
export NCCL_PROFILE_PROXY_DETACH_POOL_SIZE=5000
```

**Considerations:**
- **Memory usage**: Each event is ~100-500 bytes, so 10,000 events ≈ 1-5 MB per pool
- **Still limited**: Even with large pools, you only get the *last* N events
- **No early events**: Events from the beginning of a long run will be lost

### Option 2: Use Inspector Plugin (Recommended for Long Runs)

The **Inspector Plugin** uses incremental dumping via a background thread, so it doesn't have overflow issues:

```bash
export NCCL_PROFILER_PLUGIN="$NCCL_PATH/ext-profiler/inspector/libnccl-profiler.so"
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=1000000  # Dump every 1 second
export NCCL_INSPECTOR_DUMP_DIR="nccl-inspector-output"
export NCCL_INSPECTOR_DUMP_VERBOSE=1
```

**Advantages:**
- ✅ **No overflow**: Events are written incrementally to disk
- ✅ **Complete history**: All events are captured, not just recent ones
- ✅ **Lower memory**: Doesn't need to buffer all events in memory
- ✅ **Crash-safe**: Data is written periodically, not just at end

**Trade-offs:**
- Different output format (structured JSON vs Chrome tracing)
- Slightly more I/O overhead (but negligible for most applications)

### Option 3: Periodic Dumps (Custom Solution)

For very long runs, you could:
1. Periodically finalize and recreate communicators (not practical)
2. Use a custom profiler that dumps incrementally
3. Use Inspector plugin with appropriate dump interval

## Recommendations

### For Short Runs (< 1000 iterations)
- **Example Plugin** with default sizes is fine
- Easy to visualize in Chrome tracing

### For Medium Runs (1000-10,000 iterations)
- **Example Plugin** with increased pool sizes
- Or **Inspector Plugin** with 1-5 second dump interval

### For Long Runs (> 10,000 iterations, hours/days)
- **Inspector Plugin** is strongly recommended
- Set dump interval based on your needs:
  - **High frequency** (100ms-1s): More detailed, more I/O
  - **Low frequency** (5-60s): Less I/O, still captures everything
- Use verbose mode for detailed event traces

### For Production HPC Applications
- **Inspector Plugin** with moderate dump interval (1-10 seconds)
- Monitor disk space (output can be large for very long runs)
- Consider post-processing to aggregate/analyze data

## Example: Updating SLURM Scripts

Add pool size configuration to your SLURM scripts:

```bash
# For Example Plugin with large pools
export NCCL_PROFILE_COLL_API_POOL_SIZE=50000
export NCCL_PROFILE_COLL_POOL_SIZE=50000
export NCCL_PROFILE_GROUP_API_POOL_SIZE=10000
# ... other pools

# Or use Inspector Plugin (better for long runs)
export NCCL_PROFILER_PLUGIN="$NCCL_PATH/ext-profiler/inspector/libnccl-profiler.so"
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=1000000  # 1 second
export NCCL_INSPECTOR_DUMP_VERBOSE=1
```

## Memory Estimation

For Example Plugin with custom pool sizes:

```
Total memory ≈ sum of (pool_size × event_size) for all pools

Example with 10K per pool:
- Coll API: 10,000 × ~200 bytes = 2 MB
- Coll: 10,000 × ~300 bytes = 3 MB
- Group API: 5,000 × ~150 bytes = 0.75 MB
- P2P API: 5,000 × ~200 bytes = 1 MB
- P2P: 5,000 × ~250 bytes = 1.25 MB
- Kernel Launch: 5,000 × ~100 bytes = 0.5 MB
- Proxy Ctrl: 1,000 × ~400 bytes = 0.4 MB
- Proxy Detach: 5,000 × ~400 bytes = 2 MB

Total: ~11 MB per communicator
```

For multi-communicator applications, multiply by number of communicators.

## Detecting Overflow

The Example Plugin doesn't warn about overflow. To detect it:
1. Monitor pool indices (would require code modification)
2. Compare expected vs actual event counts in output
3. Use Inspector Plugin which doesn't have this issue

## Summary

| Scenario | Recommended Plugin | Configuration |
|----------|-------------------|---------------|
| Short runs (< 1K iterations) | Example | Default sizes |
| Medium runs (1K-10K iterations) | Example | Increased pool sizes OR Inspector |
| Long runs (> 10K iterations) | **Inspector** | 1-10s dump interval |
| Production HPC | **Inspector** | 1-10s dump interval, verbose mode |

**Bottom line**: For serious HPC benchmarking, use the **Inspector Plugin** with incremental dumping. It's designed for long-running applications and won't lose data due to pool overflow.

