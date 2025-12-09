# SLURM Scripts for NCCL Profiling

This directory contains SLURM scripts for running NCCL profilers on various NCCL examples.

## Overview

These scripts profile NCCL examples using two profiler plugins:
1. **NCCL Inspector Plugin** - Provides structured JSON output with performance metrics and event traces
2. **NCCL Example Plugin** - Provides Chrome tracing format JSON for timeline visualization

## Scripts

### Basic Examples (Single Node)

- `run_profiler_01_communicators_01_single_process.sh` - Communicator init (single process, multiple devices)
- `run_profiler_01_communicators_02_pthread.sh` - Communicator init (pthread, one device per thread)
- `run_profiler_02_point_to_point.sh` - Ring pattern P2P communication
- `run_profiler_03_collectives.sh` - AllReduce collective (single node)
- `run_profiler_04_user_buffer_registration.sh` - User buffer registration
- `run_profiler_05_symmetric_memory.sh` - Symmetric memory registration
- `run_profiler_06_device_api.sh` - Device API (full example from examples/06)

### Multi-Node Examples

- `run_profiler_01_communicators_03_mpi.sh` - Communicator init (MPI, multi-node)
- `run_profiler_03_collectives_mpi.sh` - AllReduce collective (multi-node, MPI)

### Advanced Examples (07_advanced_features)

- `run_profiler_07_cuda_graphs.sh` - Profiles CUDA graphs with NCCL operations
- `run_profiler_07_interleaved.sh` - Profiles interleaved computation-communication pipeline
- `run_profiler_07_multiple_collectives.sh` - Profiles multiple collectives in a group
- `run_profiler_07_multi_stream.sh` - Profiles multi-stream independent operations
- `run_profiler_07_device_api_simple.sh` - Profiles simplified device API example

## Configuration

Before running, adjust these paths in each script to match your environment:

```bash
NCCL_PATH="/data/cat/ws/s0949177-example_ws/nccl"
```

**Note:** We use `NCCL_PATH` not to be confused with `NCCL_HOME` (NVIDIA often uses `NCCL_HOME` environment variable for NCCL builds).

## Prerequisites

1. **Build NCCL examples:**
   ```bash
   cd examples
   make [MPI=1]  # Use MPI=1 for multi-node examples
   ```

2. **Build profiler plugins:**
   ```bash
   # Inspector plugin
   cd ext-profiler/inspector
   make
   
   # Example plugin
   cd ext-profiler/example
   make
   ```

## Usage

Submit a job:
```bash
sbatch slurm-scripts/run_profiler_03_collectives.sh
```

Or run directly (for testing):
```bash
bash slurm-scripts/run_profiler_03_collectives.sh
```

## Output Files

### Inspector Plugin Output
- Directory: `nccl-inspector-<example-name>-<timestamp>/`
- Format: JSON files with per-collective performance data
- Verbose mode: Includes detailed event traces with timestamps

### Example Plugin Output
- Files: `nccl_example_plugin_<example-name>_<timestamp>_<commHash>_<rank>.json`
- Format: Chrome tracing format (can be opened in Chrome://tracing)
- Contains: Complete event hierarchy with timestamps

## Profiler Comparison

### Inspector Plugin
**Best for:**
- Performance metrics (bandwidth, execution time)
- Per-collective analysis
- Structured JSON output for programmatic analysis

**Output includes:**
- Collective name, sequence number, message size
- Execution time, algorithm bandwidth, bus bandwidth
- Optional verbose event traces with timestamps

### Example Plugin
**Best for:**
- Timeline visualization
- Event hierarchy understanding
- Chrome tracing integration
- Detailed GPU kernel and network activity

**Output includes:**
- Complete event hierarchy (Group API → Collective API → Collective → ProxyOp → KernelCh)
- Timestamps for all events
- GPU clock information
- Network transfer details

## Which Profiler to Use?

For **reconstructing GPU activity and communication operations**, both profilers can work, but:

- **Example Plugin** is better for **timeline visualization** - you can see exactly when each GPU participates in which operation
- **Inspector Plugin** is better for **performance analysis** - you get clean metrics per operation

**Recommendation:** Start with the **Example Plugin** for timeline reconstruction, then use **Inspector Plugin** for performance analysis. The Example Plugin's Chrome tracing format makes it easy to visualize which GPU did what and when.

## Custom Profiler?

The existing profilers should be sufficient for most use cases:
- **Example Plugin** provides comprehensive timeline data
- **Inspector Plugin** provides structured performance data

However, if you need:
- Custom output formats
- Specific filtering or aggregation
- Integration with other tools
- Specialized analysis

You can build a custom profiler following the guide in `ext-profiler/WRITING_YOUR_PLUGIN.md`.

## Notes

- Inspector plugin verbose mode is enabled for detailed event traces
- Example plugin captures all events (mask=4095)
- Output files are timestamped to avoid overwrites
- Multi-node script requires MPI-enabled NCCL examples

## Pool Sizing for Long-Running Applications

The Example Plugin uses fixed-size ring buffers (default 256 events per pool). For long-running HPC applications, this may cause event loss due to overflow.

**For long runs (> 1000 iterations):**
- **Option 1**: Increase pool sizes via environment variables (see `PROFILER_POOL_SIZING.md`)
- **Option 2**: Use Inspector Plugin instead (recommended) - it dumps incrementally and doesn't have overflow issues

See `PROFILER_POOL_SIZING.md` for detailed guidance on pool sizing and recommendations for production HPC applications.

