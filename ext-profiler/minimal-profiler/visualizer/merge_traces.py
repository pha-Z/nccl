#!/usr/bin/env python3
"""
Merge per-process JSONL trace files into a single trace.jsonl file.

The C++ profiler writes per-process trace files to avoid write interleaving
on network filesystems (NFS, Lustre, GPFS). This script merges them into
a single trace.jsonl file for visualization.

Usage:
    python merge_traces.py <dump_directory> [output_directory]
    
Examples:
    python merge_traces.py /path/to/dump                    # Output to dump directory
    python merge_traces.py /path/to/dump /path/to/output    # Output to specific directory
    python merge_traces.py /path/to/dump .                  # Output to current directory
"""

import sys
import glob
from pathlib import Path


def find_trace_files(search_dir: Path) -> list:
    """Find per-process JSONL trace files in the given directory.
    
    Looks for files matching: trace_*_pid*.jsonl
    
    Returns a sorted list of Path objects.
    """
    if not search_dir.exists() or not search_dir.is_dir():
        return []
    
    pid_pattern = str(search_dir / 'trace_pid*.jsonl')
    pid_matches = glob.glob(pid_pattern)
    
    if pid_matches:
        pid_matches.sort()
        return [Path(p) for p in pid_matches]
    
    return []


def merge_trace_files(trace_files: list, output_file: Path) -> bool:
    """Merge multiple trace files into a single output file.
    
    Args:
        trace_files: List of Path objects pointing to JSONL trace files
        output_file: Path for the merged output file
    
    Returns:
        True on success, False on error
    """
    if not trace_files:
        print("Error: No trace files to merge", file=sys.stderr)
        return False
    
    try:
        with open(output_file, 'w') as outfile:
            for trace_file in trace_files:
                print(f"  Adding: {trace_file.name}")
                with open(trace_file, 'r') as infile:
                    for line in infile:
                        outfile.write(line)
                        # Ensure each line ends with newline
                        if not line.endswith('\n'):
                            outfile.write('\n')
        return True
    except Exception as e:
        print(f"Error merging trace files: {e}", file=sys.stderr)
        return False


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python merge_traces.py <dump_directory> [output_directory]", file=sys.stderr)
        print("", file=sys.stderr)
        print("Merges per-process JSONL trace files into a single trace.jsonl file.", file=sys.stderr)
        sys.exit(1)
    
    dump_dir = Path(sys.argv[1])
    
    if not dump_dir.exists():
        print(f"Error: Directory {dump_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    if not dump_dir.is_dir():
        print(f"Error: {dump_dir} is not a directory", file=sys.stderr)
        sys.exit(1)
    
    # Determine output directory
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
        if not output_dir.exists():
            print(f"Error: Output directory {output_dir} does not exist", file=sys.stderr)
            sys.exit(1)
    else:
        output_dir = dump_dir
    
    # Find trace files
    trace_files = find_trace_files(dump_dir)
    
    if not trace_files:
        print(f"Error: No trace files found in {dump_dir}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Expected files matching:", file=sys.stderr)
        print("  trace_*_pid*.jsonl", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(trace_files)} trace file(s) in {dump_dir}")
    
    # Merge files
    output_file = output_dir / 'trace.jsonl'
    print(f"Merging to: {output_file}")
    
    if not merge_trace_files(trace_files, output_file):
        sys.exit(1)
    
    print(f"Successfully merged {len(trace_files)} file(s) into {output_file}")
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)

