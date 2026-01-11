#!/usr/bin/env python3
"""
Start a web server to view the NCCL profiler timeline visualization.

This script serves the visualizer files and opens the browser. It expects
trace.jsonl to already exist in the visualizer directory.

To prepare trace files:
1. Run the profiler (generates per-process trace_*_pid*.jsonl files)
2. Run merge_traces.py to create trace.jsonl
3. Run this script to view the visualization

Usage:
    python run_visualizer.py [port]
    
Examples:
    python run_visualizer.py          # Use default port 8000
    python run_visualizer.py 8080     # Use custom port
"""

import sys
import os
import webbrowser
import http.server
import socketserver
import threading
import time
from pathlib import Path


def start_server(port=8000):
    """Start a simple HTTP server."""
    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), handler)
    print(f"Starting HTTP server on port {port}...")
    print(f"Open http://localhost:{port}/visualize_timeline.html in your browser")
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(1)
        webbrowser.open(f"http://localhost:{port}/visualize_timeline.html")
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()


def main():
    """Main entry point."""
    visualizer_dir = Path(__file__).parent
    trace_file = visualizer_dir / 'trace.jsonl'
    viz_file = visualizer_dir / 'visualize_timeline.html'
    
    # Check if visualization file exists
    if not viz_file.exists():
        print(f"Error: {viz_file} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Check if trace file exists
    if not trace_file.exists():
        print(f"Error: {trace_file} does not exist", file=sys.stderr)
        print("", file=sys.stderr)
        print("To prepare trace files:", file=sys.stderr)
        print("  1. Run the profiler to generate per-process trace files", file=sys.stderr)
        print("  2. Run: python merge_traces.py <dump_directory> <visualizer_directory>", file=sys.stderr)
        print("  3. Run this script again", file=sys.stderr)
        sys.exit(1)
    
    print(f"Using trace file: {trace_file}")
    
    # Parse port from command line
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}", file=sys.stderr)
            sys.exit(1)
    
    # Change to visualizer directory before starting server
    os.chdir(visualizer_dir)
    
    start_server(port)


if __name__ == '__main__':
    main()
