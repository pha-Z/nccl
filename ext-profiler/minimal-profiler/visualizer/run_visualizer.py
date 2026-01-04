#!/usr/bin/env python3
"""
Helper script to parse logs and start a web server for visualization.
"""

import sys
import subprocess
import webbrowser
import http.server
import socketserver
import threading
import time
import shutil
from pathlib import Path

def parse_logs(dump_dir):
    """Run the log parser."""
    print(f"Parsing log files from {dump_dir}...")
    result = subprocess.run(
        [sys.executable, "parse_logs.py", str(dump_dir)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Error parsing logs: {result.stderr}", file=sys.stderr)
        return False
    print(result.stdout)
    return True

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
    if len(sys.argv) > 1:
        arg_path = Path(sys.argv[1])
        
        # Check if argument is a JSON file (trace file)
        if arg_path.suffix.lower() == '.json':
            # Treat as trace file - copy it to events_data.json and skip parsing
            if not arg_path.exists():
                print(f"Error: Trace file {arg_path} does not exist", file=sys.stderr)
                sys.exit(1)
            
            events_data_file = Path(__file__).parent / 'events_data.json'
            print(f"Using provided trace file: {arg_path}")
            shutil.copy2(arg_path, events_data_file)
            print(f"Copied to {events_data_file}")
        elif arg_path.is_file():
            # It's a file but not JSON - error
            print(f"Error: {arg_path} is a file but not a JSON trace file", file=sys.stderr)
            sys.exit(1)
        else:
            # It's a directory - parse logs from it
            dump_dir = arg_path
            if not dump_dir.exists():
                print(f"Error: Directory {dump_dir} does not exist", file=sys.stderr)
                sys.exit(1)
            
            # Parse logs
            if not parse_logs(dump_dir):
                sys.exit(1)
    else:
        # When no argument is provided, check if trace file already exists
        events_data_file = Path(__file__).parent / 'events_data.json'
        
        if events_data_file.exists():
            print(f"Found existing trace file: {events_data_file}")
            print("Skipping log parsing. To re-parse, provide a directory as argument.")
        else:
            # No trace file exists, fall back to parsing example_dump
            dump_dir = Path(__file__).parent / 'example_dump'
            
            if not dump_dir.exists():
                print(f"Error: Directory {dump_dir} does not exist", file=sys.stderr)
                print(f"Error: No existing trace file found and example_dump directory not found", file=sys.stderr)
                sys.exit(1)
            
            # Parse logs
            if not parse_logs(dump_dir):
                sys.exit(1)
    
    # Check if visualization file exists
    viz_file = Path(__file__).parent / 'visualize_timeline.html'
    if not viz_file.exists():
        print(f"Error: {viz_file} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Start server
    port = 8000
    if len(sys.argv) > 2:
        try:
            port = int(sys.argv[2])
        except ValueError:
            print(f"Invalid port number: {sys.argv[2]}", file=sys.stderr)
            sys.exit(1)
    
    start_server(port)

if __name__ == '__main__':
    main()

