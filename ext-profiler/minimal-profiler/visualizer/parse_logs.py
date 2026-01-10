#!/usr/bin/env python3
"""
Parse minimal profiler log files and extract events into a structured format.
"""

import re
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, DefaultDict

def parse_timestamp(line: str) -> Optional[float]:
    """Extract timestamp from log line."""
    match = re.search(r'\[([\d.]+)\]', line)
    if match:
        return float(match.group(1))
    return None

def parse_event_address(line: str) -> Optional[str]:
    """Extract event address from log line."""
    match = re.search(r'event=0x([0-9a-fA-F]+)', line)
    if match:
        return '0x' + match.group(1)
    return None

def parse_parent_obj(line: str) -> Optional[str]:
    """Extract parent object address from log line."""
    match = re.search(r'parentObj=0x([0-9a-fA-F]+)', line)
    if match:
        return '0x' + match.group(1)
    # Also check for (nil) or (null) which means no parent
    if re.search(r'parentObj=\(nil\)|parentObj=\(null\)|parentObj=0x0', line):
        return None
    return None

def parse_context(line: str) -> Optional[str]:
    """Extract context address from log line."""
    match = re.search(r'ctx=0x([0-9a-fA-F]+)', line)
    if match:
        return '0x' + match.group(1)
    return None

def parse_rank_from_filename(filename: str) -> Optional[int]:
    """Extract rank number from filename."""
    match = re.search(r'rank(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def parse_commid_from_filename(filename: str) -> Optional[str]:
    """Extract communicator ID from filename.
    
    Filename format: minimal_profiler_rank{rank}_comm{commId}.log
    or: minimal_profiler_pxn_events_pid{pid}_comm{commId}.log
    Returns commId as string (it's a large uint64).
    """
    match = re.search(r'comm(\d+)', filename)
    if match:
        return match.group(1)
    return None

def parse_pid_from_pxn_filename(filename: str) -> Optional[int]:
    """Extract PID from PXN log filename.
    
    Filename format: minimal_profiler_pxn_events_pid{pid}_comm{commId}.log
    """
    match = re.search(r'pxn_events_pid(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def parse_origin_pid(line: str) -> Optional[int]:
    """Extract origin PID from PXN event line (from "PXN from pid=" field)."""
    match = re.search(r'PXN from pid=(\d+)', line)
    if match:
        return int(match.group(1))
    return None

def simplify_type_name(type_name: str) -> str:
    """Remove 'ncclProfile' prefix from event type name."""
    if type_name and type_name.startswith('ncclProfile'):
        return type_name[len('ncclProfile'):]
    return type_name if type_name else 'Unknown'

def parse_key_value_pairs(line: str) -> Dict:
    """Extract all key=value pairs from a log line."""
    details = {}
    # Match key=value patterns, handling various value types
    pattern = r'(\w+)=([^\s,)]+)'
    for match in re.finditer(pattern, line):
        key = match.group(1)
        value_str = match.group(2)
        # Try to convert to appropriate type
        try:
            if value_str.startswith('0x'):
                details[key] = value_str  # Keep hex as string
            elif '.' in value_str:
                details[key] = float(value_str)
            else:
                details[key] = int(value_str)
        except ValueError:
            details[key] = value_str  # Keep as string if conversion fails
    return details

def parse_tid(line: str) -> Optional[int]:
    """Extract thread ID from log line."""
    match = re.search(r'tid=(\d+)', line)
    if match:
        return int(match.group(1))
    return None

def parse_pid(line: str) -> Optional[int]:
    """Extract process ID from log line."""
    match = re.search(r'pid=(\d+)', line)
    if match:
        return int(match.group(1))
    return None

def parse_device_id(line: str) -> Optional[int]:
    """Extract physical GPU device ID from log line."""
    match = re.search(r'device=(\d+)', line)
    if match:
        return int(match.group(1))
    return None

def parse_start_event(line: str, require_ctx: bool = True) -> Optional[Dict]:
    """Parse a START event line.
    
    Args:
        line: The log line to parse
        require_ctx: If True, ctx is required (normal events). If False, ctx is optional (PXN events).
    """
    timestamp = parse_timestamp(line)
    event_addr = parse_event_address(line)
    ctx = parse_context(line)
    parent_obj = parse_parent_obj(line)
    tid = parse_tid(line)
    pid = parse_pid(line)
    device_id = parse_device_id(line)
    origin_pid = parse_origin_pid(line)  # For PXN events
    
    # Validate required fields
    if not timestamp or not event_addr:
        return None
    if require_ctx and not ctx:
        return None
    
    # Extract event descriptor type (e.g., "ncclProfileCollApi")
    match = re.search(r'START\s+(ncclProfile\w+)', line)
    if not match:
        return None
    
    event_type_name = match.group(1)
    
    # Extract all key-value pairs
    details = parse_key_value_pairs(line)
    
    # Extract function name if present
    func = details.pop('func', None)
    
    # Extract sequence number if present (for Coll events, not PXN)
    seq_number = details.pop('seq', None) if require_ctx else None
    
    # Remove tid, pid, and device from details since we store them separately
    details.pop('tid', None)
    details.pop('pid', None)
    details.pop('device', None)
    
    event_info = {
        'timestamp': timestamp,
        'event_addr': event_addr,
        'ctx': ctx,  # None for PXN events
        'parent_obj': parent_obj,
        'type': simplify_type_name(event_type_name),
        'type_name': event_type_name,
        'func': func,
        'tid': tid,
        'pid': pid,
        'details': details
    }
    
    # Add optional fields
    if seq_number is not None:
        event_info['seq_number'] = seq_number
    if device_id is not None:
        event_info['device_id'] = device_id
    if origin_pid is not None:
        event_info['origin_pid'] = origin_pid
    
    return event_info

def parse_stop_event(line: str, require_ctx: bool = True) -> Optional[Dict]:
    """Parse a STOP event line.
    
    Args:
        line: The log line to parse
        require_ctx: If True, ctx is required (normal events). If False, ctx is optional (PXN events).
    """
    timestamp = parse_timestamp(line)
    event_addr = parse_event_address(line)
    ctx = parse_context(line)
    tid = parse_tid(line)
    pid = parse_pid(line)
    device_id = parse_device_id(line)
    origin_pid = parse_origin_pid(line)  # For PXN events
    
    # Validate required fields
    if not timestamp or not event_addr:
        return None
    if require_ctx and not ctx:
        return None
    
    # Extract event descriptor type and function name from STOP line
    # Format: STOP ncclProfileType::funcName
    match = re.search(r'STOP\s+(ncclProfile\w+)(?:::(.+?))?\s*\(', line)
    type_name = None
    func = None
    if match:
        type_name = match.group(1)
        func = match.group(2) if match.group(2) else None
    
    # Extract all key-value pairs (including duration)
    details = parse_key_value_pairs(line)
    duration = details.pop('duration', None)
    
    # Remove tid, pid, and device from details since we store them separately
    details.pop('tid', None)
    details.pop('pid', None)
    details.pop('device', None)
    
    result = {
        'timestamp': timestamp,
        'event_addr': event_addr,
        'ctx': ctx,  # None for PXN events
        'type_name': type_name,
        'func': func,
        'duration': duration,
        'tid': tid,
        'pid': pid,
        'details': details
    }
    
    # Add optional fields
    if device_id is not None:
        result['device_id'] = device_id
    if origin_pid is not None:
        result['origin_pid'] = origin_pid
    
    return result

def parse_state_event(line: str, require_ctx: bool = True) -> Optional[Dict]:
    """Parse a STATE event line.
    
    Args:
        line: The log line to parse
        require_ctx: If True, ctx is required (normal events). If False, ctx is optional (PXN events).
    """
    timestamp = parse_timestamp(line)
    event_addr = parse_event_address(line)
    ctx = parse_context(line)
    tid = parse_tid(line)
    pid = parse_pid(line)
    device_id = parse_device_id(line)
    origin_pid = parse_origin_pid(line)  # For PXN events
    
    # Validate required fields
    if not timestamp or not event_addr:
        return None
    if require_ctx and not ctx:
        return None
    
    # Extract event descriptor type and function name
    # Format: STATE ncclProfileType::funcName -> StateName
    match = re.search(r'STATE\s+(ncclProfile\w+)(?:::(.+?))?\s+->\s+([^(]+)', line)
    type_name = None
    func = None
    state_name = None
    if match:
        type_name = match.group(1)
        func = match.group(2) if match.group(2) else None
        state_name = match.group(3).strip() if match.group(3) else None
    
    # Extract state ID
    match = re.search(r'stateId=(\d+)', line)
    state_id = int(match.group(1)) if match else None
    
    # Extract all key-value pairs (state-specific details)
    details = parse_key_value_pairs(line)
    
    # Remove tid, pid, and device from details since we store them separately
    details.pop('tid', None)
    details.pop('pid', None)
    details.pop('device', None)
    
    result = {
        'timestamp': timestamp,
        'event_addr': event_addr,
        'ctx': ctx,  # None for PXN events
        'type_name': type_name,
        'func': func,
        'state_name': state_name,
        'state_id': state_id,
        'tid': tid,
        'pid': pid,
        'details': details
    }
    
    # Add optional fields
    if device_id is not None:
        result['device_id'] = device_id
    if origin_pid is not None:
        result['origin_pid'] = origin_pid
    
    return result

def parse_log_header(filepath: Path) -> Dict:
    """Parse the header section of a log file to extract metadata.
    
    The header contains initialization info like:
    - Rank: 0/3, CommId: 12077415339453133131, CommName: (null), ctx=0x...
    - Start time, dump directory, event mask, etc.
    
    Returns a dict with: rank, nranks, commid, commname, ctx, start_time, device_id
    """
    header_info = {
        'rank': None,
        'nranks': None,
        'commid': None,
        'commname': None,
        'ctx': None,
        'start_time': None,
        'init_tid': None,
        'device_id': None,
    }
    
    with open(filepath, 'r') as f:
        for line in f:
            # Stop parsing header once we see event lines
            if line.startswith('[') and ('START' in line or 'STOP' in line or 'STATE' in line):
                break
            
            # Parse rank/nranks line: "Rank: 0/3, CommId: ..."
            rank_match = re.search(r'Rank:\s*(\d+)/(\d+)', line)
            if rank_match:
                header_info['rank'] = int(rank_match.group(1))
                header_info['nranks'] = int(rank_match.group(2))
            
            # Parse CommId from header
            commid_match = re.search(r'CommId:\s*(\d+)', line)
            if commid_match:
                header_info['commid'] = commid_match.group(1)
            
            # Parse CommName from header
            commname_match = re.search(r'CommName:\s*(\S+)', line)
            if commname_match:
                name = commname_match.group(1)
                if name != '(null)':
                    header_info['commname'] = name
            
            # Parse ctx from header
            ctx_match = re.search(r'ctx=0x([0-9a-fA-F]+)', line)
            if ctx_match:
                header_info['ctx'] = '0x' + ctx_match.group(1)
            
            # Parse start time
            start_time_match = re.search(r'Start time:\s*([\d.]+)', line)
            if start_time_match:
                header_info['start_time'] = float(start_time_match.group(1))
            
            # Parse init thread ID
            init_tid_match = re.search(r'Init thread ID \(tid\):\s*(\d+)', line)
            if init_tid_match:
                header_info['init_tid'] = int(init_tid_match.group(1))
            
            # Parse physical GPU device ID
            device_match = re.search(r'Physical GPU device ID:\s*(\d+)', line)
            if device_match:
                header_info['device_id'] = int(device_match.group(1))
    
    return header_info

def parse_log_footer(filepath: Path) -> Dict:
    """Parse the footer/finalize section of a log file.
    
    The footer contains finalization info like:
    [timestamp] === Profiler finalized (tid=..., totalDuration=... us = ... s, ctx=...) ===
    Final pool size: ...
    Final blacklist size: ...
    
    Returns a dict with: finalize_time, finalize_tid, total_duration, ctx
    """
    footer_info = {
        'finalize_time': None,
        'finalize_tid': None,
        'total_duration': None,
        'ctx': None,
    }
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    # Search from the end for finalize line
    for line in reversed(lines):
        # Parse finalize line: [timestamp] === Profiler finalized (tid=..., totalDuration=... us, ctx=...) ===
        finalize_match = re.search(r'\[([\d.]+)\]\s*===\s*Profiler finalized\s*\(tid=(\d+),\s*totalDuration=([\d.]+)\s*us.*ctx=0x([0-9a-fA-F]+)\)', line)
        if finalize_match:
            footer_info['finalize_time'] = float(finalize_match.group(1))
            footer_info['finalize_tid'] = int(finalize_match.group(2))
            footer_info['total_duration'] = float(finalize_match.group(3))
            footer_info['ctx'] = '0x' + finalize_match.group(4)
            break
    
    return footer_info

def _parse_events_from_file(filepath: Path, is_pxn: bool, rank: Optional[int], pid: Optional[int], 
                             commid: Optional[str], device_id: Optional[int] = None) -> Dict:
    """Unified function to parse events from a log file (both normal and PXN).
    
    Args:
        filepath: Path to the log file
        is_pxn: True if parsing PXN log file, False for normal log file
        rank: Rank number (None for PXN events initially)
        pid: Process ID (None for normal events, used for PXN events)
        commid: Communicator ID
        device_id: Device ID from header (None for PXN events)
    
    Returns:
        Dictionary of events keyed by unique_id
    """
    events = {}
    active_events_by_addr = defaultdict(list)
    
    def make_unique_id(event_addr: str, start_ts: float) -> str:
        """Create a unique identifier for an event."""
        if is_pxn:
            return f"{event_addr}_PXN_pid{pid}_comm{commid}_{start_ts:.3f}"
        else:
            return f"{event_addr}_rank{rank}_comm{commid}_{start_ts:.3f}"
    
    def find_active_event(event_addr: str, ctx: Optional[str], timestamp: float) -> Optional[str]:
        """Find the active event with this address that matches the timestamp."""
        # For PXN events, ctx is always None
        key = (event_addr, ctx if not is_pxn else None)
        if key not in active_events_by_addr:
            return None
        
        candidates = []
        for unique_id, start_ts in active_events_by_addr[key]:
            event = events.get(unique_id)
            if event and event['start'] is not None:
                if event['stop'] is None:
                    if start_ts <= timestamp:
                        candidates.append((unique_id, start_ts))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def create_event_from_start(start_info: Dict, event_addr: str, start_ts: float, ctx: Optional[str]) -> Dict:
        """Create an event dictionary from START event info."""
        event_device_id = None
        if not is_pxn:
            # Use device_id from event if available, otherwise from header
            event_device_id = start_info.get('device_id') if start_info.get('device_id') is not None else device_id
        
        event = {
            'start': start_ts,
            'stop': None,
            'states': [],
            'type': start_info['type'],
            'type_name': start_info.get('type_name'),
            'func': start_info.get('func'),
            'start_tid': start_info.get('tid'),
            'start_pid': start_info.get('pid'),
            'stop_tid': None,
            'stop_pid': None,
            'details': start_info.get('details', {}).copy(),
            'ctx': ctx if not is_pxn else None,
            'rank': rank,
            'commid': commid,
            'device_id': event_device_id,
            'event_addr': event_addr,
            'parent_obj': start_info.get('parent_obj')
        }
        
        # Add optional fields
        if not is_pxn and start_info.get('seq_number') is not None:
            event['seq_number'] = start_info.get('seq_number')
        if is_pxn and start_info.get('origin_pid') is not None:
            event['origin_pid'] = start_info.get('origin_pid')
        
        return event
    
    def create_event_from_stop(stop_info: Dict, event_addr: str, stop_ts: float, ctx: Optional[str]) -> Dict:
        """Create an event dictionary from STOP event info (when no matching START found)."""
        event_device_id = None
        if not is_pxn:
            event_device_id = stop_info.get('device_id') if stop_info.get('device_id') is not None else device_id
        
        event = {
            'start': None,
            'stop': stop_ts,
            'states': [],
            'type': simplify_type_name(stop_info.get('type_name', '')),
            'type_name': stop_info.get('type_name'),
            'func': stop_info.get('func'),
            'start_tid': None,
            'start_pid': None,
            'stop_tid': stop_info.get('tid'),
            'stop_pid': stop_info.get('pid'),
            'details': stop_info.get('details', {}),
            'ctx': ctx if not is_pxn else None,
            'rank': rank,
            'commid': commid,
            'device_id': event_device_id,
            'event_addr': event_addr
        }
        
        if is_pxn and stop_info.get('origin_pid') is not None:
            event['origin_pid'] = stop_info.get('origin_pid')
        
        return event
    
    def create_event_from_state(state_info: Dict, event_addr: str, state_ts: float, ctx: Optional[str]) -> Dict:
        """Create an event dictionary from STATE event info (when no matching START found)."""
        event_device_id = None
        if not is_pxn:
            event_device_id = state_info.get('device_id') if state_info.get('device_id') is not None else device_id
        
        event = {
            'start': None,
            'stop': None,
            'states': [{
                'timestamp': state_ts,
                'state_name': state_info['state_name'],
                'state_id': state_info['state_id'],
                'tid': state_info.get('tid'),
                'pid': state_info.get('pid'),
                'details': state_info['details']
            }],
            'type': simplify_type_name(state_info.get('type_name', '')),
            'type_name': state_info.get('type_name'),
            'func': state_info.get('func'),
            'start_tid': None,
            'start_pid': None,
            'stop_tid': None,
            'stop_pid': None,
            'details': {},
            'ctx': ctx if not is_pxn else None,
            'rank': rank,
            'commid': commid,
            'device_id': event_device_id,
            'event_addr': event_addr
        }
        
        if is_pxn and state_info.get('origin_pid') is not None:
            event['origin_pid'] = state_info.get('origin_pid')
        
        return event
    
    with open(filepath, 'r') as f:
        for line in f:
            # Parse START events
            start_condition = 'START' in line and ('PXN from pid=' in line if is_pxn else True)
            if start_condition:
                start_info = parse_start_event(line, require_ctx=not is_pxn)
                if start_info:
                    event_addr = start_info['event_addr']
                    start_ts = start_info['timestamp']
                    ctx = start_info['ctx']
                    
                    unique_id = make_unique_id(event_addr, start_ts)
                    events[unique_id] = create_event_from_start(start_info, event_addr, start_ts, ctx)
                    
                    # Track as active
                    active_events_by_addr[(event_addr, ctx if not is_pxn else None)].append((unique_id, start_ts))
            
            # Parse STOP events
            elif 'STOP' in line and 'STATE' not in line:
                stop_condition = 'PXN' in line if is_pxn else True
                if stop_condition:
                    stop_info = parse_stop_event(line, require_ctx=not is_pxn)
                    if stop_info:
                        event_addr = stop_info['event_addr']
                        stop_ts = stop_info['timestamp']
                        ctx = stop_info['ctx']
                        
                        unique_id = find_active_event(event_addr, ctx, stop_ts)
                        if unique_id:
                            events[unique_id]['stop'] = stop_ts
                            events[unique_id]['stop_tid'] = stop_info.get('tid')
                            events[unique_id]['stop_pid'] = stop_info.get('pid')
                            
                            # Update optional fields
                            if not is_pxn:
                                if events[unique_id].get('device_id') is None and stop_info.get('device_id') is not None:
                                    events[unique_id]['device_id'] = stop_info.get('device_id')
                            else:
                                if events[unique_id].get('origin_pid') is None:
                                    events[unique_id]['origin_pid'] = stop_info.get('origin_pid')
                            
                            if events[unique_id]['func'] is None:
                                events[unique_id]['func'] = stop_info.get('func')
                            if events[unique_id].get('type_name') is None:
                                events[unique_id]['type_name'] = stop_info.get('type_name')
                            if stop_info.get('details'):
                                events[unique_id]['details'].update(stop_info['details'])
                        else:
                            # No active event found - create a new one with just stop time
                            unique_id = make_unique_id(event_addr, stop_ts)
                            events[unique_id] = create_event_from_stop(stop_info, event_addr, stop_ts, ctx)
            
            # Parse STATE events
            elif 'STATE' in line:
                state_condition = 'PXN from pid=' in line if is_pxn else True
                if state_condition:
                    state_info = parse_state_event(line, require_ctx=not is_pxn)
                    if state_info:
                        event_addr = state_info['event_addr']
                        state_ts = state_info['timestamp']
                        ctx = state_info['ctx']
                        
                        unique_id = find_active_event(event_addr, ctx, state_ts)
                        if unique_id:
                            events[unique_id]['states'].append({
                                'timestamp': state_ts,
                                'state_name': state_info['state_name'],
                                'state_id': state_info['state_id'],
                                'tid': state_info.get('tid'),
                                'pid': state_info.get('pid'),
                                'details': state_info['details']
                            })
                            if events[unique_id]['func'] is None:
                                events[unique_id]['func'] = state_info.get('func')
                            if events[unique_id].get('type_name') is None:
                                events[unique_id]['type_name'] = state_info.get('type_name')
                            if is_pxn and events[unique_id].get('origin_pid') is None:
                                events[unique_id]['origin_pid'] = state_info.get('origin_pid')
                        else:
                            # No active event found - create a new one with just this state
                            unique_id = make_unique_id(event_addr, state_ts)
                            events[unique_id] = create_event_from_state(state_info, event_addr, state_ts, ctx)
                            active_events_by_addr[(event_addr, ctx if not is_pxn else None)].append((unique_id, state_ts))
    
    return events

def parse_log_file(filepath: Path) -> Tuple[int, str, Dict]:
    """Parse a single log file and return rank, commId, and events.
    
    Uses a unique ID combining event_addr, rank, commId, and start_timestamp to handle
    address reuse across different events and communicators.
    """
    # First parse header for reliable rank/commId info
    header = parse_log_header(filepath)
    
    # Use header info if available, fallback to filename
    rank = header.get('rank')
    if rank is None:
        rank = parse_rank_from_filename(filepath.name)
    if rank is None:
        print(f"Warning: Could not parse rank from {filepath.name}", file=sys.stderr)
    
    commid = header.get('commid')
    if commid is None:
        commid = parse_commid_from_filename(filepath.name)
    if commid is None:
        print(f"Warning: Could not parse commId from {filepath.name}", file=sys.stderr)
    
    # Get device ID from header (fallback to None if not found)
    device_id = header.get('device_id')
    
    # Parse events using unified function
    events = _parse_events_from_file(filepath, is_pxn=False, rank=rank, pid=None, 
                                     commid=commid, device_id=device_id)
    
    # Parse footer for finalize info
    footer = parse_log_footer(filepath)
    
    # Create a single combined ProfilerLifecycle event spanning from plugin init to finalize
    # This represents the profiler plugin's tracking period for this communicator on this rank
    # Note: plugin init and finalize are called for each rank for each communicator.
    # Plugin init is always called when comm is initialized on this rank, but plugin finalize
    # may be called when comm is finalized/destroyed OR just revoked (not fully destroyed)
    init_time = header.get('start_time')
    finalize_time = footer.get('finalize_time')
    
    if init_time is not None or finalize_time is not None:
        lifecycle_unique_id = f"PROFILER_LIFECYCLE_rank{rank}_comm{commid}"
        
        start_time = init_time
        stop_time = finalize_time
        
        events[lifecycle_unique_id] = {
            'start': start_time,
            'stop': stop_time,
            'states': [],
            'type': 'ProfilerLifecycle',
            'type_name': 'ProfilerLifecycle',
            'func': 'profilerInit→profilerFinalize',
            'seq_number': None,
            'start_tid': header.get('init_tid'),
            'stop_tid': footer.get('finalize_tid'),
            'details': {
                'nranks': header.get('nranks'),
                'commid': commid,
                'init_time': init_time,
                'finalize_time': finalize_time,
                'total_duration_us': footer.get('total_duration'),
            },
            'ctx': footer.get('ctx') or header.get('ctx'),
            'rank': rank,
            'commid': commid,
            'event_addr': 'PROFILER_LIFECYCLE',
            'parent_obj': None,
            'is_synthetic': True,  # Mark as synthetic event
        }
    
    return rank, commid, events


def parse_pxn_log_header(filepath: Path) -> Dict:
    """Parse the header section of a PXN log file to extract metadata."""
    header_info = {
        'commid': None,
        'start_time': None,
    }
    
    with open(filepath, 'r') as f:
        for line in f:
            # Stop parsing header once we see event lines
            if line.startswith('[') and ('START' in line or 'STOP' in line or 'STATE' in line):
                break
            
            # Parse CommId from header
            commid_match = re.search(r'CommId:\s*(\d+)', line)
            if commid_match:
                header_info['commid'] = commid_match.group(1)
            
            # Parse start time
            start_time_match = re.search(r'Start time:\s*([\d.]+)', line)
            if start_time_match:
                header_info['start_time'] = float(start_time_match.group(1))
    
    return header_info

def parse_pxn_log_file(filepath: Path) -> Tuple[Optional[int], Optional[str], Dict]:
    """Parse a PXN log file and return pid, commId, and events.
    
    PXN events don't have rank/ctx initially - they will be inferred later.
    """
    # Parse header for commId
    header = parse_pxn_log_header(filepath)
    
    # Extract PID and commId from filename
    pid = parse_pid_from_pxn_filename(filepath.name)
    commid = header.get('commid')
    if commid is None:
        commid = parse_commid_from_filename(filepath.name)
    if commid is None:
        print(f"Warning: Could not parse commId from {filepath.name}", file=sys.stderr)
    
    # Parse events using unified function
    events = _parse_events_from_file(filepath, is_pxn=True, rank=None, pid=pid, 
                                     commid=commid, device_id=None)
    
    return pid, commid, events

def infer_rank_ctx_from_commid_pid(commid: str, pid: int, all_events: Dict) -> Tuple[Optional[int], Optional[str], List]:
    """Infer rank and ctx from (commId, pid) by searching through regular events.
    
    Returns: (rank, ctx, list_of_all_matches) where rank/ctx is the most common match.
    If multiple unique matches found, returns all of them in the list.
    """
    matches = []  # List of (rank, ctx) tuples found
    
    for unique_id, event_data in all_events.items():
        event_commid = event_data.get('commid')
        if event_commid != commid:
            continue
        
        # Check if this event's PID matches pid
        start_pid = event_data.get('start_pid')
        stop_pid = event_data.get('stop_pid')
        
        if (start_pid is not None and start_pid == pid) or \
           (stop_pid is not None and stop_pid == pid):
            rank = event_data.get('rank')
            ctx = event_data.get('ctx')
            if rank is not None and ctx is not None:
                matches.append((rank, ctx))
    
    if not matches:
        return (None, None, [])
    
    # Count occurrences of each (rank, ctx) pair
    match_counts = Counter(matches)
    
    # Get the most common match
    most_common = match_counts.most_common(1)[0]
    rank, ctx = most_common[0]
    
    # Get all unique matches for logging
    unique_matches = list(set(matches))
    
    return (rank, ctx, unique_matches)

def parse_all_logs(dump_dir: Path) -> Dict:
    """Parse all log files in the dump directory."""
    regular_log_files = sorted(dump_dir.glob('minimal_profiler_rank*.log'))
    pxn_log_files = sorted(dump_dir.glob('minimal_profiler_pxn_events_pid*.log'))
    
    if not regular_log_files and not pxn_log_files:
        print(f"Error: No log files found in {dump_dir}", file=sys.stderr)
        return {}
    
    all_events = {}
    ranks = set()
    contexts = set()
    threads = set()  # Track unique thread IDs
    pids = set()  # Track unique process IDs
    commids = set()  # Track unique communicator IDs
    device_ids = set()  # Track unique GPU device IDs
    
    # Build parent-child relationship map
    # Maps parent_obj address -> list of child event unique_ids
    parent_child_map = {}
    # Maps event unique_id -> parent event unique_id
    event_to_parent = {}
    
    # First, parse all regular log files
    for log_file in regular_log_files:
        rank, commid, events = parse_log_file(log_file)
        
        # Collect rank and commid from parsed files
        if rank is not None and rank >= 0:
            ranks.add(rank)
        if commid is not None and commid != "unknown":
            commids.add(commid)
        
        for unique_id, event_data in events.items():
            # Use the unique_id as the key (already includes rank and timestamp)
            # Timestamps are kept as absolute values (not normalized)
            all_events[unique_id] = event_data
            if event_data['ctx']:
                contexts.add(event_data['ctx'])
            
            # Also collect rank and commid from event data (in case filename didn't have it)
            if event_data.get('rank') is not None:
                ranks.add(event_data['rank'])
            if event_data.get('commid') and event_data['commid'] != "unknown":
                commids.add(event_data['commid'])
            
            # Collect thread IDs
            if event_data.get('start_tid') is not None:
                threads.add(event_data['start_tid'])
            if event_data.get('stop_tid') is not None:
                threads.add(event_data['stop_tid'])
            for state in event_data.get('states', []):
                if state.get('tid') is not None:
                    threads.add(state['tid'])
            
            # Collect process IDs
            if event_data.get('start_pid') is not None:
                pids.add(event_data['start_pid'])
            if event_data.get('stop_pid') is not None:
                pids.add(event_data['stop_pid'])
            for state in event_data.get('states', []):
                if state.get('pid') is not None:
                    pids.add(state['pid'])
            
            # Collect device IDs
            if event_data.get('device_id') is not None:
                device_ids.add(event_data['device_id'])
    
    pxn_total_events = 0
    pxn_events_inferred = 0
    pxn_events_not_found = 0
    for pxn_log_file in pxn_log_files:
        pid, commid, pxn_events = parse_pxn_log_file(pxn_log_file)
        
        if commid is not None and commid != "unknown":
            commids.add(commid)
        
        for unique_id, event_data in pxn_events.items():
            pxn_total_events += 1
            # Infer rank and ctx from (commId, pid)
            # Use pid from the PXN event (start_pid or stop_pid) to match against regular events
            pid = event_data.get('start_pid') or event_data.get('stop_pid')
            if pid is not None and commid is not None:
                rank, ctx, all_matches = infer_rank_ctx_from_commid_pid(commid, pid, all_events)
                
                if len(all_matches) > 1:
                    # Multiple matches found - print warning
                    print(f"Warning: Multiple (rank, ctx) matches for PXN event (commId={commid}, pid={pid}): {all_matches}", file=sys.stderr)
                
                if rank is not None and ctx is not None:
                    event_data['rank'] = rank
                    event_data['ctx'] = ctx
                    pxn_events_inferred += 1
                    ranks.add(rank)
                    contexts.add(ctx)
                else:
                    pxn_events_not_found += 1
            
            # Add PXN event to all_events
            all_events[unique_id] = event_data
            
            # Collect commid from event data
            if event_data.get('commid') and event_data['commid'] != "unknown":
                commids.add(event_data['commid'])
            
            # Collect thread IDs
            if event_data.get('start_tid') is not None:
                threads.add(event_data['start_tid'])
            if event_data.get('stop_tid') is not None:
                threads.add(event_data['stop_tid'])
            for state in event_data.get('states', []):
                if state.get('tid') is not None:
                    threads.add(state['tid'])
            
            # Collect process IDs
            if event_data.get('start_pid') is not None:
                pids.add(event_data['start_pid'])
            if event_data.get('stop_pid') is not None:
                pids.add(event_data['stop_pid'])
            if event_data.get('origin_pid') is not None:
                pids.add(event_data['origin_pid'])
            for state in event_data.get('states', []):
                if state.get('pid') is not None:
                    pids.add(state['pid'])
    
    if pxn_events_inferred > 0 or pxn_events_not_found > 0:
        print(f"PXN events: {pxn_events_inferred} inferred rank/ctx, {pxn_events_not_found} not found", file=sys.stderr)
    
    # Build parent-child relationships after all events are parsed
    # Important: Parent and child events are ALWAYS on the same rank (memory addresses are node-local)
    # Important: Addresses can be reused, so we need to find the most recent parent
    for unique_id, event_data in all_events.items():
        parent_obj = event_data.get('parent_obj')
        if parent_obj:
            # Find the parent event by matching parent_obj to event_addr
            # We need to find the most recent parent that started before (or at) the child's start time
            child_start = event_data.get('start')
            child_rank = event_data.get('rank')
            child_ctx = event_data.get('ctx')
            
            if child_start is None:
                # If child has no start time, use the first state timestamp or skip
                if event_data.get('states'):
                    child_start = min(state.get('timestamp') for state in event_data['states'])
                else:
                    child_start = None
            
            parent_found = None
            candidate_parents = []
            
            # Find all candidate parent events with matching address
            # NOTE: parentObj addresses can reference events from different ranks
            # (NCCL may use shared memory or cross-rank event references, especially on same node)
            for other_id, other_event in all_events.items():
                if other_event.get('event_addr') == parent_obj:
                    candidate_parents.append((other_id, other_event))
            
            if candidate_parents:
                # Filter to only parents that started before (or at) the child's start time
                valid_parents = []
                for other_id, other_event in candidate_parents:
                    parent_start = other_event.get('start')
                    
                    if parent_start is None:
                        continue  # Parent must have a start time
                    
                    # If child has a start time, parent must have started before or at the same time
                    if child_start is not None and parent_start > child_start:
                        continue
                    
                    # This parent is a valid candidate
                    valid_parents.append((other_id, other_event, parent_start))
                
                if valid_parents:
                    # Among valid parents, pick the one that started most recently (highest start time)
                    valid_parents.sort(key=lambda x: x[2], reverse=True)
                    parent_found = valid_parents[0][0]
            
            if parent_found:
                event_to_parent[unique_id] = parent_found
                if parent_found not in parent_child_map:
                    parent_child_map[parent_found] = []
                parent_child_map[parent_found].append(unique_id)
    
    return {
        'events': all_events,
        'ranks': sorted(ranks),
        'contexts': sorted(contexts),
        'threads': sorted(threads),  # Unique thread IDs found in the logs
        'pids': sorted(pids),  # Unique process IDs found in the logs
        'device_ids': sorted(device_ids),  # Unique GPU device IDs found in the logs
        'commids': sorted(commids),  # Unique communicator IDs found in the logs
        'total_events': len(all_events),
        'pxn_events_inferred': pxn_events_inferred,
        'pxn_events_not_found': pxn_events_not_found,
        'parent_child_map': parent_child_map,  # parent_id -> [child_ids]
        'event_to_parent': event_to_parent      # child_id -> parent_id
    }

def add_parent_child_info_to_trace_event(trace_event: Dict, unique_id: str, data: Dict) -> None:
    """Add parent-child relationship information to a Chrome tracing event.
    
    Args:
        trace_event: The trace event dictionary to modify (adds to trace_event['args'])
        unique_id: The unique ID of the event
        data: The parsed events data containing parent_child_map and event_to_parent
    """
    # Add parent event information
    if data.get('event_to_parent') and data['event_to_parent'].get(unique_id):
        parent_id = data['event_to_parent'][unique_id]
        parent_event = data['events'].get(parent_id)
        if parent_event:
            trace_event["args"]["parent_event"] = {
                "type": parent_event.get('type'),
                "type_name": parent_event.get('type_name'),
                "func": parent_event.get('func'),
                "rank": parent_event.get('rank')
            }
    
    # Add child events information
    if data.get('parent_child_map') and data['parent_child_map'].get(unique_id):
        children = data['parent_child_map'][unique_id]
        trace_event["args"]["child_events"] = []
        for child_id in children:
            child_event = data['events'].get(child_id)
            if child_event:
                trace_event["args"]["child_events"].append({
                    "type": child_event.get('type'),
                    "type_name": child_event.get('type_name'),
                    "func": child_event.get('func'),
                    "rank": child_event.get('rank')
                })

def convert_to_chrome_tracing(data: Dict) -> List[Dict]:
    """Convert events data to Chrome Tracing Event Format.
    
    Chrome Tracing Format: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/edit
    
    Returns a list of trace events that can be loaded in chrome://tracing or Perfetto.
    """
    trace_events = []
    
    for unique_id, event in data.get('events', {}).items():
        rank = event.get('rank')
        ctx = event.get('ctx')
        # For PXN events without inferred rank, use a special value or originPid
        if rank is None and event.get('origin_pid') is not None:
            # Use originPid as a fallback for visualization (will show as negative rank)
            tid = -abs(event.get('origin_pid')) % 1000  # Keep it reasonable for visualization
        else:
            tid = rank if rank is not None else 0
        
        event_type = event.get('type')
        func = event.get('func')
        type_name = event.get('type_name') or event_type
        
        category = type_name or event_type
        
        # Create event name: only include function name if it adds information
        # If func is the same as simplified type, it's redundant - just use func
        # If func is different (like AllReduce for CollApi), use just func for clarity
        # If no func, use simplified type name
        simplified_type = simplify_type_name(type_name) if type_name else event_type
        if func and func != simplified_type and func != 'Unknown':
            # Function name adds information (e.g., "AllReduce" for CollApi)
            event_name = func
        elif func and func == simplified_type:
            # Function name is redundant with type (e.g., "ProxyCtrl" for ProxyCtrl)
            event_name = simplified_type
        else:
            # No function name or unknown, use simplified type
            event_name = simplified_type
        
        # Add complete events (with both start and stop)
        if event.get('start') is not None and event.get('stop') is not None:
            start_ts = int(event['start'])
            duration = int(event['stop'] - event['start'])
            
            trace_event = {
                "name": event_name,
                "cat": category,
                "ph": "X",
                "ts": start_ts,
                "dur": duration,
                "pid": rank if rank is not None else 0,
                "tid": tid,
                "args": {
                    "event_addr": event.get('event_addr', ''),
                    "ctx": ctx,
                    "type": event_type,
                    "type_name": type_name,
                    "commid": event.get('commid', ''),
                }
            }
            
            # Add thread IDs if present
            if event.get('start_tid') is not None:
                trace_event["args"]["start_tid"] = event.get('start_tid')
            if event.get('stop_tid') is not None:
                trace_event["args"]["stop_tid"] = event.get('stop_tid')
            
            # Add process IDs if present
            if event.get('start_pid') is not None:
                trace_event["args"]["start_pid"] = event.get('start_pid')
            if event.get('stop_pid') is not None:
                trace_event["args"]["stop_pid"] = event.get('stop_pid')
            
            # Add sequence number if present
            if event.get('seq_number') is not None:
                trace_event["args"]["seq_number"] = event.get('seq_number')
            
            # Add originPid for PXN events
            if event.get('origin_pid') is not None:
                trace_event["args"]["origin_pid"] = event.get('origin_pid')
            
            # Add parent-child relationship information
            add_parent_child_info_to_trace_event(trace_event, unique_id, data)
            
            # Add all event details to args
            if event.get('details'):
                trace_event["args"].update(event['details'])
            
            trace_events.append(trace_event)
        
        # Add start-only events (begin marker)
        elif event.get('start') is not None:
            start_ts = int(event['start'])
            trace_event = {
                "name": event_name,
                "cat": category,
                "ph": "B",
                "ts": start_ts,
                "pid": rank if rank is not None else 0,
                "tid": tid,
                "args": {
                    "event_addr": event.get('event_addr', ''),
                    "ctx": ctx,
                    "type": event_type,
                    "type_name": type_name,
                }
            }
            
            # Add thread ID if present
            if event.get('start_tid') is not None:
                trace_event["args"]["start_tid"] = event.get('start_tid')
            
            # Add process ID if present
            if event.get('start_pid') is not None:
                trace_event["args"]["start_pid"] = event.get('start_pid')
            
            # Add sequence number if present
            if event.get('seq_number') is not None:
                trace_event["args"]["seq_number"] = event.get('seq_number')
            
            # Add originPid for PXN events
            if event.get('origin_pid') is not None:
                trace_event["args"]["origin_pid"] = event.get('origin_pid')
            
            # Add parent-child relationship information
            add_parent_child_info_to_trace_event(trace_event, unique_id, data)
            
            trace_events.append(trace_event)
        
        # Add stop-only events (end marker)
        elif event.get('stop') is not None:
            stop_ts = int(event['stop'])
            trace_event = {
                "name": event_name,
                "cat": category,
                "ph": "E",
                "ts": stop_ts,
                "pid": rank if rank is not None else 0,
                "tid": tid,
                "args": {
                    "event_addr": event.get('event_addr', ''),
                    "ctx": ctx,
                    "type": event_type,
                    "type_name": type_name,
                }
            }
            
            # Add thread ID if present
            if event.get('stop_tid') is not None:
                trace_event["args"]["stop_tid"] = event.get('stop_tid')
            
            # Add process ID if present
            if event.get('stop_pid') is not None:
                trace_event["args"]["stop_pid"] = event.get('stop_pid')
            
            # Add sequence number if present
            if event.get('seq_number') is not None:
                trace_event["args"]["seq_number"] = event.get('seq_number')
            
            # Add originPid for PXN events
            if event.get('origin_pid') is not None:
                trace_event["args"]["origin_pid"] = event.get('origin_pid')
            
            # Add parent-child relationship information
            add_parent_child_info_to_trace_event(trace_event, unique_id, data)
            
            trace_events.append(trace_event)
        
        # Add state transitions as instant events
        for state in event.get('states', []):
            state_ts = int(state['timestamp'])
            state_name = state.get('state_name')
            
            # For state transitions, show state name (event name is already in category/context)
            state_trace_event = {
                "name": state_name,
                "cat": category,
                "ph": "i",
                "ts": state_ts,
                "pid": rank if rank is not None else 0,
                "tid": tid,
                "s": "t",  # Scope: thread
                "args": {
                    "event_name": event_name,  # Include event name in args for reference
                    "state_name": state_name,
                    "state_id": state.get('state_id'),
                    "event_addr": event.get('event_addr', ''),
                    "ctx": ctx,
                }
            }
            
            # Add thread ID if present
            if state.get('tid') is not None:
                state_trace_event["args"]["tid"] = state.get('tid')
            
            # Add process ID if present
            if state.get('pid') is not None:
                state_trace_event["args"]["pid"] = state.get('pid')
            
            # Add state-specific details
            if state.get('details'):
                state_trace_event["args"].update(state['details'])
            
            trace_events.append(state_trace_event)
    
    # Sort by timestamp for better visualization
    trace_events.sort(key=lambda x: x['ts'])
    
    return trace_events

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        dump_dir = Path(sys.argv[1])
    else:
        # Default to example_dump directory
        dump_dir = Path(__file__).parent / 'example_dump'
    
    if not dump_dir.exists():
        print(f"Error: Directory {dump_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    print(f"Parsing log files from {dump_dir}...", file=sys.stderr)
    data = parse_all_logs(dump_dir)
    
    # Write standard events_data.json
    output_file = dump_dir.parent / 'events_data.json'
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Write Chrome tracing format
    chrome_trace_file = dump_dir.parent / 'chrome_trace.json'
    trace_events = convert_to_chrome_tracing(data)
    with open(chrome_trace_file, 'w') as f:
        json.dump(trace_events, f, indent=2)
    
    print(f"Parsed {data['total_events']} events", file=sys.stderr)
    print(f"Found {len(data['ranks'])} ranks: {data['ranks']}", file=sys.stderr)
    print(f"Found {len(data['contexts'])} contexts", file=sys.stderr)
    print(f"Found {len(data['threads'])} unique thread IDs: {data['threads']}", file=sys.stderr)
    print(f"Found {len(data['pids'])} unique process IDs: {data['pids']}", file=sys.stderr)
    print(f"Found {len(data['device_ids'])} unique GPU device IDs: {data['device_ids']}", file=sys.stderr)
    print(f"Found {len(data['commids'])} communicators: {data['commids']}", file=sys.stderr)
    if 'pxn_events_inferred' in data or 'pxn_events_not_found' in data:
        inferred = data.get('pxn_events_inferred', 0)
        not_found = data.get('pxn_events_not_found', 0)
        if inferred > 0 or not_found > 0:
            print(f"PXN events: {inferred} with inferred rank/ctx, {not_found} without match", file=sys.stderr)
    print(f"Output written to {output_file}", file=sys.stderr)
    print(f"Chrome trace written to {chrome_trace_file}", file=sys.stderr)
    print(f"  → Load in Chrome: chrome://tracing (drag & drop the file)", file=sys.stderr)
    print(f"  → Or use Perfetto: https://ui.perfetto.dev/", file=sys.stderr)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

