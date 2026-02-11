#!/usr/bin/env python3
"""Analyze cProfile output and identify performance bottlenecks."""

import pstats
from pstats import SortKey
import sys

def analyze_profile(profile_file='prof.out'):
    """Load and analyze profiling data."""
    try:
        p = pstats.Stats(profile_file)
    except FileNotFoundError:
        print(f"Error: Profile file '{profile_file}' not found!")
        return
    
    print("=" * 100)
    print("PROFILING ANALYSIS REPORT")
    print("=" * 100)
    
    # Total time
    print(f"\nTotal execution time: {p.total_tt:.2f} seconds")
    print(f"Total function calls: {p.total_calls}")
    
    # Top functions by cumulative time
    print("\n" + "=" * 100)
    print("TOP 30 FUNCTIONS BY CUMULATIVE TIME (time including subcalls)")
    print("=" * 100)
    p.sort_stats(SortKey.CUMULATIVE).print_stats(30)
    
    # Top functions by self time
    print("\n" + "=" * 100)
    print("TOP 30 FUNCTIONS BY SELF TIME (time excluding subcalls)")
    print("=" * 100)
    p.sort_stats(SortKey.TIME).print_stats(30)
    
    # Get specific hotspots
    print("\n" + "=" * 100)
    print("IDENTIFIED HOTSPOTS")
    print("=" * 100)
    
    stats = p.stats
    # Convert to list and sort by cumulative time
    sorted_stats = sorted(stats.items(), key=lambda x: x[1][3], reverse=True)
    
    print("\nFunctions consuming > 1% of total time:")
    threshold = p.total_tt * 0.01
    for func, (cc, nc, tt, ct, callers) in sorted_stats[:50]:
        if ct > threshold:
            filename, line, func_name = func
            print(f"  {ct:8.2f}s ({ct/p.total_tt*100:5.1f}%) - {func_name} ({filename}:{line})")

if __name__ == '__main__':
    profile_file = sys.argv[1] if len(sys.argv) > 1 else 'prof.out'
    analyze_profile(profile_file)
