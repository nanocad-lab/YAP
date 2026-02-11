import pstats
import sys

p = pstats.Stats('prof.out')
p.strip_dirs()
print("=" * 100)
print("TOP 30 BY CUMULATIVE TIME")
print("=" * 100)
p.sort_stats('cumulative').print_stats(30)

print("\n" + "=" * 100)
print("TOP 30 BY SELF TIME")  
print("=" * 100)
p.sort_stats('time').print_stats(30)

# Show callers for expensive functions
print("\n" + "=" * 100)
print("CALLERS OF MOST EXPENSIVE FUNCTIONS")
print("=" * 100)
p.sort_stats('cumulative').print_callers(10)
