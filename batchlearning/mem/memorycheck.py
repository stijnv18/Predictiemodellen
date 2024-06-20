import os
import subprocess
from memory_profiler import memory_usage

def main():
    scripts = ['prohet.py']
    max_ram_usage = {}

    for script in scripts:
        # Run the script and measure the memory usage
        mem_usage = memory_usage((subprocess.run, (['d:/bachelor/BACH/Scripts/python.exe', script],), {'check': True}))
        max_ram_usage[script] = max(mem_usage)

        # Save the max RAM usage to a .txt file
        with open('max_ram_usage.txt', 'a') as f:
            f.write(f'{script}: {max_ram_usage[script]} MiB\n')

if __name__ == '__main__':
    main()