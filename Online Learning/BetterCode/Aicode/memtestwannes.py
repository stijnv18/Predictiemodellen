import subprocess
import time
import psutil

# Start the script as a subprocess
proc = subprocess.Popen(['C:\\Users\\Stijn\\AppData\\Local\\Programs\\Python\\Python310\\python.exe', 'prophetmem.py'])

# Create a Process object for the subprocess
p = psutil.Process(proc.pid)

# Monitor the memory usage of the subprocess
max_mem_usage = 0
while p.is_running():
    mem_info = p.memory_info()
    mem_usage = mem_info.rss  # resident set size: memory usage in bytes
    if mem_usage > max_mem_usage:
        max_mem_usage = mem_usage
    time.sleep(0.1)  # sleep for 100 ms

# Print the maximum memory usage in MB
print(f'Max memory usage: {max_mem_usage / (1024 * 1024)} MB')