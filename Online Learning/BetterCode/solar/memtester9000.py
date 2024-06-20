import subprocess
import psutil
import time

# Define the Python file to be run
python_file = 'TraininNN.py'

# Start a subprocess to run the Python file
process = subprocess.Popen(['python', python_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Get the process id
pid = process.pid

# Create a Process object
p = psutil.Process(pid)

# Initialize max memory usage as 0
max_mem_usage = 0

# Monitor the memory usage of the subprocess
while process.poll() is None:
    mem_info = p.memory_info()
    mem_usage = mem_info.rss  # resident set size: memory usage in bytes
    if mem_usage > max_mem_usage:
        max_mem_usage = mem_usage
    time.sleep(0.1)  # sleep for 100 ms

# Print the maximum memory usage in MB
print(f'Max memory usage: {max_mem_usage / (1024 * 1024)} MB')