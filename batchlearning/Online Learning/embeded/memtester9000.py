import subprocess
import psutil
import time

# Define the Python file to be run
python_file = 'TIDE.py'
print("Running the Python file: ", python_file)
# Start a subprocess to run the Python file
process = subprocess.Popen(['python', python_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print("Subprocess started with PID: ", process.pid)

# Get the process id
pid = process.pid

# Create a Process object
p = psutil.Process(pid)

# Initialize max memory usage as 0
max_mem_usage = 0

# Monitor the memory usage of the subprocess
while process.poll() is None:
    print("Monitoring memory usage...")
    print("PID: ", pid)
    print("Max memory usage: ", max_mem_usage)
    print("Max in gb: ", max_mem_usage / (1024 * 1024 * 1024))
    mem_info = p.memory_info()
    mem_usage = mem_info.rss  # resident set size: memory usage in bytes
    if mem_usage > max_mem_usage:
        max_mem_usage = mem_usage
    time.sleep(0.1)  # sleep for 100 ms

# Print the maximum memory usage in MB
print(f'Max memory usage: {max_mem_usage / (1024 * 1024)} MB')