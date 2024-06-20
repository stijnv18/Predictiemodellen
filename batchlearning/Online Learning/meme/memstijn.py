import subprocess
import psutil
import time

# Define the Python file to be run
python_file = 'LSTMsolar.py'
print("Running the Python file: ", python_file)
# Start a subprocess to run the Python file
process = subprocess.Popen(['C:\\Users\\stijn\\anaconda3\\python.exe', python_file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print("Subprocess started with PID: ", process.pid)

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

    # Read from stdout and stderr
    for line in iter(process.stdout.readline, b''):
        print("Output: ", line.decode('utf-8'))

# Print the maximum memory usage in MB
print(f'Max memory usage: {max_mem_usage / (1024 * 1024)} MB')