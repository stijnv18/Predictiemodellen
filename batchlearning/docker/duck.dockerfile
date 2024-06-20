# Use Python 3.10 Buster as a base image
FROM python:3.10-buster

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the conta/app
COPY . /app

# Install the psutil and darts libraries
RUN pip install psutil darts

# The command to run when the container starts
CMD ["python", "./your_script.py"]