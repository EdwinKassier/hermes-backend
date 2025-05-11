# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory to /app
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install -r requirements.txt

# Explicit env copy
COPY .env .env

# Copy the rest of the application code to the working directory
COPY . .

EXPOSE 8080

# Define the command to run your Flask application
CMD ["python", "run.py"]
