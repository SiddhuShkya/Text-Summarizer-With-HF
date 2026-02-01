FROM python:3.10-slim-bullseye

# Install system dependencies
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    awscli \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the application port
EXPOSE 8000

# Command to run the application
CMD ["python3", "app.py"]
