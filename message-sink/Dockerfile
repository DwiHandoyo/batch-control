FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY metrics_collector.py .
COPY lqr_controller.py .
COPY sink.py .

# Create logs directory
RUN mkdir -p /app/logs

# Run the sink
CMD ["python", "-u", "sink.py"]
