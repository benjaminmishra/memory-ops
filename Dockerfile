FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by bitsandbytes and accelerate
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc git && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency specification
COPY requirements.txt ./

# Install Python dependencies.  \
# Note: torch+bitsandbytes wheels are installed from PyPI.  
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app ./app
COPY README.md ./

# Expose port and set default command
ENV PORT 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]