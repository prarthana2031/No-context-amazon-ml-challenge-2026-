# Amazon ML Challenge 2026 - Business Entity Resolution
# Reproducible environment for the whole team

FROM python:3.11-slim-bookworm

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies (needed for some ML packages & rapidfuzz)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (good practice)
RUN useradd -m -u 1000 eruser
WORKDIR /workspace

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Switch to non-root user
USER eruser

# Default command
CMD ["bash"]