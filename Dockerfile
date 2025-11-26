# Production Dockerfile for jpgovsummary
FROM python:3.12-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies required by docling and other packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    ca-certificates \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI
RUN install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian bullseye stable" \
    | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update \
    && apt-get install -y docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install --no-cache-dir poetry==1.8.3

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Configure poetry to not create virtual environment (we're in a container)
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-dev --no-interaction --no-ansi

# Copy application code
COPY src/ ./src/
COPY README.md LICENSE ./

# Install the application
RUN poetry install --no-dev --no-interaction --no-ansi

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
ENTRYPOINT ["jpgovsummary"]
CMD ["--help"]
