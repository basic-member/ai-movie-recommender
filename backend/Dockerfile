# Stage 1: Build
FROM python:3.9-slim AS builder
WORKDIR /build

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Final
FROM python:3.9-slim
WORKDIR /app

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy only the contents of the backend folder (assuming backend/ is the context)
COPY . .

EXPOSE 8000

# Use the PORT environment variable if available, fallback to 8000
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
