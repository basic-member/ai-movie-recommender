# Stage 1: Build
FROM python:3.9-slim AS builder
WORKDIR /build
COPY backend/requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Final
FROM python:3.9-slim
WORKDIR /app

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY . .

COPY ./models_data ./models_data

EXPOSE 8000

CMD ["python", "main.py"]