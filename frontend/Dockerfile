FROM python:3.9-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


FROM python:3.9-slim
WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH

EXPOSE 8501

# Use the PORT environment variable if available, fallback to 8501
CMD ["sh", "-c", "streamlit run app_frontend.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]