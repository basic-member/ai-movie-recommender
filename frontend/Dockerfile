FROM python:3.9-slim AS builder
WORKDIR /build
COPY frontend/requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


FROM python:3.9-slim
WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH

EXPOSE 8501

CMD ["streamlit", "run", "app_frontend.py", "--server.port=8501", "--server.address=0.0.0.0"] 