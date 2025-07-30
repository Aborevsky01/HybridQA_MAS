FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3.9 install --no-cache-dir -r requirements.txt

COPY app ./app
COPY main.py .

COPY table_map.pkl* table_id_vectors* ./
RUN ls table_map.pkl 2>/dev/null || echo "table_map.pkl not found" && \
    ls table_id_vectors 2>/dev/null || echo "table_id_vectors not found"

ENV PYTHONPATH=/app
ENV CHROMA_DB_PATH=/app/chroma_data

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 