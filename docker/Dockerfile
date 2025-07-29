FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3.9 install --no-cache-dir -r requirements.txt

COPY app ./app
COPY main.py table_map.pkl table_id_vectors .

ENV PYTHONPATH=/app
ENV CHROMA_DB_PATH=/app/chroma_data

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 