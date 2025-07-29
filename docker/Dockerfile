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

# Use shell commands to check and copy files
RUN if [ -f ../table_map.pkl ]; then cp ../table_map.pkl .; fi && \
    if [ -f ../table_id_vectors ]; then cp ../table_id_vectors .; fi

ENV PYTHONPATH=/app
ENV CHROMA_DB_PATH=/app/chroma_data

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 