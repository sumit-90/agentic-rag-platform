FROM python:3.11-slim

WORKDIR /app

# System dependencies required by PyMuPDF and lxml
RUN apt-get update && apt-get install -y --no-install-recommends     gcc     g++     libxml2-dev     libxslt1-dev     curl     && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
