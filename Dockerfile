FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements_space.txt .
RUN pip install --no-cache-dir -r requirements_space.txt uvicorn[standard]

# Copy application code
COPY . .

# HF Spaces runs on port 7860
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
