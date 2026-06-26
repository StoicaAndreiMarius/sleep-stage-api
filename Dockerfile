FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install PyTorch as a CPU-only build (no bundled CUDA libraries) so the image stays small.
# torch is deliberately not in requirements.txt, so this is the only place it is installed.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]