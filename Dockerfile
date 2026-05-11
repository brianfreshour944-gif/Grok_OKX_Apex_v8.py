
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir ccxt pandas numpy pandas_ta tqdm
CMD ["python3", "Grok_OKX_Apex_v8.py"]
