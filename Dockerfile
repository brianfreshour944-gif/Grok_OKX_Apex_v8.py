# We use 3.10 because it's the most stable "sweet spot" for pandas_ta
FROM python:3.10-slim

WORKDIR /app

# Install basic system tools that pandas_ta sometimes needs to build
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# We install them one by one to make sure pandas_ta doesn't trip over the others
RUN pip install --no-cache-dir ccxt pandas numpy tqdm
RUN pip install --no-cache-dir pandas_ta

CMD ["python3", "Grok_OKX_Apex_v8.py"]
