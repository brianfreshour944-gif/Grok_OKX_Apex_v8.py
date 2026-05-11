FROM ghcr.io/ukewea/python-talib:ubuntu24.04-python3.12-20240915

WORKDIR /app

# Upgrade pip (optional but good)
RUN pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your bot code
COPY . .

# Start the bot
CMD ["python3", "Grok_OKX_Apex_v8.py"]
