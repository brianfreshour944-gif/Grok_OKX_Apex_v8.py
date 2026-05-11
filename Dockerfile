FROM techtrader/python-ta-lib:latest

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your bot code
COPY . .

# Run the application
CMD ["python3", "Grok_OKX_Apex_v8.py"]
