FROM python:3.10-slim

WORKDIR /app

# Copy only requirements first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy ONLY the necessary Python files
COPY Grok_OKX_Apex_v8.py .
# Copy any other specific .py files your bot needs
# COPY *.py .

# Optional: If you need config files
# COPY config.json .

CMD ["python3", "Grok_OKX_Apex_v8.py"]
