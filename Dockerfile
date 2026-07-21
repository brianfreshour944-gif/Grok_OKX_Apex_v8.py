FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Create a non-root user for security
RUN adduser --disabled-password --gecos '' botuser

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project code
COPY . .

# Give the bot user ownership of the app directory
RUN chown -R botuser:botuser /app
USER botuser

# Run your application
CMD ["python", "main_bot.py"]
