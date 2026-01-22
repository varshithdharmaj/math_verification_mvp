# Use official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for OpenCV and Tesseract)
RUN apt-get update && apt-get install -y \
  tesseract-ocr \
  libgl1-mesa-glx \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements (if exists, else handle dynamically? For now we assume user has one or we create it)
# We will create a requirements.txt in the next step, so this command assumes it exists.
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY backend/ ./backend/

# Expose port
EXPOSE 8000

# Define environment variables
ENV PYTHONPATH=/app

# Run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
