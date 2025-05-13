FROM python:3.12-slim

# Set workdir
WORKDIR /app

# copy the requirements file
COPY requirements.txt ./

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# copy the app
COPY app ./app

# Expose API port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
