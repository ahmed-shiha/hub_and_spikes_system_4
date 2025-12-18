
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# This assumes medins_ml_utils is available in a package repository
RUN pip install --no-cache-dir -r requirements.txt

# Copy the deployment script into the container
COPY ./deployment/serve.py .

# Make port 80 available to the world outside this container
EXPOSE 80

# Run serve.py when the container launches.
# The model will be loaded from the MLflow registry at startup.
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "80"]
