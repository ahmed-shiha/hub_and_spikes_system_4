
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .

# In a real system, you would configure pip to trust your private repository
# RUN pip config set global.index-url <YOUR_PRIVATE_PYPI_URL>
# RUN pip install --no-cache-dir -r requirements.txt

# For this simulation, we assume medins_ml_utils is available
RUN pip install --no-cache-dir -r requirements.txt

# Copy the deployment script and config file into the container
COPY ./deployment/serve.py .
COPY ./config.json .

# Make port 80 available to the world outside this container
EXPOSE 80

# Run serve.py when the container launches.
# The models will be loaded from the MLflow registry at startup.
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "80"]
