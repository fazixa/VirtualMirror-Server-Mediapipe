# Use the specified base image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone https://github.com/fazixa/VirtualMirror-Server-Mediapipe.git /app

# Install Python requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose default port (optional, depends on your app's requirements)
EXPOSE 5000

# Set environment variables for the script (can be overridden at runtime)
ENV HOST=0.0.0.0
ENV PORT=5000
ENV DEBUG=false
ENV CAMERA_INDEX=0

# Default command to run the application, reading environment variables
CMD ["sh", "-c", "python3 app.py --host $HOST --port $PORT --debug $DEBUG --camera-index $CAMERA_INDEX"]
