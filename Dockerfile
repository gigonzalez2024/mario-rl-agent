FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /mario

# Install required Python packages
RUN pip install --no-cache-dir \
    gym==0.21.0 \
    gym-super-mario-bros==7.4.0 \
    nes-py==8.2.1 \
    stable-baselines3[extra] \
    opencv-python

# Copy your training script
COPY . .

CMD ["python", "train_mario.py"]
