FROM python:3.11-slim

# System deps for mujoco/glfw offscreen and pybullet
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev libgl1 libglu1-mesa xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Default command runs full suite headlessly
ENTRYPOINT ["python", "run_all.py"]