FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY perception_stack/ perception_stack/

RUN pip install --no-cache-dir -e .

COPY configs/ configs/

CMD ["python", "-m", "perception_stack.cli"]
