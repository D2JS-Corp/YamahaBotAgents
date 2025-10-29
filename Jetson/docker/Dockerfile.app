FROM dustynv/l4t-pytorch:r36.2.0

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential cmake git curl wget pkg-config \
    ffmpeg libavformat-dev libavcodec-dev libavutil-dev libswresample-dev libswscale-dev \
    libopus-dev libvpx-dev \
    python3-dev python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# ---------- FAISS-GPU compilation ----------
WORKDIR /opt/src
RUN git clone --depth=1 https://github.com/facebookresearch/faiss.git
WORKDIR /opt/src/faiss
ARG CUDA_ARCH="72"
RUN cmake -B build \
  -DFAISS_ENABLE_GPU=ON \
  -DFAISS_ENABLE_PYTHON=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
  -DBUILD_TESTING=OFF
RUN cmake --build build -j$(nproc)
RUN cmake --install build
# Binding Python
WORKDIR /opt/src/faiss/build/faiss/python
RUN pip3 install .
# ---------- fin FAISS-GPU ----------

# Requisitos Python de tu app
WORKDIR /workspace
COPY Jetson/requirements.txt /workspace/requirements.txt
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Variables útiles para CUDA/torch en Jetson
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    OMP_NUM_THREADS=1

# Copia la app
COPY Jetson/app /workspace/app
WORKDIR /workspace/app

# Puerto FastAPI
EXPOSE 8000

# Arranque por defecto de tu servidor (puedes ajustar a uvicorn/gunicorn)
CMD ["python3", "server.py"]
