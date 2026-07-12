# NeoGMA — pose-based automated GMA. GPU by default.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Analysis stack (no torch here — torch is pinned last, see below).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# PyTorch LAST and PINNED.
#
# Two hard-won lessons from the Nmotion build:
#  * CUDA 12.8 wheels are REQUIRED for Blackwell (RTX 50-series, sm_120).
#    The cu121 line silently fails on that hardware.
#  * An unpinned `torch>=x` resolve gets clobbered back to a CPU wheel by a
#    later dependency resolution, and the failure is silent — you only find out
#    when inference runs 20x slow. Pin it, and install it last.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG TORCH_SPEC=torch==2.11.0 torchvision==0.26.0
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX} ${TORCH_SPEC}

# MMPose stack for ViTPose-H. Installed after torch, which mim requires.
RUN pip install --no-cache-dir openmim \
    && mim install "mmengine>=0.10" "mmcv>=2.1.0,<2.3.0" "mmdet>=3.2.0" \
    && pip install --no-cache-dir "mmpose>=1.3.0"

COPY pipeline ./pipeline
COPY webapp ./webapp

ENV NEOGMA_TARGET_FPS=30 \
    NEOGMA_WINDOW_SECONDS=5.0 \
    NEOGMA_OVERLAP=0.5 \
    NEOGMA_DATA_DIR=/app/webapp/data_runtime \
    TORCH_HOME=/app/.torch \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]
