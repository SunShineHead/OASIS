# -------------------------------------------------------
# Base image: NVIDIA CUDA 11.8 + cuDNN8 (Ubuntu 22.04)
# -------------------------------------------------------
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# -------------------------------------------------------
# System dependencies + Python 3.11
# -------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-distutils python3-pip \
        git wget curl build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Make Python 3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN pip install --upgrade pip

# -------------------------------------------------------
# PyTorch with CUDA 11.8
# -------------------------------------------------------
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# -------------------------------------------------------
# Core data-science libraries
# -------------------------------------------------------
RUN pip install \
        numpy \
        pandas \
        scipy \
        scikit-learn \
        statsmodels \
        polars \
        matplotlib \
        seaborn \
        plotly \
        altair

# -------------------------------------------------------
# Deep learning + AI ecosystem
# -------------------------------------------------------
RUN pip install \
        transformers \
        datasets \
        accelerate \
        tensorboard \
        lightning

# -------------------------------------------------------
# Productivity + notebooks
# -------------------------------------------------------
RUN pip install \
        jupyterlab \
        ipykernel \
        tqdm \
        joblib \
        python-dotenv

# -------------------------------------------------------
# Default workspace
# -------------------------------------------------------
WORKDIR /workspace

# Expose JupyterLab port
EXPOSE 8888

# Start container in bash by default
CMD ["/bin/bash"]