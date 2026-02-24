 
# python package with in conda

.github/workflows/ci.yml
 

 
name: Train, Test, and Upload Artifacts

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

jobs:
  train-test:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.10"

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest

    - name: Train model
      run: python src/train_pipeline.py

    - name: Run tests
      run: pytest -v

    - name: Upload trained model artifact
      uses: actions/upload-artifact@v3
      with:
        name: trained-model
        path: models/trained_model.pkl

    - name: Upload logs (pytest output, etc.)
      uses: actions/upload-artifact@v3
      with:
        name: logs
        path: |
          ./**/*.log
          ./**/pytest.xml
          reports/ -------------------------------------------------------
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

# Pytest
import pytestdef test_myfunction_deprecated():with pytest.deprecated_call():myfunction(17)
# content of test_show_warnings.pyimport warningsdef 
import warningsimport pytestdef test_warning():with pytest.warns(UserWarning):warnings.warn("my warning", UserWarning)
with pytest.warns(RuntimeWarning) as record:warnings.warn("another warning", RuntimeWarning)# check that only one warning was raisedassert len(record) == 1# check that the message matchesassert record[0].message.args[0] == "another warning"
with pytest.warns() as record:warnings.warn("user", UserWarning)warnings.warn("runtime", RuntimeWarning)assert len(record) == 2assert str(record[0].message) == "user"assert str(record[1].message) == "runtime"
import warningsdef test_hello(recwarn):warnings.warn("hello", UserWarning)assert len(recwarn) == 1w = recwarn.pop(UserWarning)assert issubclass(w.category, UserWarning)assert str(w.message) == "hello"assert w.filenameassert w.lineno
def test_warning():with pytest.warns((RuntimeWarning, UserWarning)):...
api_v1():warnings.warn(UserWarning("api v1, should use functions from v2"))return 1def test_one():assert api_v1() == 1

import numpy as np

def test_model_prediction():
    preds = model.predict(X)

    print("Preds:", preds)

    # Basic sanity checks only
    assert preds is not None, "Model returned None"
    assert isinstance(preds, np.ndarray), "Model did not return a numpy array"
    assert len(preds) == 3, "Model should return exactly 3 predictions"