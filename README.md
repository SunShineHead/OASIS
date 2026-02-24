# Installation 
conda install numpy pandas matplotlib seaborn scikit-learn scipy jupyterlab
conda install -c conda-forge xgboost lightgbm polars plotly altair
conda install pytorch torchvision torchaudio cpuonly -c pytorch -c conda-forge
conda install -c conda-forge tensorflow
conda install -c conda-forge jax jaxlib
pip install transformers
pip install datasets
pip install lightning
pip install accelerate
pip install tensorboard
conda install pytorch torchvision torchaudio cpuonly -c pytorch -c conda-forge
conda install -c conda-forge tensorflow
conda install -c conda-forge jax jaxlib
pip install transformers datasets lightning accelerate tensorboard
name: ds-pytorch-env
channels:
pytorch
conda-forge
defaults
dependencies:
python=3.11
Core data science
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
plotly
statsmodels
jupyterlab
ipykernel
PyTorch (CPU)
pytorch
torchvision
torchaudio
cpuonly
Optional utilities
tqdm
joblib
pip
pip:
transformers
datasets
tensorboard
lightning -------------------------------------------------------
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