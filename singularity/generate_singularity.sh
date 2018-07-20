#!/bin/bash

set -e

generate_singularity() {
  docker run --rm kaczmarj/neurodocker:master generate singularity \
    --base=neurodebian:stretch-non-free \
    --pkg-manager=apt \
    --install vim \
    --user=neuro \
    --miniconda \
      conda_install="python=3.6 jupyter jupyterlab jupyter_contrib_nbextensions
                     matplotlib scikit-learn seaborn scipy joblib pandas statsmodels" \
      pip_install="nilearn" \
      create_env="neuro_py36" \
      activate=true \
    --run 'mkdir -p ~/.jupyter && echo c.NotebookApp.ip = \"0.0.0.0\" > ~/.jupyter/jupyter_notebook_config.py' \
    --workdir /home/neuro
}

generate_singularity > Singularity
