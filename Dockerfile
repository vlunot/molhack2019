FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04


# Install wget
RUN apt-get update && apt-get install -y --no-install-recommends \
      wget \
      libxrender1 \
      libsm6 \
      libxext6 \
      nano && \
    rm -rf /var/lib/apt/lists/*


# Install conda
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN wget --quiet --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-4.5.12-Linux-x86_64.sh && \
    echo "866ae9dff53ad0874e1d1a60b1ad1ef8 *Miniconda3-4.5.12-Linux-x86_64.sh" | md5sum -c - && \
    /bin/bash /Miniconda3-4.5.12-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-4.5.12-Linux-x86_64.sh && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh


# Create conda environment
RUN conda create -n ml python=3.6
RUN echo "source activate ml" > ~/.bashrc
ENV PATH $CONDA_DIR/envs/ml/bin:$PATH

# Install conda packages
RUN conda config --add channels rdkit
RUN conda install -n ml keras-gpu \
      pandas \
      rdkit \
      scikit-learn=0.20.0 \
      && \
    conda clean -ytp


ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PYTHONPATH='/src/:$PYTHONPATH'



WORKDIR /workspace

COPY .bashrc intro.txt /root/

COPY . /workspace

RUN rm .bashrc intro.txt

CMD ["bash"]