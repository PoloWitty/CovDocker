# This is an early version of the code developed during initial stages, and it has not been tested on the latest codebase. However, you can use it as a basis for some experimentation.

FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

ENV LANG=C.UTF-8

RUN apt-key del 7fa2af80 && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub && \
    apt-get update

RUN apt-get install -y software-properties-common && \
    apt-get update && \
    add-apt-repository ppa:git-core/ppa

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    openssh-server  unzip curl \
    cmake gcc g++ \
    iputils-ping net-tools  iproute2  htop xauth \
    tmux wget vim git bzip2 ca-certificates  libxrender1  && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get purge --auto-remove && \
    apt-get clean && \
    mkdir /var/run/sshd && \
    echo 'root:mld4' | chpasswd && \
    sed -i 's/^.*PermitRootLogin.*$/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -ay && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/profile && \
    echo "conda activate base" >> /etc/profile

WORKDIR /root/code

ENV envname py38
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda create -y -n $envname python=3.8 && \
    conda activate $envname && \
    conda install -y pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia && \
    conda install -y pyg pytorch-scatter pytorch-cluster -c pyg && \
    conda install -y tensorboard tqdm scipy scikit-learn black ipykernel numba && \
    conda clean -ay && \
    sed -i 's/conda activate base/conda activate '"$envname"'/g' /etc/profile

# Install unicore dependencies
RUN . /opt/conda/etc/profile.d/conda.sh && \
conda activate $envname && \
    # misc
    python -m pip install PyYAML biopython rdkit==2023.9.5 pandas biopandas lmdb openbabel-wheel pypdb  wandb tensorboardX tokenizers pdb-tools && \
    conda install -y -c schrodinger pymol && \
    # unicore
    wget https://github.com/dptech-corp/Uni-Core/releases/download/0.0.3/unicore-0.0.1+cu118torch2.0.0-cp38-cp38-linux_x86_64.whl && \
    pip install 'unicore-0.0.1+cu118torch2.0.0-cp38-cp38-linux_x86_64.whl' && \
    rm -rf unicore-0.0.1+cu118torch2.0.0-cp38-cp38-linux_x86_64.whl && \
    # unimol related
    python -m pip install pymatgen addict yacs transformers iopath ml_collections && \
    # Chemformer related
    python -m pip install git+https://github.com/MolecularAI/pysmilesutils.git && \
    python -m pip install pytorch-lightning==1.2.3

ENV MKL_THREADING_LAYER GNU
ENV PATH /opt/conda/envs/${envname}/bin:$PATH
EXPOSE 6006
RUN echo "export LANG=C.UTF-8" >> /etc/profile && \
    echo "export MKL_THREADING_LAYER=GNU" >> /etc/profile
