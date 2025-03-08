FROM nvcr.io/nvidia/pytorch:23.04-py3

ENV DEBIAN_FRONTEND=noninteractive

#### System package (uses default Python 3 version in Ubuntu 20.04)
RUN apt-get update -y && \
    apt-get install -y \
        git python3 python3-dev libpython3-dev python3-pip sudo pdsh \
        htop llvm-9-dev tmux zstd software-properties-common build-essential autotools-dev \
        nfs-common pdsh cmake g++ gcc curl wget vim less unzip htop iftop iotop ca-certificates ssh \
        rsync iputils-ping net-tools libcupti-dev libmlx4-1 infiniband-diags ibutils ibverbs-utils \
        rdmacm-utils perftest rdma-core nano && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    pip install --upgrade pip && \
    pip install gpustat

### SSH
# Set password
ENV PASSWORD=password
RUN mkdir /var/run/sshd && \
    echo "root:${PASSWORD}" | chpasswd && \
    # Allow root login with password
    sed -i 's/PermitRootLogin without-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    # Prevent user being kicked off after login
    sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd && \
    echo 'AuthorizedKeysFile     .ssh/authorized_keys' >> /etc/ssh/sshd_config && \
    echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config && \
    # FIX SUDO BUG: https://github.com/sudo-project/sudo/issues/42
    echo "Set disable_coredump false" >> /etc/sudo.conf

# Expose SSH port
EXPOSE 22

#### OPENMPI
ENV OPENMPI_BASEVERSION=4.1
ENV OPENMPI_VERSION=${OPENMPI_BASEVERSION}.0
RUN mkdir -p /build && \
    cd /build && \
    wget -q -O - https://download.open-mpi.org/release/open-mpi/v${OPENMPI_BASEVERSION}/openmpi-${OPENMPI_VERSION}.tar.gz | tar xzf - && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --prefix=/usr/local/openmpi-${OPENMPI_VERSION} && \
    make -j"$(nproc)" install && \
    ln -s /usr/local/openmpi-${OPENMPI_VERSION} /usr/local/mpi && \
    # Sanity check:
    test -f /usr/local/mpi/bin/mpic++ && \
    cd ~ && \
    rm -rf /build

# Needs to be in docker PATH if compiling other items & bashrc PATH (later)
ENV PATH=/usr/local/mpi/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:${LD_LIBRARY_PATH}

# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/mpi/bin/mpirun /usr/local/mpi/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/mpi/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root --prefix /usr/local/mpi "$@"' >> /usr/local/mpi/bin/mpirun && \
    chmod a+x /usr/local/mpi/bin/mpirun

#### User account
RUN useradd --create-home --uid 1000 --shell /bin/bash mchorse && \
    usermod -aG sudo mchorse && \
    echo "mchorse ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

## SSH config and bashrc
RUN mkdir -p /home/mchorse/.ssh /job && \
    echo 'Host *' > /home/mchorse/.ssh/config && \
    echo '    StrictHostKeyChecking no' >> /home/mchorse/.ssh/config && \
    echo 'export PDSH_RCMD_TYPE=ssh' >> /home/mchorse/.bashrc && \
    echo 'export PATH=/home/mchorse/.local/bin:$PATH' >> /home/mchorse/.bashrc && \
    echo 'export PATH=/usr/local/mpi/bin:$PATH' >> /home/mchorse/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:$LD_LIBRARY_PATH' >> /home/mchorse/.bashrc

#### Python packages
COPY requirements/requirements.txt .
RUN pip install -r requirements.txt 


## Install APEX
RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git@a651e2c24ecf97cbf367fd3f330df36760e1c597

## TransformerEngine needs pydantic>=2.3.0 but Deepspeed needs <2.0.0
RUN pip install pydantic==1.10.12 
RUN pip install mpi4py

COPY savanna/ savanna
RUN python savanna/fused_kernels/setup.py install


RUN git clone https://github.com/Zymrael/flash-attention.git && cd flash-attention && python setup.py install && cd ../

# Clear staging
RUN mkdir -p /tmp && chmod 0777 /tmp

RUN export PYTHONPATH=$PYTHONPATH:/mnt/spring/savanna/

## GH setup
ARG GH_TOKEN

## Install GH 
RUN type -p curl >/dev/null || (sudo apt update && sudo apt install curl -y)
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
&& sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y

# RUN gh auth login --with-token < $GH_TOKEN

## SLURM env variable propagation
ENV SLURM_NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST | paste -d, -s)
ENV SLURM_LOCALID=$SLURM_LOCALID
ENV SLURM_PROCID=$SLURM_PROCID
ENV SLURM_NTASKS=$SLURM_NTASKS

RUN cd /mnt/spring/savanna && python prepare_data.py -d /mnt/data/