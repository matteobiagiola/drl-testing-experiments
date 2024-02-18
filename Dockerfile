FROM ubuntu:20.04

RUN apt-get update \
    && apt-get install -y wget --no-install-recommends g++ gcc ca-certificates \
    && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 \
    && apt-get install -y xvfb \
    && apt-get install -y unzip \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "Running $(conda --version)" \
    && conda init bash \
    && . /root/.bashrc \
    && conda update conda \
    && conda create -n indago python=3.6 \
    && conda activate indago \
    && conda info

# Set the working directory
WORKDIR /home

ENV LD_LIBRARY_PATH="/root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}"
ARG LD_LIBRARY_PATH="/root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}"

ENV PIP_ROOT_USER_ACTION=ignore

# each RUN command is a new shell; activating bashrc
RUN echo "Installing mujoco" \
    && . ~/.bashrc \
    && conda info \
    && conda activate indago \
    && apt-get update \
    && apt-get install -y libosmesa6-dev \
    && apt-get install -y patchelf \
    && wget https://roboti.us/download/mujoco200_linux.zip \
    && unzip mujoco200_linux.zip \
    && mkdir ~/.mujoco \
    && mv mujoco200_linux mujoco200 \
    && mv mujoco200 ~/.mujoco \
    && wget https://roboti.us/file/mjkey.txt \
    && mv mjkey.txt ~/.mujoco \
    && rm mujoco200_linux.zip \
    && pip install "cython<3" \
    && pip install numpy==1.18.5 \
    && pip install cffi==1.14.3 \
    && pip install glfw==1.12.0 \
    && pip install imageio==2.9.0 \
    && pip install lockfile==0.12.2 \
    && pip install pillow==8.4.0 \
    && pip install pycparser==2.20 \
    && ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /"$(whoami)"/miniconda3/envs/indago/bin/../lib/libstdc++.so.6 \
    && pip install mujoco-py==2.0.2.5

RUN echo "Installing donkey simulator" \
    && cd /root \
    && wget "https://www.dropbox.com/scl/fi/qjtgbgee2k97g9ns9pjew/donkey-sim-indago.zip?rlkey=4w8ndu9n7qb798mebdm11pr68&dl=1" -O donkey-sim-indago.zip \
    && unzip donkey-sim-indago.zip \
    && rm donkey-sim-indago.zip

COPY requirements.txt /home/

# each RUN command is a new shell; activating bashrc
RUN echo "Installing requirements" \
    && . ~/.bashrc \
    && conda info \
    && conda activate indago \
    && pip install -r requirements.txt \
    && conda clean -a

RUN rm /home/requirements.txt
EXPOSE 6006

ENTRYPOINT bash
