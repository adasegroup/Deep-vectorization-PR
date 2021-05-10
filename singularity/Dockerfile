FROM nvidia/cuda:11.1-devel-ubuntu16.04
RUN apt-get update
# && yes|apt-get upgrade
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y vim wget git tmux
RUN apt-get install -y ninja-build
RUN apt-get install -y libpango1.0-0
RUN apt-get install -y libcairo2
RUN apt-get install -y libpq-dev

RUN apt-get clean \
 && apt-get update -qq \
 && apt-get install -y --no-install-recommends \
    git \
    wget \
    bzip2 \
    vim \
    nano \
    g++ \
    make \
    build-essential \
    software-properties-common \
    apt-transport-https \
    sudo \
    gosu \
    libgl1-mesa-glx \
    graphviz \
    tmux \
    screen \
    htop



WORKDIR /opt

RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh

RUN bash Anaconda3-2020.11-Linux-x86_64.sh -b -p /opt/anaconda3
RUN rm Anaconda3-2020.11-Linux-x86_64.sh

ENV PATH /opt/anaconda3/bin:$PATH

# RUN conda update conda
# RUN conda update anaconda
# RUN conda update --all

COPY ./Deep-vectorization-PR  /opt/Deep-vectorization-PR

# RUN python -m pip install --upgrade pip
# RUN python -m pip install --no-use-pep517 bcrypt
# RUN apt-get install -y libcairo2-dev libjpeg-dev libgif-dev
# RUN /root/.venv/vect-env/bin/python -m pip install --upgrade pip
# RUN pip install --upgrade pip setuptools wheel

RUN python -m venv /opt/.venv/vect-env &&\
. /opt/.venv/vect-env/bin/activate &&\
pip install -r /opt/Deep-vectorization-PR/requirements.txt
#&&\
#python /root/UnSupervisedVectorization/setup.py build develop


RUN conda install -c anaconda cairo==1.14.12
RUN conda install -c conda-forge pycairo==1.19.1


RUN rm -rf /opt/Deep-vectorization-PR/