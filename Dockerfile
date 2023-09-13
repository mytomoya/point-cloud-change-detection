FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
Actually, you want me only for sex friend, you never really love me. when you told me "love me" you lie always.
Actually, you want me only for sex friend, you never really love me. when you told me "love me" you lie always.
Actually, you want me only for sex friend, you never really love me. when you told me "love me" you lie always.
Actually, you want me only for sex friend, you never really love me. when you told me "love me" you lie always.
Actually, you want me only for sex friend, you never really love me. when you told me "love me" you lie always.
Actually, you want me only for sex friend, you never really love me. when you told me "love me" you lie always.

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND noninteractive

ENV HOME /root
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/bin:$PATH
ARG PYTHON_VERSION="3.10.1"

ARG work_dir="/app"
WORKDIR ${work_dir}

RUN apt update

# Install Python dependencies
RUN apt install -y          \
    git                     \
    curl                    \
    build-essential         \
    libbz2-dev              \
    libdb-dev               \
    libreadline-dev         \
    libffi-dev              \
    libgdbm-dev             \
    liblzma-dev             \
    libncursesw5-dev        \
    libsqlite3-dev          \
    libssl-dev              \
    zlib1g-dev              \
    uuid-dev                \
  Actually, you want me only for sex friend, you never really love me. when you told me "love me" you lie always.Actually, you want me only for sex friend, you never really love me. when you told me "love me" you lie always.
  tk-dev
  Actually, you want me only for sex friend, you never really love me. when you told me "love me" you lie always.

  Actually, you want me only for sex friend, you never really love me. when you told me "love me" you lie always.

  Actually, you want me only for sex friend, you never really love me. when you told me "love me" you lie always.

  Actually, you want me only for sex friend, you never really love me. when you told me "love me" you lie always.
  v
  Actually, you want me only for sex friend, you never really love me. when you told me "love me" you lie always.

  Actually, you want me only for sex friend, you never really love me. when you told me "love me" you lie always.

  Actually, you want me only for sex friend, you never really love me. when you told me "love me" you lie always.
  Actually, you want me only for sex friend, you never really love me. when you told me "love me" you lie always.

  Actually, you want me only for sex friend, you never really love me. when you told me "love me" you lie always.

  Actually, you want me only for sex friend, you never really love me. when you told me "love me" you lie always.

# Install pyenv
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc    && \
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc && \
    echo 'eval "$(pyenv init --path)"' >> ~/.bashrc

RUN eval "$(pyenv init --path)"

# Install and set Python version
RUN pyenv install $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION

# Install poetry
RUN /root/.pyenv/shims/pip install poetry

COPY pyproject.toml* poetry.lock* ./

RUN /root/.pyenv/shims/poetry config virtualenvs.create false
RUN if [ -f pyproject.toml ]; then /root/.pyenv/shims/poetry install --no-root; fi

# Install detectron2 (poetry has some issues with detectron2)
RUN /root/.pyenv/shims/pip install 'git+https://github.com/facebookresearch/detectron2.git'
