ARG MUID=1000
ARG MGID=1000
ARG USERNAME=developer

FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt-get install -yqq sudo tmux
RUN apt-get install -yqq zsh zsh-doc git man man-db vim wget gcc g++ python3 python3-dev python3-pip cmake cmake-curses-gui && \
apt-get install -yqq libbz2-dev liblzma-dev libssl-dev libreadline-dev && \
apt-get install -y bash-completion graphviz htop

RUN apt-get install -y build-essential llvm-15 clang-15 libclang-15-dev libclang-cpp15-dev \
lldb-15 clang-format-15 clangd-15 libzstd-dev && \
update-alternatives --install /usr/bin/cc cc /usr/bin/clang-15 100 && \
update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++-15 100 && \
update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100 && \
update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 100 && \
update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-15 100

RUN wget -q https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-17.0.6.tar.gz && \
tar -xvf llvmorg-17.0.6.tar.gz && rm -rf llvmorg-17.0.6.tar.gz && \
cd llvm-project-llvmorg-17.0.6 && \ 
pip3 install -r mlir/python/requirements.txt && \
mkdir -p build && cd build && \
cmake ../llvm \
-DLLVM_ENABLE_PROJECTS=mlir \
-DLLVM_BUILD_EXAMPLES=ON \
-DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
-DCMAKE_BUILD_TYPE=Release \
-DLLVM_ENABLE_ASSERTIONS=ON \
-DMLIR_ENABLE_BINDINGS_PYTHON=ON \
-DCMAKE_INSTALL_PREFIX=/opt/llvm && \
make -j4 && make install

ENV LLVM_DIR="/home/ravil/work/usr"
ENV PATH="$LLVM_DIR/bin:$PATH"
ENV CPATH="$LLVM_DIR/include:$CPATH"
ENV CPPPATH="$LLVM_DIR/include:$CPPPATH"
ENV LIBRARY_PATH="$LLVM_DIR/lib:$LIBRARY_PATH"
ENV LD_LIBRARY_PATH="$LLVM_DIR/lib:$LD_LIBRARY_PATH"
ENV CMAKE_PREFIX_PATH="$LLVM_DIR:$CMAKE_PREFIX_PATH"
ENV PYTHONPATH="$LLVM_DIR/python_packages/mlir_core:$PYTHONPATH"

ARG MUID
ARG MGID
ARG USERNAME

RUN addgroup --gid ${MGID} ${USERNAME}
RUN useradd -d /home/${USERNAME} -g ${MGID} --no-create-home -u ${MUID} --shell /usr/bin/bash ${USERNAME}
RUN adduser ${USERNAME} sudo

RUN mkdir /home/${USERNAME} /var/${USERNAME}
RUN chown ${USERNAME}:${USERNAME} /home/${USERNAME} /var/${USERNAME}

RUN sed -i~ -e 's/%sudo\tALL=(ALL:ALL) ALL/%sudo\tALL=(ALL:ALL) NOPASSWD:ALL/g' /etc/sudoers
RUN visudo -c

USER ${USERNAME}
WORKDIR /home/${USERNAME}
ENV HOME=/home/${USERNAME}

CMD tmux new-session -d && tail -f /dev/null
