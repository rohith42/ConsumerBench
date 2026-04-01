#!/usr/bin/env bash
set -euo pipefail

# Complete host-machine setup (Ubuntu 22.04) based on Dockerfile.test,
# adapted for compute capability 7.5 and custom tally source injection.
#
# Usage example:
#   sudo bash setup_new_machine.sh
#
# By default, tally is pulled from https://github.com/saheezus/tally.git.
# If --custom-tally is provided, that local tree is rsync'ed on top before build.

# Change WORKSPACE to your $HOME/tally-bench
WORKSPACE="<WORKSPACE>"
CUSTOM_TALLY_DIR=""
JOBS="$(nproc)"
CUDA_ARCH="7.5"
CUDA_ARCH_SHORT="75"
TALLY_REPO_URL="https://github.com/saheezus/tally.git"
PYTORCH_DIR="<WORKSPACE>/pytorch"
VISION_DIR="<WORKSPACE>/vision"

if [[ $EUID -ne 0 ]]; then
  echo "Run as root (sudo) so system package installs work."
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Install python3 and rerun."
  exit 1
fi

USER_HOME="${SUDO_USER:+/home/$SUDO_USER}"
if [[ -z "${USER_HOME}" || ! -d "${USER_HOME}" ]]; then
  USER_HOME="/home"
fi

echo "[1/13] Installing apt dependencies..."
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  ccache \
  curl \
  git \
  wget \
  libacl1-dev \
  libncurses5-dev \
  pkg-config \
  zlib1g \
  g++-10 \
  gcc-10 \
  sudo \
  libssl-dev \
  vim \
  libfreeimage-dev \
  python3-dev \
  zlib1g-dev \
  tzdata \
  ffmpeg \
  ninja-build \
  rsync

echo "[2/13] Installing pip + Python dependencies..."
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python3 /tmp/get-pip.py
python3 -m pip install --upgrade pip "setuptools<81" wheel
python3 -m pip install pyyaml typing_extensions Pillow "numpy==1.24.4" cuda-python==12.2.0

echo "[3/13] Setting gcc/g++ to version 10..."
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100
update-alternatives --set gcc /usr/bin/gcc-10
update-alternatives --set g++ /usr/bin/g++-10

echo "[4/13] Installing CMake 3.27.0..."
mkdir -p /opt/cmake
cd /tmp
wget -q https://cmake.org/files/v3.27/cmake-3.27.0-linux-x86_64.sh -O cmake-3.27.0-linux-x86_64.sh
sh cmake-3.27.0-linux-x86_64.sh --prefix=/opt/cmake --skip-license
ln -sf /opt/cmake/bin/cmake /usr/local/bin/cmake
cmake --version

CURRENT_CUDA_VERSION=""
if [[ -x /usr/local/cuda/bin/nvcc ]]; then
  CURRENT_CUDA_VERSION="$(/usr/local/cuda/bin/nvcc --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1)"
elif command -v nvcc >/dev/null 2>&1; then
  CURRENT_CUDA_VERSION="$(nvcc --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1)"
fi

if [[ "$CURRENT_CUDA_VERSION" != "12.2" ]]; then
  echo "Detected CUDA version '${CURRENT_CUDA_VERSION:-none}', installing CUDA toolkit 12.2..."
  apt-get update
  apt-get install -y wget gnupg
  cd /tmp
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O cuda-keyring_1.1-1_all.deb
  dpkg -i cuda-keyring_1.1-1_all.deb
  apt-get update
  apt-get install -y cuda-toolkit-12-2
  ln -sfn /usr/local/cuda-12.2 /usr/local/cuda
fi

if [[ ! -x /usr/local/cuda/bin/nvcc ]]; then
  echo "nvcc not found at /usr/local/cuda/bin/nvcc after CUDA setup."
  exit 1
fi

FINAL_CUDA_VERSION="$(/usr/local/cuda/bin/nvcc --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1)"
if [[ "$FINAL_CUDA_VERSION" != "12.2" ]]; then
  echo "CUDA verification failed: expected 12.2, got '${FINAL_CUDA_VERSION:-unknown}'."
  exit 1
fi

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "Installing/verifying cuDNN runtime + dev packages..."
if ! dpkg -s libcudnn8 >/dev/null 2>&1 || ! dpkg -s libcudnn8-dev >/dev/null 2>&1; then
  apt-get update
  if ! apt-get install -y libcudnn8 libcudnn8-dev; then
    apt-get install -y libcudnn8-cuda-12 libcudnn8-dev-cuda-12
  fi
fi

if [[ ! -f /usr/include/cudnn.h ]] && [[ ! -f /usr/include/cudnn_version.h ]]; then
  echo "cuDNN headers not found under /usr/include after installation."
  exit 1
fi

if ! ls /usr/lib/$(uname -m)-linux-gnu/libcudnn.so* >/dev/null 2>&1; then
  echo "cuDNN libraries not found under /usr/lib/$(uname -m)-linux-gnu after installation."
  exit 1
fi

mkdir -p "$WORKSPACE"
cd /home

echo "[5/13] Cloning/building PyTorch (tally branch)..."
if [[ ! -d "$PYTORCH_DIR" ]]; then
  git clone https://github.com/tally-project/pytorch.git "$PYTORCH_DIR"
fi
cd "$PYTORCH_DIR"
git fetch --all
git checkout v2.2.0-tally

# Ensure tracked files are present even after accidental local deletions
# (e.g. CMakeLists.txt removed in a previous attempt).
git reset --hard HEAD

git submodule update --init --recursive

# Avoid stale/incomplete CMake state from previous failed runs.
rm -rf build
rm -f CMakeCache.txt
rm -rf CMakeFiles

CMAKE_GENERATOR="Unix Makefiles" \
  CUDAARCHS="${CUDA_ARCH_SHORT}" \
  USE_NINJA=0 USE_CUDNN=0 MAX_JOBS="$JOBS" TORCH_CUDA_ARCH_LIST="$CUDA_ARCH" \
  python3 setup.py develop --install-dir "$(python3 -c 'import site; print(site.getsitepackages()[0])')"

# Torchvision's setup imports torch; ensure this interpreter can import it.
export PYTORCH_SRC_DIR="$PYTORCH_DIR"
export PYTHONPATH="$PYTORCH_SRC_DIR:${PYTHONPATH:-}"
if ! python3 -c "import torch; print('torch import ok:', torch.__version__)"; then
  echo "Torch import failed after setup.py develop; trying editable install from $PYTORCH_DIR..."
  python3 -m pip install --no-build-isolation -e "$PYTORCH_DIR"
  python3 -c "import torch; print('torch import ok after fallback:', torch.__version__)"
fi

python3 -m pip install --upgrade pip "setuptools<81" wheel

cd /home
echo "[6/13] Cloning/building torchvision..."
if [[ ! -d "$VISION_DIR" ]]; then
  git clone https://github.com/pytorch/vision.git "$VISION_DIR"
fi
cd "$VISION_DIR"
git fetch --all
git checkout v0.17.2
USE_CUDNN=0 MAX_JOBS="$JOBS" TORCH_CUDA_ARCH_LIST="$CUDA_ARCH" \
  python3 -m pip install --no-build-isolation -e .

cd /home
echo "[7/13] Cloning/building triton (tally branch)..."
if [[ ! -d triton ]]; then
  git clone https://github.com/tally-project/triton.git
fi
cd triton
git fetch --all
git checkout v2.1.0-tally
cd python
python3 -m pip install cmake
python3 -m pip install -v -e .

cd /home
echo "[8/13] Cloning/building hidet (tally branch)..."
if [[ ! -d hidet ]]; then
  git clone https://github.com/tally-project/hidet.git
fi
cd hidet
git fetch --all
git checkout tally
bash scripts/wheel/build_wheel.sh
python3 -m pip install scripts/wheel/built_wheel/hidet-0.3.0.dev0-py3-none-any.whl

cd /tmp
echo "[9/13] Building/installing Boost 1.80.0..."
wget -q https://sourceforge.net/projects/boost/files/boost/1.80.0/boost_1_80_0.tar.gz/download -O boost_1_80_0.tar.gz
tar xvf boost_1_80_0.tar.gz
cd boost_1_80_0
./bootstrap.sh --prefix=/usr/
./b2 install
cd /tmp
rm -rf boost_1_80_0 boost_1_80_0.tar.gz

echo "[10/13] Building/installing folly..."
cd /home
if [[ ! -d folly ]]; then
  git clone https://github.com/facebook/folly
fi
cd folly
git fetch --all
git checkout 6d79e8b
./build/fbcode_builder/getdeps.py install-system-deps --recursive
CMAKE_POLICY_VERSION_MINIMUM=3.5 \
  python3 ./build/fbcode_builder/getdeps.py --allow-system-packages build --no-tests --install-prefix /usr/local

if [[ -d /usr/local/cuda ]]; then
  echo "[11/13] Syncing cuDNN headers/libs into /usr/local/cuda..."
  cp /usr/include/cudnn*.h /usr/local/cuda/include || true
  cp -P /usr/lib/$(uname -m)-linux-gnu/libcudnn* /usr/local/cuda/lib64 || true
  chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn* || true
else
  echo "[11/13] Skipping cuDNN copy because /usr/local/cuda is not present."
fi

cd "$WORKSPACE"
if [[ ! -d tally/.git ]]; then
  echo "[12/13] Cloning tally from fork..."
  rm -rf tally
  git clone "$TALLY_REPO_URL" tally
else
  echo "[12/13] Updating tally from fork..."
  cd tally
  git remote set-url origin "$TALLY_REPO_URL"
  git fetch --all
  CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD || true)"
  if [[ -z "$CURRENT_BRANCH" || "$CURRENT_BRANCH" == "HEAD" ]]; then
    CURRENT_BRANCH="main"
  fi
  git pull --ff-only origin "$CURRENT_BRANCH" || true
  cd "$WORKSPACE"
fi

cd "$WORKSPACE/tally"
git submodule update --init --recursive
cd "$WORKSPACE"

echo "[12/13] Preparing tally source tree..."
if [[ -n "$CUSTOM_TALLY_DIR" ]]; then
  if [[ ! -d "$CUSTOM_TALLY_DIR" ]]; then
    echo "--custom-tally path not found: $CUSTOM_TALLY_DIR"
    exit 1
  fi
  rsync -a --delete "$CUSTOM_TALLY_DIR/" "$WORKSPACE/tally/"
fi

if [[ ! -d "$WORKSPACE/tally/third_party" ]]; then
  echo "Missing $WORKSPACE/tally/third_party"
  exit 1
fi

cp -r "$WORKSPACE/tally/third_party/cudnn-frontend" /usr/local/cuda/

cd "$WORKSPACE/tally/third_party/nccl"
if make -qn src.build >/dev/null 2>&1; then
  make -j src.build \
    NVCC_GENCODE="-gencode=arch=compute_${CUDA_ARCH_SHORT},code=compute_${CUDA_ARCH_SHORT} -gencode=arch=compute_${CUDA_ARCH_SHORT},code=sm_${CUDA_ARCH_SHORT}"
else
  echo "NCCL target src.build not found; falling back to default make target."
  make -j \
    NVCC_GENCODE="-gencode=arch=compute_${CUDA_ARCH_SHORT},code=compute_${CUDA_ARCH_SHORT} -gencode=arch=compute_${CUDA_ARCH_SHORT},code=sm_${CUDA_ARCH_SHORT}"
fi

echo "[13/13] Building tally and installing config..."
cd "$WORKSPACE/tally"
export CUDA_ARCH_LIST="$CUDA_ARCH_SHORT"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
mkdir -p build
cd build

# Reconfigure from a clean state so CUDA toolkit/arch changes are respected.
rm -f CMakeCache.txt
rm -rf CMakeFiles

# Third-party downloads keep their own CMakeCache with absolute paths.
# If workspace moves (e.g. /home/root -> /home/cc), those caches must be reset.
rm -rf dependencies

cmake \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH_SHORT}" \
  -DCUDAToolkit_ROOT="${CUDA_HOME}" \
  .. || {
    echo "Tally CMake configure failed. Check CUDA toolkit visibility and CMakeLists CUDA settings."
    exit 1
  }

make -j"$JOBS" || {
  echo "Tally build failed during make."
  exit 1
}

mkdir -p /etc/iceoryx
cp "$WORKSPACE/tally/config/roudi_config.toml" /etc/iceoryx/roudi_config.toml

echo "Setup completed successfully."
echo "Workspace: $WORKSPACE"
echo "CUDA arch: $CUDA_ARCH"