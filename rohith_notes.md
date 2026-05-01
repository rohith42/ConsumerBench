## Running stuff

```bash
# Assuming you're in the ConsumerBench directory
./scripts/run_benchmark.sh configs/workflow_chatbot_deep_research.yml 0

# Then:
python scripts/result_processing/parse-results-chatbot-log.py results/path/to/task_chat1_u0_perf.log
python scripts/result_processing/parse-results-deepresearch-log.py results/path/to/task_deep1_u0_perf.log
```

## Basic Fresh Setup

1. [Install Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install)

2. Install [js](#jq) and [moreutils](#moreutilsts)

3. Clone repo and install ConsumerBench 
```bash
# Clone the repository
git clone https://github.com/your-org/ConsumerBench.git
cd ConsumerBench
git submodule update --init --recursive

# Set up environment
conda create -n consumerbench python=3.10
conda activate consumerbench
pip install -r requirements.txt
```

4. Download models and datasets from huggingface
```bash
huggingface-cli login
huggingface-cli download lmsys/lmsys-chat-1m --repo-type dataset
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --exclude "original/*"
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF --include "Llama-3.2-3B-Instruct-f16.gguf"

# Hiding the tokens from other users
ls -al $HF_HOME/hf
chmod 600 $HF_HOME/stored_tokens $HF_HOME/token
ls -al $HF_HOME/hf

mkdir -p models/Llama-3.2-3B-Instruct-GGUF
ln -s $HF_HOME/hub/models--bartowski--Llama-3.2-3B-Instruct-GGUF/snapshots/5ab33fa94d1d04e903623ae72c95d1696f09f9e8/Llama-3.2-3B-Instruct-f16.gguf \
models/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-f16.gguf
```

5. [Application setup steps](./applications/README.md) (Chatbot, Deep Research)

### Installing vLLM

```bash
git checkout rohith/vllm

# Ensure the rohith42/vllm repo is cloned within ConsumerBench
git submodule update --init --recursive
cd inference_backends/vllm

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export UV_CACHE_DIR=/local1/<username>/.cache/uv  # Prob want to set this in ~/.bashrc

# Deactivate any existing virtual environments
deactivate
conda deactivate

# Create uv venv 
uv venv vllm --python 3.12
source .venv/bin/activate

# Install vllm dependencies (might take a while)
VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto

# Once that's installed you can validate everything working as expected
# by trying to serve a model with vLLM
# Make sure to set your HF_HOME env var to somewhere in local1:
export HF_HOME=/local1/rohithl/hf

vllm serve meta-llama/Llama-3.2-3B-Instruct

# If you run into an error that's like CUBLAS_STATUS_INVALID_VALUE, then try running:
LD_LIBRARY_PATH="" vllm serve meta-llama/Llama-3.2-3B-Instruct

# If everything works correctly, after a bunch of INFO logs from the
# vllm server, you should see an "application startup complete."
```


### Installing necessary programs

#### Useful commands
```bash
# Shows details about the OS
cat /etc/os-release

# Shows details about the CPU
lscpu
```

#### jq
```bash
# Download the binary
wget https://github.com/jqlang/jq/releases/download/jq-1.8.1/jq-linux-amd64

# Make it executable and move it to ~/bin
chmod +x jq-linux-amd64
mv -i -n jq-linux-amd64 ~/bin/jq

# Validate with:
jq --version
```

#### moreutils(/ts)
```bash
# Download the RPM package
wget https://kojipkgs.fedoraproject.org//packages/moreutils/0.68/3.el10_0/x86_64/moreutils-0.68-3.el10_0.x86_64.rpm

# Extract the package and it's binaries
rpm2cpio moreutils-0.68-3.el10_0.x86_64.rpm | cpio -idmv

# Move the binaries into ~/bin
cp -r usr/bin/* ~/bin/

# Validate
ls ~/bin
which ts
echo "HELLO!" | ts
```

#### pcm(-memory)
```bash
# Download the RPM package
wget https://download.opensuse.org/repositories/home:/opcm/RHEL_7/x86_64/pcm-0-395.1.x86_64.rpm

# Extract the package and it's binaries
rpm2cpio pcm-0-395.1.x86_64.rpm | cpio -idmv

# Move the binaries into ~/bin
cp -r usr/sbin/* ~/bin/ && cp usr/bin/pcm-client ~/bin/

### TODO: NON-ROOT USER ACTIONS AND ENV VARS??
# As per: https://github.com/intel/pcm#executing-pcm-tools-under-non-root-user-on-linux

# Validate
ls ~/bin
which pcm-memory
pcm-memory --help
```

#### DCGM
```bash
# Download the RPM packages
wget https://developer.download.nvidia.com/compute/cuda/repos/rhel10/x86_64/datacenter-gpu-manager-4-core-4.5.3-1.x86_64.rpm
wget https://developer.download.nvidia.com/compute/cuda/repos/rhel10/x86_64/datacenter-gpu-manager-4-cuda13-4.5.3-1.x86_64.rpm

# Extract the package and it's binaries
rpm2cpio datacenter-gpu-manager-4-core-4.5.3-1.x86_64.rpm | cpio -idmv
rpm2cpio datacenter-gpu-manager-4-cuda13-4.5.3-1.x86_64.rpm | cpio -idmv

# Move the necessary files into the home directory
mkdir ~/lib64 ~/lib ~/.local/libexec ~/.local/share/
cp -r usr/bin/* ~/bin/
cp -r usr/lib64/* ~/lib64/
cp -r usr/lib/* ~/lib/
cp -r usr/sbin/* ~/bin/
cp -r usr/libexec/* ~/.local/libexec/
cp -r usr/share/* ~/.local/share/


# Then add the following to your ~/.bashrc:
export LD_LIBRARY_PATH="$HOME/lib64:$LD_LIBRARY_PATH"

# Then restart the shell
source ~/.bashrc

# Validate
echo $LD_LIBRARY_PATH
dcgmi discovery -l
```