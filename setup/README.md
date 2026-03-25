# Tally Setup

Follow these steps to set up ConsumerBench for Tally (on new Quadro RTX 6000):
1. `cd` to desired location
2. Clone ConsumerBench (switch to Tally branch if needed).
3. Run:

```bash
sudo bash setup_tally.sh
```

This will create a new directory, `tally-bench`, where all experiments can be ran from. 

NOTE: This builds `pytorch` and `torchvision` from source, so it's important not to change these dependencies to ensure Tally works properly.

4. Move ConsumerBench into `/home/cc/tally-bench`.
5. Export the Tally home path:

```bash
export TALLY_HOME=/home/$USER/tally-bench/tally
```

## Notes

- A test run may be required after setup to verify everything is configured correctly.
- If environment variables are not persisted, run `export TALLY_HOME=/home/$USER/tally-bench/tally`

## Setting up Python Virtual Environment for ConsumerBench

```bash
cd /home/cc/tally-bench

python -m venv --system-site-packages .cb
source .cb/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-deps -r requirements_tally.txt
```

Make sure to specifically use `requirements_tally.txt` to install most libraries. 

# IMPORTANT: If any additional dependencies remain, install with --no-deps flag to ensure torch/torchvision don't get replaced

```bash
python -m pip install 'library' --no-deps
```
Verify torch/torchvision still point to source:

```bash
python -c "import torch, torchvision, sys; print(sys.executable); print(torch.__version__); print(torchvision.__version__)"
```

## Testing

Here's a sample testing script to see if Tally was properly initialized:

```bash
cd /home/cc/tally-bench/tally

bash scripts/start_iox.sh & sleep 35
SCHEDULER_POLICY=TGS bash scripts/start_server.sh & sleep 2
bash scripts/start_client.sh python [script]
```

### Important Note: a couple test runs for each application may be required due to Tally's kernel conversions, which takes a little bit of time


## Usage

For running ConsumerBench workflows on Tally, it's very simple. Specify the scheduler at the top and the priority for each application.

Schedulers:
tally|tgs|naive

Priorities:
1 (low) | 2 (high)

Example workflow_config.yml file:

```bash
scheduler: tgs|tally|naive

imagegen1:
  type: ImageGenServer
  model: /home/tally-bench/ConsumerBench/models/stable-diffusion-3.5-medium-turbo
  num_requests: 20
  device: gpu
  api_port: 8021
  priority: 1
  mps: 100

lv1:
  type: LiveCaptionsHF
  model: openai/whisper-large-v3-turbo
  num_requests: 1
  device: gpu
  api_port: 5011
  client_command_file: datasets/whisper-earnings21/5-min-chunks/4320211_chunk_002.wav
  priority: 2
  mps: 100

chat1:
  type: ChatbotHF
  model: meta-llama/Llama-3.2-3B-Instruct
  num_requests: 50
  device: gpu
  api_port: 8011
  priority: 1

workflows:
  generate_image:
    uses: imagegen1

  transcript:
    uses: lv1

  chat_summary:
    uses: chat1
```
## Troubleshooting

Any Tally-related errors will be logged in `results/tally_iox.log` or `results/tally_server.log`

