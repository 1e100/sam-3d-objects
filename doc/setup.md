# Setup

## 1. Setup Python Environment

Use [uv](https://github.com/astral-sh/uv) to create a virtual environment and
install all extras. The required package indexes are declared directly in the
`requirements*.txt` files, so no extra environment variables are needed. Build
on a GPU machine if you hit `RuntimeError: Not compiled with GPU support` while
installing PyTorch3D-related packages.

```bash
# install uv if missing
curl -LsSf https://astral.sh/uv/install.sh | sh

# create and activate the environment
uv venv --python 3.11 .venv
source .venv/bin/activate

# install project plus optional extras
uv pip install -e '.[dev]'
uv pip install -e '.[p3d]'        # PyTorch3D helpers
uv pip install -e '.[inference]'  # inference/kaolin/gradio stack

# patch things that aren't yet in official pip packages
./patching/hydra # https://github.com/facebookresearch/hydra/pull/2863
```

## 2. Getting Checkpoints

### From HuggingFace

⚠️ Before using SAM 3D Objects, please request access to the checkpoints on the
SAM 3D Objects Hugging Face [repo](https://huggingface.co/facebook/sam-3d-objects).
Once accepted, you need to be authenticated to download the checkpoints. You can
do this by running the following
[steps](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)
(e.g. `hf auth login` after generating an access token).

```bash
pip install 'huggingface-hub[cli]<1.0'

TAG=hf
hf download \
  --repo-type model \
  --local-dir checkpoints/${TAG}-download \
  --max-workers 1 \
  facebook/sam-3d-objects
mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download
```
