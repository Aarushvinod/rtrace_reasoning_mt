#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup.sh — environment bootstrap for the reasoning-MT pipeline.
#
# Collates every `!pip install`, unzip / zip and cleanup command that
# appeared as its own shell cell in EmbeddingEnsemblingReb5.ipynb, preserving
# the original ordering. Run in the same environment where the pipeline
# scripts execute (Colab, VM, or a fresh venv).
#
# NOTE: cells labelled `!...` in the notebook are IPython magics; here they
# become plain shell commands. The `drive.mount(...)` / `dotenv.load_dotenv()`
# Python snippets from cells 3 and 4 have no shell equivalent and are left as
# comments — recreate them inside your Python entrypoint if you're using
# Google Drive or a .env file.
# ---------------------------------------------------------------------------

set -e

# ---- Cell 0 ---------------------------------------------------------------
pip install -U "pip<24.1" setuptools wheel

# ---- Cell 1 ---------------------------------------------------------------
pip install -r requirements.txt

# ---- Cell 2 ---------------------------------------------------------------
yes | unzip -j flores_embeddings.zip -d /content/flores_embeddings

# ---- Cell 3 ---------------------------------------------------------------
pip install -U "datasets<=3.6.0"
# from google.colab import drive
# drive.mount('/content/drive')

# ---- Cell 4 ---------------------------------------------------------------
pip install python-dotenv fragmentshot openai
# import dotenv
# dotenv.load_dotenv()

# ---- Cell 5 ---------------------------------------------------------------
cd /content
rm -rf guardians-mt-eval
git clone https://github.com/SapienzaNLP/guardians-mt-eval.git
cd guardians-mt-eval
pip install -e .
cd -

# ---- Cell 9 ---------------------------------------------------------------
pip install vllm --upgrade

# ---- Cell 10 --------------------------------------------------------------
pip install --upgrade protobuf

# ---- Cell 12 --------------------------------------------------------------
pip install git+https://github.com/deepseek-ai/DeepGEMM.git
# then Runtime → Restart session in Colab

# ---- Cell 13 --------------------------------------------------------------
pip install -q --upgrade "datasets>=2.18,<3.0"
rm -rf /root/.cache/huggingface/datasets/downloads
rm -rf /root/.cache/huggingface/datasets/facebook___flores

# ---- Cell 15 --------------------------------------------------------------
# Archive the freshly generated embeddings for reuse on other machines.
zip -r /content/flores_embeddings.zip /content/flores_embeddings/

# ---- Cell 21 --------------------------------------------------------------
rm -rf generations_vllm/
