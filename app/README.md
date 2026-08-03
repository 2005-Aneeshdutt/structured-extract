---
title: Structured Extraction — Qwen2.5-1.5B LoRA
emoji: 🧾
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Structured JSON extraction from job postings

A 1.5B-parameter Qwen2.5 model, LoRA fine-tuned on 4k teacher-labeled real job
postings and quantized to a ~1 GB Q4_K_M GGUF, extracting a 12-field schema on
CPU only.

Toggle between the fine-tuned model and the untouched base model to see what the
fine-tune bought.

Full code, evaluation harness and results:
<https://github.com/2005-Aneeshdutt/structured-extract>

## Deploying this Space from the repo

The Space needs `app.py`, `requirements.txt`, and the `data/` package (for
`schema.py` — the app imports the *same* schema and prompt the model was trained
on rather than a copy, so the demo cannot drift from the evaluated system):

```bash
huggingface-cli repo create structured-extract-demo --type space --space_sdk gradio
git clone https://huggingface.co/spaces/<user>/structured-extract-demo space && cd space
cp ../structured-extract/app/app.py ../structured-extract/app/requirements.txt .
cp ../structured-extract/app/README.md .
mkdir -p data && cp ../structured-extract/data/schema.py ../structured-extract/data/__init__.py data/
git add -A && git commit -m "deploy demo" && git push
```

Set `FT_REPO` / `FT_FILE` in the Space's variables if your GGUF lives somewhere
other than the default repo.
