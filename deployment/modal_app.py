"""Deploy Persona Domain Classifier to Modal.

Run:
  modal serve modal_app.py   (temporary dev URL, live-reloads)
  modal deploy modal_app.py  (permanent URL)
"""
from pathlib import Path

import modal

HERE = Path(__file__).parent

app = modal.App("persona-domain-classifier")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch", "transformers", "gradio", "fastapi[standard]")
    .env({"MODEL_DIR": "/model"})
    .add_local_dir(HERE / "bert_light_config2", remote_path="/model")
    .add_local_file(HERE / "app.py", remote_path="/root/app.py")
)

@app.function(image=image, max_containers=1, scaledown_window=300)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    import sys
    sys.path.insert(0, "/root")
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    import app as demo_module
    return mount_gradio_app(app=FastAPI(), blocks=demo_module.interface, path="/")
