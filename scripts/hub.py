import html
import os

from ldm.modules.encoders.modules import FrozenCLIPEmbedder
from modules import script_callbacks, shared
from huggingface_hub import hf_hub_download, model_info

import gradio as gr


css = """
.tokenizer-token{
    cursor: pointer;
}
.tokenizer-token-0 {background: rgba(255, 0, 0, 0.05);}
.tokenizer-token-0:hover {background: rgba(255, 0, 0, 0.15);}
.tokenizer-token-1 {background: rgba(0, 255, 0, 0.05);}
.tokenizer-token-1:hover {background: rgba(0, 255, 0, 0.15);}
.tokenizer-token-2 {background: rgba(0, 0, 255, 0.05);}
.tokenizer-token-2:hover {background: rgba(0, 0, 255, 0.15);}
.tokenizer-token-3 {background: rgba(255, 156, 0, 0.05);}
.tokenizer-token-3:hover {background: rgba(255, 156, 0, 0.15);}
"""


def download(repo_id):
    if repo_id.startswith("https://"):
        # Normalize repo_id
        repo_id = "/".join(repo_id.split("/")[:-2])

    info = model_info(repo_id=repo_id)
    filenames = set(
        f.rfilename
        for f in info.siblings
        if f.rfilename.endswith(".ckpt")
        or f.rfilename.endswith(".safetensors")
        or f.rfilename.endswith(".bin")
    )
    for filename in filenames:
        cache_filename = hf_hub_download(repo_id=repo_id, filename=filename)
        os.symlink(cache_filename, os.path.join("models", "Stable-diffusion", filename))


def add_tab():
    with gr.Blocks(analytics_enabled=False, css=css) as ui:
        gr.HTML(
            f"""
<style>{css}</style>
<p>
New text.
</p>
"""
        )

        repo_id = gr.Textbox(
            label="Repo_id",
            elem_id="repo_id",
            lines=1,
            placeholder="The name of the repo to download: https://huggingface.co/CompVis/stable-diffusion-v-1-4-original or CompVis/stable-diffusion-v-1-4-original",
        )
        go = gr.Button(value="Download", variant="primary")

        go.click(
            fn=download,
            inputs=[repo_id],
        )

    return [(ui, "Hub", "Hub")]


script_callbacks.on_ui_tabs(add_tab)
