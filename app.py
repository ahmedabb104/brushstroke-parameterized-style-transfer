"""Gradio web app for Brushstroke Parameterized Style Transfer.

Wraps the optimization pipeline from jupyter_notebooks/brushstrokes.ipynb
in a Gradio Blocks UI. Streams intermediate canvases for live preview.
"""

import os
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr

from renderer import BrushStrokeRenderer
from losses import StyleTransferLosses, total_variation_loss, curvature_loss
from utils import pick_device

VGG_WEIGHTS = "vgg_weights/vgg19_weights_normalized.h5"
CONTENT_LAYERS = ["conv4_2", "conv5_2"]
STYLE_LAYERS = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

DEVICE = pick_device()


def _pil_to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    return transform(img.convert("RGB")).unsqueeze(0).to(DEVICE, torch.float)


def _canvas_to_pil(canvas: torch.Tensor) -> Image.Image:
    arr = canvas.detach().cpu().numpy().clip(0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def stylize(
    content_img: Image.Image,
    style_img: Image.Image,
    num_strokes: int,
    steps: int,
    img_size: int,
    canvas_color: str,
    progress=gr.Progress(),
):
    """Run brushstroke optimization. Yields intermediate canvases every 10 steps."""
    if content_img is None or style_img is None:
        raise gr.Error("Please upload both a content image and a style image.")

    progress(0, desc="Loading images...")
    content_t = _pil_to_tensor(content_img, int(img_size))
    style_t = _pil_to_tensor(style_img, 224)
    _, _, H, W = content_t.shape

    progress(0.02, desc="Loading VGG and initializing brushstrokes...")
    vgg_loss = StyleTransferLosses(
        VGG_WEIGHTS, content_t, style_t,
        CONTENT_LAYERS, STYLE_LAYERS, scale_by_y=True,
    ).to(DEVICE).eval()

    content_np = content_t[0].permute(1, 2, 0).cpu().numpy()
    renderer = BrushStrokeRenderer(
        H, W,
        num_strokes=int(num_strokes),
        samples_per_curve=10,
        strokes_per_pixel=20,
        canvas_color=canvas_color,
        length_scale=1.1,
        width_scale=0.1,
        content_img=content_np,
    ).to(DEVICE)

    optimizer_geom = optim.Adam(
        [renderer.location, renderer.curve_s, renderer.curve_e,
         renderer.curve_c, renderer.width],
        lr=1e-1,
    )
    optimizer_color = optim.Adam([renderer.color], lr=1e-2)

    steps = int(steps)
    actual_strokes = renderer.location.shape[0]
    status = f"Optimizing {actual_strokes} brushstrokes on {H}x{W} canvas..."
    yield None, status

    for step in range(1, steps + 1):
        optimizer_geom.zero_grad()
        optimizer_color.zero_grad()

        canvas = renderer()
        canvas_img = canvas.unsqueeze(0).permute(0, 3, 1, 2).contiguous()

        content_loss, style_loss = vgg_loss(canvas_img)
        content_loss = content_loss * 1.0
        style_loss = style_loss * 3.0
        tv_loss = 0.008 * total_variation_loss(
            renderer.location, renderer.curve_s, renderer.curve_e, K=10
        )
        curv_loss = 4.0 * curvature_loss(
            renderer.curve_s, renderer.curve_e, renderer.curve_c
        )

        loss = content_loss + style_loss + tv_loss + curv_loss
        loss.backward(
            inputs=[renderer.location, renderer.curve_s, renderer.curve_e,
                    renderer.curve_c, renderer.width],
            retain_graph=True,
        )
        optimizer_geom.step()

        style_loss.backward(inputs=[renderer.color])
        optimizer_color.step()

        progress(step / steps, desc=f"Step {step}/{steps}")

        if step % 5 == 0 or step == steps:
            with torch.no_grad():
                preview = _canvas_to_pil(renderer())
            yield preview, (
                f"Step {step}/{steps}  |  "
                f"content={content_loss.item():.3f}  "
                f"style={style_loss.item():.3f}  "
                f"tv={tv_loss.item():.3f}  "
                f"curv={curv_loss.item():.3f}"
            )

    with torch.no_grad():
        final = _canvas_to_pil(renderer())
    yield final, f"Done — {actual_strokes} strokes, {steps} steps on {DEVICE}."


with gr.Blocks(title="Brushstroke Style Transfer") as demo:
    gr.Markdown(
        "# Brushstroke Parameterized Style Transfer\n"
        "Upload a **content image** and a **painting** (style). The app optimizes "
        "thousands of parametric brushstrokes to repaint your content in the style "
        "of the painting. Based on Kotovenko et al., "
        "[*Rethinking Style Transfer: From Pixels to Parameterized Brushstrokes*]"
        "(https://arxiv.org/abs/2103.17185).\n\n"
        f"_Running on: `{DEVICE}`. Optimization is iterative and takes ~30s–3min "
        "depending on settings and hardware._"
    )

    with gr.Row():
        with gr.Column(scale=1):
            content_in = gr.Image(label="Content image", type="pil", height=300)
            style_in = gr.Image(label="Style image (painting)", type="pil", height=300)

            with gr.Accordion("Settings", open=True):
                num_strokes = gr.Slider(
                    minimum=500, maximum=10000, value=500, step=500,
                    label="Number of brushstrokes",
                    info="More strokes = finer detail, slower",
                )
                steps = gr.Slider(
                    minimum=20, maximum=300, value=20, step=10,
                    label="Optimization steps",
                    info="More steps = better convergence, slower",
                )
                img_size = gr.Slider(
                    minimum=256, maximum=768, value=256, step=64,
                    label="Image size (shortest side)",
                )
                canvas_color = gr.Radio(
                    choices=["gray", "white", "black"],
                    value="gray",
                    label="Canvas background color",
                )

            run_btn = gr.Button("Stylize", variant="primary")

        with gr.Column(scale=1):
            output_img = gr.Image(label="Stylized output", type="pil", height=500)
            status = gr.Textbox(label="Status", interactive=False)

    examples_dir = "images"
    content_examples = os.path.join(examples_dir, "content")
    style_examples = os.path.join(examples_dir, "style")
    if os.path.isdir(content_examples) and os.path.isdir(style_examples):
        content_files = sorted(
            os.path.join(content_examples, f)
            for f in os.listdir(content_examples)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )[:4]
        style_files = sorted(
            os.path.join(style_examples, f)
            for f in os.listdir(style_examples)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )[:4]
        if content_files and style_files:
            pairs = [
                [content_files[i % len(content_files)], style_files[i % len(style_files)]]
                for i in range(min(4, max(len(content_files), len(style_files))))
            ]
            gr.Examples(
                examples=pairs,
                inputs=[content_in, style_in],
                label="Example pairs",
            )

    run_btn.click(
        fn=stylize,
        inputs=[content_in, style_in, num_strokes, steps, img_size, canvas_color],
        outputs=[output_img, status],
        show_progress="minimal",
    )


if __name__ == "__main__":
    demo.queue(max_size=4).launch()
