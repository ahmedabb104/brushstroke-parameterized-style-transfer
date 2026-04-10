import argparse
import csv
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.optim as optim
from torchvision.utils import save_image

from deception_score import (
    compute_deception_rate,
    get_artist_labels,
    get_deception_paths,
    load_deception_model,
)
from losses import StyleTransferLosses, curvature_loss, total_variation_loss
from renderer import BrushStrokeRenderer
from utils import image_loader, pick_device


def _parse_int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _infer_target_artist_from_style_filename(style_path: Path) -> str:
    name = style_path.name.lower()
    if "monet" in name:
        return "claude-monet"
    if "cezanne" in name:
        return "paul-cezanne"
    if "starry" in name or "van-gogh" in name or "vangogh" in name or "van_gogh" in name:
        return "vincent-van-gogh"
    if "gauguin" in name:
        return "paul-gauguin"
    if "kandinsky" in name:
        return "wassily-kandinsky"
    return "claude-monet"


def _load_style_manifest(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"style_path", "artist_slug"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(
                f"{path} must contain CSV columns: {sorted(required)}"
            )
        for row in reader:
            rows.append(row)
    return rows


def _collect_content_images(content_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = [p for p in sorted(content_dir.iterdir()) if p.suffix.lower() in exts]
    if not files:
        raise ValueError(f"No content images found in {content_dir}")
    return files


def _run_one_pair(
    device: torch.device,
    content_path: Path,
    style_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    content_img = image_loader(str(content_path), args.img_size, device)
    style_img = image_loader(str(style_path), 224, device)
    _, _, H, W = content_img.shape

    vgg_loss_bs = StyleTransferLosses(
        args.vgg_weights,
        content_img,
        style_img,
        ["conv4_2", "conv5_2"],
        ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"],
        scale_by_y=True,
    )
    vgg_loss_bs.to(device).eval()

    content_np = content_img[0].permute(1, 2, 0).cpu().numpy()
    bs_renderer = BrushStrokeRenderer(
        H,
        W,
        num_strokes=args.num_strokes,
        samples_per_curve=args.samples_per_curve,
        strokes_per_pixel=args.brushes_per_pixel,
        canvas_color=args.canvas_color,
        length_scale=args.length_scale,
        width_scale=args.width_scale,
        content_img=content_np,
    ).to(device)

    optimizer_geom = optim.Adam(
        [
            bs_renderer.location,
            bs_renderer.curve_s,
            bs_renderer.curve_e,
            bs_renderer.curve_c,
            bs_renderer.width,
        ],
        lr=args.lr_geom,
    )
    optimizer_color = optim.Adam([bs_renderer.color], lr=args.lr_color)

    metrics = {}
    for step in range(1, args.steps + 1):
        optimizer_geom.zero_grad()
        optimizer_color.zero_grad()

        canvas = bs_renderer()
        canvas_img = canvas.unsqueeze(0).permute(0, 3, 1, 2).contiguous()

        content_loss, style_loss = vgg_loss_bs(canvas_img)
        content_loss = content_loss * args.content_weight
        style_loss = style_loss * args.style_weight
        tv_loss = args.tv_weight * total_variation_loss(
            bs_renderer.location, bs_renderer.curve_s, bs_renderer.curve_e, K=10
        )
        curv_loss = args.curv_weight * curvature_loss(
            bs_renderer.curve_s, bs_renderer.curve_e, bs_renderer.curve_c
        )
        total_loss = content_loss + style_loss + tv_loss + curv_loss

        total_loss.backward(
            inputs=[
                bs_renderer.location,
                bs_renderer.curve_s,
                bs_renderer.curve_e,
                bs_renderer.curve_c,
                bs_renderer.width,
            ],
            retain_graph=True,
        )
        optimizer_geom.step()

        style_loss.backward(inputs=[bs_renderer.color])
        optimizer_color.step()

        metrics = {
            "content_loss": float(content_loss.item()),
            "style_loss": float(style_loss.item()),
            "tv_loss": float(tv_loss.item()),
            "curvature_loss": float(curv_loss.item()),
            "total_loss": float(total_loss.item()),
        }

<<<<<<< HEAD
        if step % args.save_every == 0 or step == args.steps:
=======
        if args.save_every > 0 and step % args.save_every == 0:
>>>>>>> a062557663a18d8f5a33ff14bcce82878acce3e1
            save_image(canvas_img, output_dir / f"stroke_step_{step:04d}.png")

    with torch.no_grad():
        final_canvas = bs_renderer()
        final_canvas_img = final_canvas.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        save_image(final_canvas_img, output_dir / "brushstroke_result.png")

    # Reconstruction quality against content image.
    mse = torch.mean((final_canvas_img - content_img) ** 2).item()
    metrics["reconstruction_mse"] = float(mse)
    return final_canvas_img.detach(), metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch brushstroke style transfer sweeps.")
    parser.add_argument("--content-dir", default="images/content")
    parser.add_argument("--style-manifest", default="images/style_manifest.csv")
    parser.add_argument("--output-root", default="results/experiments")
    parser.add_argument("--vgg-weights", default="vgg_weights/vgg19_weights_normalized.h5")
    parser.add_argument("--deception-dir", default="deception_score_vgg")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--num-strokes-list", default="1000,2500,5000")
    parser.add_argument("--steps-list", default="50,100,200")
    parser.add_argument("--samples-per-curve", type=int, default=10)
    parser.add_argument("--brushes-per-pixel", type=int, default=20)
    parser.add_argument("--length-scale", type=float, default=1.1)
    parser.add_argument("--width-scale", type=float, default=0.1)
    parser.add_argument("--canvas-color", default="gray")
    parser.add_argument("--content-weight", type=float, default=1.0)
    parser.add_argument("--style-weight", type=float, default=3.0)
    parser.add_argument("--tv-weight", type=float, default=0.008)
    parser.add_argument("--curv-weight", type=float, default=4.0)
    parser.add_argument("--lr-geom", type=float, default=1e-1)
    parser.add_argument("--lr-color", type=float, default=1e-2)
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save intermediate step images every N steps. Use 0 to disable.",
    )
    parser.add_argument("--max-content", type=int, default=10)
    parser.add_argument("--max-styles", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--content",
        default=None,
        metavar="PATH",
        help="Single content image (use with --style) for a one-off run",
    )
    parser.add_argument(
        "--style",
        default=None,
        metavar="PATH",
        help="Single style image (use with --content)",
    )
    parser.add_argument(
        "--target-artist",
        default=None,
        help="Deception slug (e.g. claude-monet). If omitted, inferred from style filename.",
    )
    args = parser.parse_args()

    if (args.content is None) ^ (args.style is None):
        raise SystemExit("Error: --content and --style must be passed together")

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"Using device: {device}")

    if args.content is not None:
        content_path = Path(args.content).expanduser().resolve()
        style_path = Path(args.style).expanduser().resolve()
        if not content_path.is_file():
            raise SystemExit(f"Content image not found: {content_path}")
        if not style_path.is_file():
            raise SystemExit(f"Style image not found: {style_path}")
        target_artist = args.target_artist or _infer_target_artist_from_style_filename(
            style_path
        )
        if args.target_artist is None:
            print(f"Inferred deception target_artist={target_artist} from style filename")
        content_files = [content_path]
        style_rows = [
            {"style_path": str(style_path), "artist_slug": target_artist},
        ]
    else:
        content_files = _collect_content_images(Path(args.content_dir))[: args.max_content]
        style_rows = _load_style_manifest(Path(args.style_manifest))[: args.max_styles]
    stroke_values = _parse_int_list(args.num_strokes_list)
    step_values = _parse_int_list(args.steps_list)

    deception_model = None
    artist_labels = None
    split_hdf5_path = ""
    try:
        _, split_hdf5_path, _ = get_deception_paths(args.deception_dir)
        deception_model = load_deception_model(ckpt_dir=args.deception_dir)
        artist_labels = get_artist_labels(split_hdf5_path)
        print(f"Loaded deception model with {len(artist_labels)} known artists.")
    except FileNotFoundError:
        print("Deception checkpoint/split not found. Skipping deception score.")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "metrics.csv"
    fieldnames = [
        "content_image",
        "style_image",
        "target_artist",
        "num_strokes",
        "steps",
        "runtime_sec",
        "content_loss",
        "style_loss",
        "tv_loss",
        "curvature_loss",
        "total_loss",
        "reconstruction_mse",
        "deception_rate",
        "deception_correct",
        "deception_total",
        "output_dir",
    ]

    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for content_path in content_files:
            for style_row in style_rows:
                style_path = Path(style_row["style_path"])
                target_artist = style_row["artist_slug"]
                if not style_path.exists():
                    print(f"Skipping missing style image: {style_path}")
                    continue

                for num_strokes in stroke_values:
                    for steps in step_values:
                        args.num_strokes = num_strokes
                        args.steps = steps
                        run_name = (
                            f"{content_path.stem}__{style_path.stem}"
                            f"__n{num_strokes}__s{steps}"
                        )
                        run_dir = output_root / run_name
                        run_dir.mkdir(parents=True, exist_ok=True)
                        print(f"Running: {run_name}")
                        t0 = time.time()
                        final_img, run_metrics = _run_one_pair(
                            device=device,
                            content_path=content_path,
                            style_path=style_path,
                            output_dir=run_dir,
                            args=args,
                        )
                        runtime_sec = time.time() - t0

                        deception_rate = ""
                        deception_correct = ""
                        deception_total = ""
                        if deception_model is not None and artist_labels is not None:
                            try:
                                rate, details = compute_deception_rate(
                                    deception_model,
                                    [final_img],
                                    target_artist=target_artist,
                                    split_hdf5_path=split_hdf5_path,
                                    artist_labels=artist_labels,
                                )
                                deception_rate = rate
                                deception_correct = details["correct"]
                                deception_total = details["total"]
                            except ValueError as err:
                                print(f"Deception skip for {run_name}: {err}")

                        writer.writerow(
                            {
                                "content_image": str(content_path),
                                "style_image": str(style_path),
                                "target_artist": target_artist,
                                "num_strokes": num_strokes,
                                "steps": steps,
                                "runtime_sec": runtime_sec,
                                "content_loss": run_metrics["content_loss"],
                                "style_loss": run_metrics["style_loss"],
                                "tv_loss": run_metrics["tv_loss"],
                                "curvature_loss": run_metrics["curvature_loss"],
                                "total_loss": run_metrics["total_loss"],
                                "reconstruction_mse": run_metrics["reconstruction_mse"],
                                "deception_rate": deception_rate,
                                "deception_correct": deception_correct,
                                "deception_total": deception_total,
                                "output_dir": str(run_dir),
                            }
                        )
                        f.flush()

    print(f"Finished. Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
