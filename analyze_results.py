import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate and plot experiment metrics.")
    parser.add_argument("--metrics-csv", default="results/experiments/metrics.csv")
    parser.add_argument("--output-dir", default="results/experiments")
    args = parser.parse_args()

    metrics_path = Path(args.metrics_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_path)
    if df.empty:
        raise ValueError(f"No rows found in {metrics_path}")

    numeric_cols = [
        "num_strokes",
        "steps",
        "runtime_sec",
        "reconstruction_mse",
        "deception_rate",
        "total_loss",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    summary = (
        df.groupby(["num_strokes", "steps"], as_index=False)
        .agg(
            mean_runtime_sec=("runtime_sec", "mean"),
            mean_reconstruction_mse=("reconstruction_mse", "mean"),
            mean_total_loss=("total_loss", "mean"),
            mean_deception_rate=("deception_rate", "mean"),
            num_runs=("output_dir", "count"),
        )
        .sort_values(["num_strokes", "steps"])
    )
    summary.to_csv(out_dir / "aggregate_summary.csv", index=False)
    print(summary.to_string(index=False))

    mse_curve = (
        df.groupby("num_strokes", as_index=False)["reconstruction_mse"]
        .mean()
        .sort_values("num_strokes")
    )
    plt.figure(figsize=(6, 4))
    plt.plot(mse_curve["num_strokes"], mse_curve["reconstruction_mse"], marker="o")
    plt.xlabel("Number of strokes")
    plt.ylabel("Mean reconstruction MSE")
    plt.title("Reconstruction MSE vs Stroke Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "mse_vs_strokes.png", dpi=160)
    plt.close()

    runtime_curve = (
        df.groupby("steps", as_index=False)["runtime_sec"]
        .mean()
        .sort_values("steps")
    )
    plt.figure(figsize=(6, 4))
    plt.plot(runtime_curve["steps"], runtime_curve["runtime_sec"], marker="o")
    plt.xlabel("Optimization steps")
    plt.ylabel("Mean runtime (sec)")
    plt.title("Runtime vs Optimization Steps")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "runtime_vs_steps.png", dpi=160)
    plt.close()

    if "deception_rate" in df.columns and df["deception_rate"].notna().any():
        deception_curve = (
            df.groupby("num_strokes", as_index=False)["deception_rate"]
            .mean()
            .sort_values("num_strokes")
        )
        plt.figure(figsize=(6, 4))
        plt.plot(deception_curve["num_strokes"], deception_curve["deception_rate"], marker="o")
        plt.xlabel("Number of strokes")
        plt.ylabel("Mean deception rate")
        plt.title("Deception Rate vs Stroke Count")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "deception_vs_strokes.png", dpi=160)
        plt.close()

    print(f"Saved analysis artifacts in {out_dir}")


if __name__ == "__main__":
    main()
