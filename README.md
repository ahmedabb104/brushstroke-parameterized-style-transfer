# Brushstroke Parameterized Style Transfer

Reimplementation of Kotovenko et al.: [Rethinking Style Transfer: From Pixels to Parameterized Brushstrokes](https://arxiv.org/pdf/2103.17185)

## Quickstart (Single Run)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Open and run `brushstrokes.ipynb`.

## Proposal-Aligned Batch Experiments

### 1) Set up deception-score assets (~1 GB)


```bash
python3 download_sanakoyeu.py
```

Rename the `evaluation_data/` folder to `deception_score_vgg/`.


### 2) Download starter content/style images

```bash
python3 download_project_images.py
```

### 3) Run parameter sweeps

```bash
python3 run_experiments.py \
  --max-content 10 \
  --max-styles 3 \
  --num-strokes-list "200,1200,2500" \
  --steps-list "100"
```

### 4) Aggregate metrics and plots

```bash
python3 analyze_results.py \
  --metrics-csv results/experiments/metrics.csv \
  --output-dir results/experiments
```

## Outputs

- Per-run images and checkpoints: `results/experiments/*`
- Run-level metrics CSV: `results/experiments/metrics.csv`
- Aggregate summary + plots: `results/experiments/aggregate_summary.csv` and `results/experiments/*.png`