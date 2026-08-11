# Working links for FuzzyFoxes
[Google Slides for Dev](https://docs.google.com/presentation/d/1QHsxICGASqcS6CoNdJNOBJAiXddZ59xZPVVH4TBtfyQ/edit?usp=sharing). 

  - Please write down your name and what you are currently trying to do and how is it working.
  - Everytime you try something different, comment it verbally and see if someone else has tried that before.

# Current best sinble backbone probe model
python -m student.train_probe \
  --features-dir features/dinov2_vitg14_reg4_518 \
  --backbone vit_giant_patch14_reg4_dinov2 \
  --img-size 518 --pooling cls \
  --brier-weight 0.5 \
  --epochs 200 --lr 1e-3 --weight-decay 1e-4 --patience 20 \
  --out-dir runs/dinov2_vitg14_518_loss_bw05

  Extract features from the backbone using the following command:
  python -m student.extract_features \
  --data-root challenge_data \
  --backbone vit_giant_patch14_reg4_dinov2 \
  --img-size 518 \
  --out-dir features/dinov2_vitg14_reg4_518 \
  --batch-size 128 --num-workers 8 --gpu 0

# Summer School Challenge 2026 — Predictive Uncertainty on iWildCam
Welcome to the Predictive Uncertainty Challenge 2026, where deep learning meets the wild… and occasionally gets very unsure about what it just saw.
In this year’s summer school challenge, you’re not just building a model that says “this is a zebra”—you’re building a model that can confidently say “this is a zebra… and I’m 93% sure.” Because in the real world, confidence matters just as much as correctness.
So grab your GPU, sharpen your softmax, and prepare to step into the untamed wilderness of machine learning—where images are messy, and distributions shifts and overconfidence is the real predator.

This challenge is made in conjunction with the [Ph. D. school on uncertainty in machine learning](https://uncertainty.compute.dtu.dk/).

You are given a real-world wildlife dataset from camera traps (yes—actual motion-triggered cameras lurking in nature like very patient photographers). Your job is to:
- Train a classifier that recognizes animal species
- Quantify how confident your model should be
- Avoid the classic trap: being confidently wrong

In short: don’t just predict—predict responsibly.

## Why does uncertainty matter?
In controlled benchmarks, models often look impressively confident. But in the wild—literally—they can fall apart:
- New locations
- New lighting conditions
- New species combinations
- And occasionally… something that looks like a raccoon but isn’t

In real-world applications (ecology, medicine, autonomous systems), knowing when you don’t know is critical. A model that is slightly less accurate—but well-calibrated—can be far more valuable than one that is wildly overconfident.
This challenge is about building models that don’t just perform, but also understand their limits.

## What you will do during the challenge
- **Tame the baseline:** Start from a ResNet-50 and see how far you can push it.
- **Calibrate like a pro:** Make your probabilities trustworthy—not just numerically convenient.
- **Embrace the unknown:** Handle domain shifts (new camera locations, lighting, environments).
- **Battle the metrics:** It’s not just about accuracy—your model will be judged on:
  - Accuracy (obviously)
  - Calibration (ECE)
  - Negative log-likelihood (NLL)
  - Brier score
  - Misclassification AUROC (a fancy way of asking: does your model know when it’s wrong?)
- **Climb the leaderboard:** Scores are combined via Borda count, so consistency beats one-trick ponies.

## The dataset: iWildCam
![plot](iwildcam.jpg)
*Photo credit: [iWildCam challenge](https://github.com/visipedia/iwildcam_comp/tree/master)*

You’ll be working with images collected from camera traps around the world. Some environments you’ve seen before (in-distribution), and some are completely new (out-of-distribution—surprise!).
Animals may appear:
- Centered… or as a tiny blur in the corner
- In daylight… or in mysterious infrared night mode
- Alone… or photobombed
Nature doesn’t care about your training distribution—and now neither does this challenge.

## Preparing for the challenge
Please download the challenge image data before the challenge http://fungi.compute.dtu.dk:8080/login
Place the data on a cluster that you will have access to during the challenge.

Unzip the release so the layout is:

```
challenge_data/
├── train/
│   ├── images/<uid>.jpg
│   └── labels.csv
├── val/
│   ├── images/<uid>.jpg
│   └── labels.csv
├── test_public/
│   └── images/<uid>.jpg
├── test_private/
│   └── images/<uid>.jpg
├── sample_submission.csv
└── class_mapping.json
```

- `train/labels.csv` and `val/labels.csv` have columns `uid, y, domain`.
- `val`'s `domain` is `id` (same camera locations as train, different days) or `ood` (held-out locations). Useful for split-conditional analysis.
- `test_public/` is the live leaderboard set; `test_private/` is the   held-out final-score set. **Neither has labels** — predict on both and submit a single CSV. The server uses `test_public` rows for the leaderboard and `test_private` rows for final scoring.
- `sample_submission.csv` covers every uid across both test splits with uniform `1/K` probabilities — a working format reference.
- `class_mapping.json` carries `num_classes` (`K`) and the mapping between contiguous `[0, K-1]` labels and the original WILDS class ids.

## How to get started
You are given a starter codebase, a labeled training and validation set, and an unlabeled `test_public` set. You will train a classifier, produce a `submission.csv`, and upload it. The server scores your submission on four metrics; the leaderboard aggregates them via Borda count.

### 1. Install

```bash
pip install -r requirements.txt
```

This installs `torch`, `torchvision`, `pandas`, `numpy`, `pillow`, `tqdm`.
No `wilds`, no `scikit-learn` — the local evaluator is self-contained.


### 2. Train your baseline

```bash
python -m student.train \
    --data-root /path/to/challenge_data \
    --output-dir runs/baseline \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-4 \
    --head-lr 1e-3 \
    --patience 3
```

This runs ResNet-50 with AdamW (lower LR on the pretrained backbone, higher on the new classification head), a cosine LR schedule, and early stopping on val NLL. After training, a single-scalar temperature `T` is fit on `val` via LBFGS on NLL. Two checkpoints are written:

- `runs/baseline/model.pt`              — early-stopped best weights (`T = 1.0`)
- `runs/baseline/model_temp_scaled.pt`  — same weights, with the learned `T`

Both checkpoints share the same format — a dict containing `state_dict`,
`num_classes`, `temperature`, `backbone` (timm model id), and the resolved
`hyperparameters` dict. Pass `--backbone <timm-id>` to swap (e.g.
`--backbone resnet18` or `--backbone vit_base_patch16_224`); pass
`--pretrained` to start from timm's published weights.

### 3. Evaluate locally

```bash
python -m student.eval \
    --checkpoint runs/baseline/model_temp_scaled.pt \
    --data-root /path/to/challenge_data
```

Prints the same five numbers the server will compute:

- `accuracy`
- `ece` (15 equal-width bins on max-softmax)
- `nll` (mean negative log-likelihood)
- `brier` (mean Brier score)
- `misclassification_auroc` (score = max-softmax confidence, target = correct)

The local metrics are bit-for-bit equivalent to the server. Use them to iterate before submitting.

### 4. Generate a submission

```bash
python -m student.predict \
    --checkpoint runs/baseline/model_temp_scaled.pt \
    --data-root /path/to/challenge_data \
    --output submission.csv
```

### 5. Submission format

**One row per `uid` across `test_public/images/` and `test_private/images/`.**
Columns:

```
uid, p_0, p_1, ..., p_{K-1}
```

- `p_k` is the predicted probability of class `k` (the remapped contiguous
  label space — see `class_mapping.json`).
- Probabilities must sum to ~1 per row (tolerance `1e-3`).
- Row order does not matter; the server joins on `uid`.
- One combined submission for both splits — `student.predict` does this by default. The server uses `test_public` uids for the live leaderboard and `test_private` uids for final scoring.

`challenge_data/sample_submission.csv` is a working template with uniform
`1/K` probabilities — it will be accepted by the server (just very badly
scored).

## Survival tips for the Wild
Concrete suggestions for starting points:
- **Temperature scaling** is already in the baseline; try alternative calibration objectives (Brier instead of NLL).
- **Deep ensembles** — train N models, average their probabilities.
- **MC dropout** — enable dropout at eval time, average over T forward passes.
- **Class re-weighting / focal loss** — iWildCam is long-tailed.

Most experimentation lives in `student/model.py` — the `Classifier` class exposes `self.backbone`, `self.head`, and an `embed(x)` method, so you can
swap backbones, add projection heads, or override `forward` for MC sampling without touching `train.py`. Keep `forward(x) → logits` and the `backbone / head` split, and the rest of the pipeline still works.

Don't change the metric formulas in `student/metrics.py` — the server uses the same ones.
