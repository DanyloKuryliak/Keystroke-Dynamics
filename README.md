# Keystroke dynamics — user verification with PCA

Keystroke timing features → column scaling → PCA on the covariance matrix → Euclidean distance in the projected subspace → accept if below a threshold τ.

## Setup

```bash
python -m venv .venv
```

Activate the venv (Windows: `.venv\Scripts\activate` — PowerShell: `Activate.ps1` — macOS/Linux: `source .venv/bin/activate`), then:

```bash
pip install -r requirements.txt
```

## Layout

| Path | Role |
|------|------|
| `keystroke.py` | Implementation and CLI |
| `plots.py` | Report figures from the experiment summary |
| `data/keystroke_cmu.csv` | Timing dataset (one row per repetition) |
| `keystroke_dynamics.ipynb` | Report notebook (runs the same code, shows figures) |
| `output/figures/` | Created when you export figures (see below) |

## Commands

**Run the experiment on the CSV (no keyboard):**

```bash
python keystroke.py
```

**Export seven PNG figures for the report:**

```bash
python keystroke.py --figures output/figures
```

**Optional single spectrum plot:**

```bash
python keystroke.py --plot output/figures/spectrum_only.png
```

**Live capture**

```bash
python keystroke.py live-enroll --user alice --reps 10
python keystroke.py live-verify --user alice
```

Live samples: `data/typing_<user>.csv`. Model: `data/model_<user>.npz`.

During live capture, keys still go to the terminal (the phrase may appear in the shell). **macOS:** grant **Accessibility** (and **Input Monitoring** if listed) to the application that runs Python (Terminal, iTerm, or the integrated terminal in your editor).

The phrase typed in live mode is `TARGET_PHRASE` at the top of `keystroke.py`. The CMU CSV uses its own feature columns (31 timing fields per row).

## Notebook

Run all cells in `keystroke_dynamics.ipynb` to reproduce numbers and refresh `output/figures/`.
