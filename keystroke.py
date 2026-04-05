from __future__ import annotations

import argparse
import csv
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and fixed phrase for live capture
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_CSV = BASE_DIR / "data" / "keystroke_cmu.csv"
TARGET_PHRASE = "keyStroke-Dynamics"

EPS = 1e-9
VERIFIED = "User Verified"
INTRUDER = "Intruder Detected"


def phrase_len() -> int:
    return len(TARGET_PHRASE)


def live_feature_dim() -> int:
    """Live capture: [H1,F1,...,Hn,Fn] with Fn=0 → 2n values."""
    return 2 * phrase_len()


def live_feature_names() -> List[str]:
    n = phrase_len()
    out: List[str] = []
    for i in range(n):
        out.append(f"H_{i + 1}")
        out.append(f"F_{i + 1}")
    return out


# ---------------------------------------------------------------------------
# CMU dataset → numeric matrix
# ---------------------------------------------------------------------------


def load_cmu_csv(path: Path | None = None) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load data/keystroke_cmu.csv: one row per typing repetition, many timing columns.
    Returns X (float matrix), subject id per row, feature column names.
    """
    path = path or DATASET_CSV
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    df = df.dropna()
    meta = {"subject", "sessionIndex", "rep"}
    feat_cols = [c for c in df.columns if c not in meta]
    if not feat_cols:
        raise ValueError("No numeric feature columns in CSV.")

    X = df[feat_cols].to_numpy(dtype=np.float64)
    subjects = df["subject"].astype(str).to_numpy()
    return X, subjects, feat_cols


# ---------------------------------------------------------------------------
# Linear algebra: scaling, covariance PCA, Euclidean score
# ---------------------------------------------------------------------------


@dataclass
class PCAModel:
    scales: np.ndarray
    mu_scaled: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray


def _covariance_ddof1(centered: np.ndarray) -> np.ndarray:
    m, _ = centered.shape
    if m < 2:
        raise ValueError("Need at least 2 rows for covariance.")
    return (centered.T @ centered) / (m - 1)


def fit_pca(X: np.ndarray, n_components: int) -> PCAModel:
    """Scale columns by 1/σ, center, C = (1/(m-1)) DᵀD, top-k eigenvectors."""
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    m, d_dim = X.shape
    if m < 2:
        raise ValueError("Need at least 2 training samples.")

    sigma = X.std(axis=0, ddof=1)
    sigma = np.where(sigma < EPS, EPS, sigma)
    scales = 1.0 / sigma
    X_scaled = X * scales
    mu_scaled = X_scaled.mean(axis=0)
    d_centered = X_scaled - mu_scaled
    c_mat = _covariance_ddof1(d_centered)

    evals, evecs = np.linalg.eigh(c_mat)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    pos_rank = int(np.sum(evals > 1e-12))
    k = min(n_components, d_dim, m - 1, pos_rank if pos_rank > 0 else d_dim)
    k = max(k, 1)
    components = evecs[:, :k]
    chosen = evals[:k]
    total = np.sum(evals[evals > 0])
    ratio = chosen / total if total > 0 else np.ones(k) / k

    return PCAModel(scales, mu_scaled, components, chosen, ratio)


def project(model: PCAModel, x_raw: np.ndarray) -> np.ndarray:
    x = np.asarray(x_raw, dtype=np.float64).reshape(-1)
    if x.shape[0] != model.scales.shape[0]:
        raise ValueError("Feature length does not match model.")
    xs = x * model.scales
    return model.components.T @ (xs - model.mu_scaled)


def training_distances(model: PCAModel, X: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.stack([project(model, row) for row in X]), axis=1)


@dataclass
class VerificationModel:
    pca: PCAModel
    threshold: float
    user_label: str
    source: str  # e.g. "cmu_csv" or "live_typing"


def suggest_threshold(X_train: np.ndarray, model: PCAModel, percentile: float = 92.0, margin: float = 1.25) -> float:
    dists = training_distances(model, X_train)
    base = float(np.percentile(dists, percentile))
    return max(base * margin, 1e-6)


def fit_verifier(
    X_train: np.ndarray,
    user_label: str,
    source: str,
    n_components: int = 8,
    percentile: float = 92.0,
    margin: float = 1.25,
) -> VerificationModel:
    pca = fit_pca(X_train, n_components=n_components)
    thr = suggest_threshold(X_train, pca, percentile=percentile, margin=margin)
    return VerificationModel(pca=pca, threshold=thr, user_label=user_label, source=source)


def _distances_raw(pca: PCAModel, X: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.stack([project(pca, row) for row in X]), axis=1)


def subspace_norms(pca: PCAModel, X: np.ndarray) -> np.ndarray:
    """‖Pᵀ(x_scaled − μ)‖ per row (verification score before thresholding)."""
    return _distances_raw(pca, X)


def threshold_from_holdout(
    pca: PCAModel,
    X_train: np.ndarray,
    X_genuine_holdout: np.ndarray,
    X_impostor_holdout: np.ndarray,
) -> float:
    """
    Pick a distance threshold that balances genuine acceptance vs impostor rejection
    on the hold-out rows (used only for the bundled dataset demo).
    """
    d_tr = training_distances(pca, X_train)
    gen_d = _distances_raw(pca, X_genuine_holdout)
    imp_d = _distances_raw(pca, X_impostor_holdout)

    lo = float(min(gen_d.min(), imp_d.min(), d_tr.min()))
    hi = float(max(gen_d.max(), imp_d.max(), d_tr.max()))
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(lo, hi, num=500),
                np.percentile(d_tr, np.linspace(50.0, 99.9, 40)),
            ]
        )
    )

    best_t = float(np.median(d_tr))
    best_min = -1.0
    for t in candidates:
        if t <= 0:
            continue
        g = float(np.mean(gen_d < t))
        r = float(np.mean(imp_d > t))
        m = min(g, r)
        if m > best_min:
            best_min = m
            best_t = float(t)
    return max(best_t, 1e-6)


def verify_row(vm: VerificationModel, x: np.ndarray) -> Tuple[str, float]:
    d = float(np.linalg.norm(project(vm.pca, x)))
    return (VERIFIED if d < vm.threshold else INTRUDER), d


def save_model(path: Path, vm: VerificationModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        scales=vm.pca.scales,
        mu_scaled=vm.pca.mu_scaled,
        components=vm.pca.components,
        eigenvalues=vm.pca.eigenvalues,
        explained_variance_ratio=vm.pca.explained_variance_ratio,
        threshold=np.array([vm.threshold]),
        user_label=np.array(vm.user_label),
        source=np.array(vm.source),
    )


def load_model(path: Path) -> VerificationModel:
    z = np.load(path, allow_pickle=True)
    pca = PCAModel(
        scales=np.asarray(z["scales"]),
        mu_scaled=np.asarray(z["mu_scaled"]),
        components=np.asarray(z["components"]),
        eigenvalues=np.asarray(z["eigenvalues"]),
        explained_variance_ratio=np.asarray(z["explained_variance_ratio"]),
    )
    thr = float(np.asarray(z["threshold"]).reshape(-1)[0])
    ulab = str(np.asarray(z["user_label"]).reshape(-1)[0])
    src = str(np.asarray(z["source"]).reshape(-1)[0])
    return VerificationModel(pca=pca, threshold=thr, user_label=ulab, source=src)


# ---------------------------------------------------------------------------
# Demo: one genuine subject, holdout genuine + impostor trials
# ---------------------------------------------------------------------------


def run_dataset_demo(
    csv_path: Path | None = None,
    genuine_subject: str | None = None,
    n_components: int = 12,
    train_fraction: float = 0.65,
    max_impostor_test: int = 400,
    seed: int = 42,
    plot_path: Path | None = None,
) -> dict:
    """
    Uses columns from data/keystroke_cmu.csv (timing features per row).
    Picks a genuine subject (default: first available in a small preferred list, else the busiest id).
    """
    X, subjects, feat_cols = load_cmu_csv(csv_path)

    if genuine_subject is None:
        counts = pd.Series(subjects).value_counts()
        preferred_order = ["s057", "s032", "s002"]
        genuine_subject = None
        for cand in preferred_order:
            if cand in counts.index and int(counts[cand]) >= 40:
                genuine_subject = cand
                break
        if genuine_subject is None:
            genuine_subject = str(counts.idxmax())

    mask_g = subjects == genuine_subject
    if int(mask_g.sum()) < 8:
        raise ValueError(f"Subject {genuine_subject!r} has too few rows; pick another.")

    Xg = X[mask_g]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(Xg))
    n_train = max(5, int(train_fraction * len(Xg)))
    tr, te = order[:n_train], order[n_train:]
    if len(te) < 2:
        te = tr[-2:]
        tr = tr[:-2]

    X_train = Xg[tr]
    X_genuine_test = Xg[te]

    X_other = X[subjects != genuine_subject]
    n_imp = min(max_impostor_test, len(X_other))
    imp_pick = rng.choice(len(X_other), size=n_imp, replace=False)
    X_imp = X_other[imp_pick]

    k = min(n_components, X_train.shape[1], X_train.shape[0] - 1)
    pca = fit_pca(X_train, n_components=k)
    thr = threshold_from_holdout(pca, X_train, X_genuine_test, X_imp)
    vm = VerificationModel(
        pca=pca,
        threshold=thr,
        user_label=genuine_subject,
        source="cmu_csv",
    )

    gen_ok = sum(verify_row(vm, row)[0] == VERIFIED for row in X_genuine_test)
    imp_rej = sum(verify_row(vm, row)[0] == INTRUDER for row in X_imp)

    gen_sub = _distances_raw(vm.pca, X_genuine_test)
    imp_sub = _distances_raw(vm.pca, X_imp)
    train_sub = training_distances(vm.pca, X_train)

    g_rate = gen_ok / max(len(X_genuine_test), 1)
    i_rate = imp_rej / max(len(X_imp), 1)
    print("=== Keystroke demo ===")
    print(f"CSV: {csv_path or DATASET_CSV}")
    print(f"Features per row: {len(feat_cols)}")
    print(f"Genuine subject: {genuine_subject}")
    print(f"Train rows (genuine): {len(X_train)}  |  PCA k: {vm.pca.components.shape[1]}")
    print(f"Threshold (PCA distance): {vm.threshold:.4f}")
    print(f"Genuine test: accepted {gen_ok}/{len(X_genuine_test)}")
    print(f"Impostor test: rejected {imp_rej}/{len(X_imp)}")
    print(
        "Strong separation on this split."
        if min(g_rate, i_rate) >= 0.72
        else "Lower rates on this split; adjust --subject or --components."
    )

    if plot_path is not None:
        try:
            import matplotlib.pyplot as plt

            vals = vm.pca.eigenvalues
            fig, ax = plt.subplots(figsize=(6, 3.2))
            ax.plot(np.arange(1, len(vals) + 1), vals, "o-", color="tab:blue")
            ax.set_xlabel("Eigenvector index (by λ)")
            ax.set_ylabel("Eigenvalue")
            ax.set_title(f"PCA spectrum — subject {genuine_subject}")
            fig.tight_layout()
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=120)
            plt.close(fig)
            print(f"Saved plot: {plot_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"(Could not save plot: {exc})")

    return {
        "genuine_subject": genuine_subject,
        "n_features": len(feat_cols),
        "train_n": len(X_train),
        "genuine_accept": gen_ok,
        "genuine_total": len(X_genuine_test),
        "impostor_reject": imp_rej,
        "impostor_total": len(X_imp),
        "threshold": vm.threshold,
        "model": vm,
        "genuine_subspace_distances": gen_sub,
        "impostor_subspace_distances": imp_sub,
        "train_subspace_distances": train_sub,
    }


# ---------------------------------------------------------------------------
# Live typing (optional — requires pynput)
# ---------------------------------------------------------------------------


def _events_to_vector(press_s: np.ndarray, release_s: np.ndarray) -> np.ndarray:
    n = phrase_len()
    h = release_s - press_s
    f = np.zeros(n, dtype=np.float64)
    for i in range(n - 1):
        f[i] = press_s[i + 1] - release_s[i]
    out = np.empty(2 * n, dtype=np.float64)
    out[0::2] = h
    out[1::2] = f
    return out


def _char_from_key(key) -> Optional[str]:
    try:
        if hasattr(key, "char") and key.char is not None:
            return key.char
    except AttributeError:
        pass
    return None


# Hyphens / minus signs that should count as ASCII '-' for the phrase
_DASH_LIKE = frozenset(
    {
        "\u002d",  # hyphen-minus
        "\u2010",  # hyphen
        "\u2011",  # non-breaking hyphen
        "\u2012",  # figure dash
        "\u2013",  # en dash
        "\u2014",  # em dash
        "\u2015",  # horizontal bar
        "\u2212",  # minus sign
        "\ufe58",  # small em dash
        "\ufe63",  # small hyphen-minus
        "\uff0d",  # fullwidth hyphen-minus
    }
)


def _normalize_key_char(ch: str) -> str:
    """Map look-alike Unicode to the character we compare against TARGET_PHRASE."""
    if not ch:
        return ch
    ch = unicodedata.normalize("NFKC", ch)
    if ch in _DASH_LIKE:
        return "-"
    return ch


def _chars_match(typed: str, expected: str) -> bool:
    """Compare after normalization; letters are case-insensitive."""
    nt = _normalize_key_char(typed)
    ne = _normalize_key_char(expected)
    if nt == ne:
        return True
    if len(nt) == 1 and len(ne) == 1 and nt.isalpha() and ne.isalpha():
        return nt.lower() == ne.lower()
    return False


def _keys_correspond(press_key, release_key) -> bool:
    """
    macOS/Linux often omit key.char on key-up; match the physical key via vk or equality.
    """
    if press_key is None:
        return False
    if press_key == release_key:
        return True
    try:
        from pynput.keyboard import KeyCode

        if isinstance(press_key, KeyCode) and isinstance(release_key, KeyCode):
            v1, v2 = press_key.vk, release_key.vk
            if v1 is not None and v2 is not None and int(v1) == int(v2):
                return True
    except (AttributeError, TypeError, ValueError):
        pass
    return False


class LiveCollector:
    """Record dwell/flight times for TARGET_PHRASE; Enter saves, ESC cancels, Backspace edits."""

    def __init__(self, on_status: Optional[Callable[[str], None]] = None) -> None:
        self._on_status = on_status or (lambda _m: None)
        self._expected = list(TARGET_PHRASE)
        self._n = len(self._expected)
        self._press = np.zeros(self._n, dtype=np.float64)
        self._release = np.zeros(self._n, dtype=np.float64)
        self._pos = 0
        self._active_idx: Optional[int] = None
        self._press_key_for_active = None
        self._vector: Optional[np.ndarray] = None
        self._listener = None

    def _progress_bar(self) -> str:
        p = self._pos
        n = self._n
        return "[" + ("█" * p) + ("░" * (n - p)) + f"]  {p}/{n}"

    def _abort_current_key(self) -> None:
        if self._active_idx is None:
            return
        self._press[self._active_idx] = 0.0
        self._active_idx = None
        self._press_key_for_active = None

    def _feedback_wrong(self, got_ch: Optional[str]) -> None:
        need = self._expected[self._pos]
        got = repr(got_ch) if got_ch else "modifier / special key (ignored)"
        hint = ""
        if need == "-":
            hint = "  (use the key next to 0, or any standard hyphen/minus)"
        elif need == "_":
            hint = "  (Shift + hyphen)"
        self._on_status(
            f"{self._progress_bar()}\n"
            f"Expected {repr(need)} next — you typed {got}.{hint}\n"
            "Nothing was erased; try again."
        )

    def _finalize(self) -> None:
        self._vector = _events_to_vector(self._press, self._release)
        self._on_status("Saved.")
        if self._listener is not None:
            self._listener.stop()

    def _handle_backspace(self) -> None:
        if self._active_idx is not None:
            self._press[self._active_idx] = 0.0
            self._active_idx = None
            self._press_key_for_active = None
            self._on_status(
                f"{self._progress_bar()}\nCancelled key-in-progress — type {repr(self._expected[self._pos])} again."
            )
            return
        if self._pos == self._n:
            self._pos = self._n - 1
            self._press[self._pos] = 0.0
            self._release[self._pos] = 0.0
            self._on_status("Editing last character — retype it.")
            return
        if self._pos > 0:
            self._pos -= 1
            self._press[self._pos] = 0.0
            self._release[self._pos] = 0.0
            self._on_status(f"Backspace: retype from character {self._pos + 1} / {self._n}.")

    def _on_press(self, key) -> None:
        from pynput import keyboard as kb

        if key == kb.Key.enter:
            if self._pos == self._n:
                self._finalize()
            return
        if key == kb.Key.esc:
            self._vector = None
            self._on_status("Cancelled.")
            if self._listener is not None:
                self._listener.stop()
            return
        if key == kb.Key.backspace:
            self._handle_backspace()
            return

        ch = _char_from_key(key)
        if ch is None or self._pos >= self._n:
            return

        # Waiting for key-up: ignore OS key-repeat (same character) or abandon if user hit another key
        if self._active_idx is not None:
            if self._active_idx == self._pos and _chars_match(ch, self._expected[self._pos]):
                return
            self._abort_current_key()
            self._on_status(
                f"{self._progress_bar()}\n"
                "New key before the last one was released — timing for that letter reset. "
                f"Now type {repr(self._expected[self._pos])}."
            )

        if not _chars_match(ch, self._expected[self._pos]):
            self._feedback_wrong(ch)
            return

        self._press[self._pos] = time.perf_counter()
        self._active_idx = self._pos
        self._press_key_for_active = key
        self._on_status(f"{self._progress_bar()}  Key down — now release it.")

    def _on_release(self, key) -> None:
        from pynput import keyboard as kb

        if key in (kb.Key.enter, kb.Key.esc, kb.Key.backspace):
            return

        if self._active_idx is None:
            return
        idx = self._active_idx
        if idx != self._pos:
            return

        ch_rel = _char_from_key(key)
        same_key = _keys_correspond(self._press_key_for_active, key)
        char_ok = ch_rel is not None and _chars_match(ch_rel, self._expected[idx])
        if not (same_key or char_ok):
            return

        self._release[idx] = time.perf_counter()
        self._active_idx = None
        self._press_key_for_active = None
        self._pos += 1
        if self._pos == self._n:
            self._on_status(
                f"{self._progress_bar()}\n"
                "Phrase complete — press ENTER to save (ESC = cancel)."
            )
        else:
            self._on_status(
                f"{self._progress_bar()}\n"
                f"Next character: {repr(self._expected[self._pos])}"
            )

    def collect_one(self) -> Optional[np.ndarray]:
        from pynput import keyboard  # noqa: WPS433

        self._press.fill(0.0)
        self._release.fill(0.0)
        self._pos = 0
        self._active_idx = None
        self._press_key_for_active = None
        self._vector = None
        self._on_status(
            "Live capture — a wrong key does not reset progress.\n"
            f"Phrase:  {TARGET_PHRASE!r}\n"
            f"First key:  {repr(self._expected[0])}\n\n"
            "• Wrong key → message only; keep going from the same spot.\n"
            "• Backspace → undo one finished letter (or cancel the key you are holding).\n"
            "• Letters: upper or lower case both work.\n"
            "• Hyphen - : use the minus key; en/em dash also accepted.\n"
            "• Finish the line, then press ENTER to save (ESC = cancel).\n"
            "• Keys also reach the terminal (you can type normally); the phrase may echo in the shell.\n"
            "• macOS: grant Accessibility to the app that runs this terminal if keys do nothing."
        )
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=False,
        )
        self._listener.start()
        self._listener.join()
        return self._vector


def _append_live_csv(path: Path, user_id: str, attempt: int, vector: np.ndarray) -> None:
    names = ["user_id", "attempt"] + live_feature_names()
    new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(names)
        w.writerow([user_id, attempt] + [f"{v:.6f}" for v in vector.tolist()])


def _load_live_csv(path: Path) -> tuple[np.ndarray, List[str]]:
    df = pd.read_csv(path)
    cols = live_feature_names()
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c!r}")
    users = df["user_id"].astype(str).tolist()
    return df[cols].to_numpy(dtype=np.float64), users


def _ensure_pynput() -> bool:
    try:
        import pynput  # noqa: F401
    except ImportError:
        return False
    return True


def cmd_live_enroll(args: argparse.Namespace) -> int:
    if not _ensure_pynput():
        print("Install pynput: pip install pynput", file=sys.stderr)
        return 1

    csv_path = BASE_DIR / "data" / f"typing_{args.user}.csv"
    if csv_path.exists() and not args.append:
        csv_path.unlink()

    start = 0
    if args.append and csv_path.exists():
        start = int(pd.read_csv(csv_path)["attempt"].max())

    print(f"Enrollment phrase: {TARGET_PHRASE!r}\n")
    print(f"Phrase length {phrase_len()} → {live_feature_dim()} features.")
    got = 0
    attempt = start
    while got < args.reps:
        col = LiveCollector(on_status=print)
        vec = col.collect_one()
        if vec is None:
            print("Skipped.")
            continue
        attempt += 1
        got += 1
        _append_live_csv(csv_path, args.user, attempt, vec)
        print(f"Stored attempt {got}/{args.reps}.\n")

    X, users = _load_live_csv(csv_path)
    X = X[np.array(users) == args.user]
    if len(X) < 2:
        print("Need at least 2 samples.", file=sys.stderr)
        return 1

    k = min(args.components, X.shape[0] - 1, X.shape[1])
    vm = fit_verifier(X, user_label=args.user, source="live_typing", n_components=k)
    model_path = BASE_DIR / "data" / f"model_{args.user}.npz"
    save_model(model_path, vm)
    print(f"Model saved: {model_path}")
    return 0


def cmd_live_verify(args: argparse.Namespace) -> int:
    if not _ensure_pynput():
        print("Install pynput: pip install pynput", file=sys.stderr)
        return 1

    model_path = BASE_DIR / "data" / f"model_{args.user}.npz"
    if not model_path.is_file():
        print(f"No model at {model_path}", file=sys.stderr)
        return 1

    vm = load_model(model_path)
    print(f"Verify as {vm.user_label} (threshold {vm.threshold:.4f})")
    vec = LiveCollector(on_status=print).collect_one()
    if vec is None:
        return 1
    label, dist = verify_row(vm, vec)
    print(f"Distance: {dist:.4f} → {label}")
    return 0 if label == VERIFIED else 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Keystroke PCA verification")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Override path to keystroke CSV (default: data/keystroke_cmu.csv)",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Genuine subject id in CSV (default: subject with most rows)",
    )
    parser.add_argument("--components", type=int, default=12, help="PCA subspace size k")
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Save eigenvalue plot to this path (e.g. data/spectrum.png)",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=None,
        metavar="DIR",
        help="Save full report figure set into this directory (PNG files)",
    )
    sub = parser.add_subparsers(dest="cmd", required=False)

    p_enroll = sub.add_parser("live-enroll", help="Record live samples (needs pynput)")
    p_enroll.add_argument("--user", required=True)
    p_enroll.add_argument("--reps", type=int, default=12)
    p_enroll.add_argument("--append", action="store_true")

    p_ver = sub.add_parser("live-verify", help="One live trial (needs pynput)")
    p_ver.add_argument("--user", required=True)

    args = parser.parse_args(argv)

    if args.cmd == "live-enroll":
        return cmd_live_enroll(args)
    if args.cmd == "live-verify":
        return cmd_live_verify(args)

    summary = run_dataset_demo(
        csv_path=args.dataset,
        genuine_subject=args.subject,
        n_components=args.components,
        plot_path=args.plot,
    )
    if args.figures is not None:
        from plots import save_report_figures

        paths = save_report_figures(summary, args.figures)
        print("Report figures:")
        for p in paths:
            print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
