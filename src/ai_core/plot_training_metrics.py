import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd


def resolve_metrics_path(metrics_path: str | None) -> str:
    if metrics_path:
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
        return metrics_path

    candidates = sorted(glob.glob(os.path.join("models", "training_metrics_*.csv")))
    if not candidates:
        raise FileNotFoundError("No training_metrics_*.csv found in models/")
    return candidates[-1]


def summarize(df: pd.DataFrame) -> dict:
    best_idx = df["val_total"].idxmin()
    best_row = df.loc[best_idx]
    final_row = df.iloc[-1]

    return {
        "best_epoch": int(best_row["epoch"]),
        "best_val_total": float(best_row["val_total"]),
        "final_train_total": float(final_row["train_total"]),
        "final_val_total": float(final_row["val_total"]),
        "final_val_gap": float(final_row["val_gap"]),
    }


def risk_label(final_gap: float) -> str:
    if final_gap <= 0.05:
        return "LOW"
    if final_gap <= 0.15:
        return "MEDIUM"
    return "HIGH"


def plot_metrics(metrics_path: str, output_path: str | None = None) -> str:
    df = pd.read_csv(metrics_path)

    required_cols = {"epoch", "train_total", "val_total", "val_gap"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in metrics CSV: {sorted(missing)}")

    if output_path is None:
        stem = os.path.splitext(os.path.basename(metrics_path))[0].replace("training_metrics", "training_plot")
        output_path = os.path.join("models", f"{stem}.png")

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(df["epoch"], df["train_total"], label="Train Total Loss", linewidth=2)
    ax1.plot(df["epoch"], df["val_total"], label="Validation Total Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total Loss")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["val_gap"], "--", label="Validation Gap", linewidth=1.5)
    ax2.set_ylabel("Val Gap")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    summary = summarize(df)
    title = (
        f"CVAE Training vs Validation | Best Epoch: {summary['best_epoch']} | "
        f"Final Gap: {summary['final_val_gap']:.4f} ({risk_label(summary['final_val_gap'])})"
    )
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Metrics source: {metrics_path}")
    print(f"Plot saved to: {output_path}")
    print(
        "Summary | "
        f"best_epoch={summary['best_epoch']} "
        f"best_val_total={summary['best_val_total']:.4f} "
        f"final_train_total={summary['final_train_total']:.4f} "
        f"final_val_total={summary['final_val_total']:.4f} "
        f"final_val_gap={summary['final_val_gap']:.4f} "
        f"risk={risk_label(summary['final_val_gap'])}"
    )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CVAE training and validation metrics.")
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Path to a training_metrics_*.csv file. If omitted, latest in models/ is used.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path. If omitted, saved in models/ with a timestamped name.",
    )
    args = parser.parse_args()

    metrics_path = resolve_metrics_path(args.metrics)
    plot_metrics(metrics_path, args.out)


if __name__ == "__main__":
    main()
