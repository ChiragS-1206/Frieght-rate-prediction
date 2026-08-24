from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_ROWS = 12000
EXPECTED_IDS = {f"TE-{index:06d}" for index in range(1, EXPECTED_ROWS + 1)}

DECEMBER_DATES = pd.date_range("2025-12-01", "2025-12-31", freq="D")

FIXED_PICKUP = "Lexington"
FIXED_DELIVERY = "Fort Wayne"
FIXED_DISTANCE = 360.0
FIXED_EQUIPMENT = "Dry Van"
FIXED_WEIGHT = 32000.0


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        fail(f"{label} file not found: {path}")
    try:
        return pd.read_csv(path)
    except Exception as exc:
        fail(f"could not read {label}: {exc}")


def numeric_series(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all():
        fail(f"{label} contains invalid {column} values")
    return values.astype(float)


def validate_predictions(predictions: pd.DataFrame) -> None:
    if list(predictions.columns) != ["load_id", "predicted_rate"]:
        fail("predictions must contain exactly two columns: load_id,predicted_rate")

    if len(predictions) != EXPECTED_ROWS:
        fail(f"predictions must contain exactly {EXPECTED_ROWS} rows")

    if predictions["load_id"].isna().any() or predictions["load_id"].duplicated().any():
        fail("predictions contains missing or duplicate load_id values")

    submitted_ids = set(predictions["load_id"].astype(str))
    missing = EXPECTED_IDS - submitted_ids
    extra = submitted_ids - EXPECTED_IDS

    if missing or extra:
        fail(
            "prediction IDs do not match validation set "
            f"(missing={len(missing)}, extra={len(extra)})"
        )

    predicted_rate = numeric_series(predictions, "predicted_rate", "predictions")

    if (predicted_rate <= 0).any():
        fail("predicted_rate must be positive")


def validate_december(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]

    if list(frame.columns) != columns:
        fail("December file must have exact columns and order")

    result = frame.copy()

    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        fail("invalid dates in December file")

    result["distance"] = numeric_series(result, "distance", "December file")
    result["weight"] = numeric_series(result, "weight", "December file")
    result["predicted_rate"] = numeric_series(result, "predicted_rate", "December file")

    if result["date"].duplicated().any():
        fail("duplicate dates in December file")

    if len(result) != 31 or set(result["date"]) != set(DECEMBER_DATES):
        fail("December file must contain all 31 dates from Dec 1–31")

    if not result["pickup"].eq(FIXED_PICKUP).all():
        fail(f"pickup must be {FIXED_PICKUP}")

    if not result["delivery"].eq(FIXED_DELIVERY).all():
        fail(f"delivery must be {FIXED_DELIVERY}")

    if not np.isclose(result["distance"], FIXED_DISTANCE).all():
        fail(f"distance must be {FIXED_DISTANCE}")

    if not result["equipment"].eq(FIXED_EQUIPMENT).all():
        fail(f"equipment must be {FIXED_EQUIPMENT}")

    if not np.isclose(result["weight"], FIXED_WEIGHT).all():
        fail(f"weight must be {FIXED_WEIGHT}")

    if (result["predicted_rate"] <= 0).any():
        fail("predicted_rate must be positive")

    return result.sort_values("date")


def save_december_chart(december: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.8), dpi=180)

    color = "#064A56"

    ax.plot(
        december["date"],
        december["predicted_rate"],
        color=color,
        linewidth=2.6,
        marker="o",
        markersize=3.2,
    )

    floor = float(december["predicted_rate"].min())

    ax.fill_between(
        december["date"],
        december["predicted_rate"],
        floor - max(10.0, floor * 0.02),
        color=color,
        alpha=0.08,
    )

    ax.set_title("December 2025 Predicted Load Rate", loc="left", fontsize=15, fontweight="bold")
    ax.set_ylabel("Predicted rate ($)")
    ax.grid(axis="y", linewidth=0.8)

    ax.tick_params(axis="x", rotation=35)

    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--december-predictions", required=True)
    parser.add_argument("--output-dir", default="scorer_results")

    args = parser.parse_args()

    validate_predictions(read_csv(Path(args.predictions), "predictions"))

    december = validate_december(
        read_csv(Path(args.december_predictions), "December file")
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chart_path = output_dir / "candidate_december.png"

    save_december_chart(december, chart_path)

    print("Validation successful ✅")
    print(f"Chart saved at: {chart_path}")


if __name__ == "__main__":
    main()