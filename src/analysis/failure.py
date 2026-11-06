import pandas as pd

from src.data.datasets import prepare_display_image
from src.utils.io import ensure_dir
from src.utils.plots import save_montage


def annotate_errors(predictions, threshold=0.5):
    out = predictions.copy()
    if "pred_label" not in out.columns:
        out["pred_label"] = (out["prob"] >= threshold).astype(int)
    if "error_type" not in out.columns:
        out["error_type"] = "tn"
        out.loc[(out["label"] == 0) & (out["pred_label"] == 1), "error_type"] = "fp"
        out.loc[(out["label"] == 1) & (out["pred_label"] == 0), "error_type"] = "fn"
        out.loc[(out["label"] == 1) & (out["pred_label"] == 1), "error_type"] = "tp"
    return out


def top_failures(predictions, error_type, top_k=25):
    predictions = annotate_errors(predictions)
    subset = predictions[predictions["error_type"] == error_type].copy()
    if error_type == "fp":
        subset = subset.sort_values("prob", ascending=False)
    elif error_type == "fn":
        subset = subset.sort_values("prob", ascending=True)
    return subset.head(top_k).reset_index(drop=True)


def _titles_from_rows(df):
    titles = []
    for _, row in df.iterrows():
        titles.append(
            f"{row.get('organelle_label', '')} | p={row.get('prob', 0):.3f} | {row.get('source', '')}"
        )
    return titles


def save_failure_montages(predictions, out_dir, top_k=25):
    out_dir = ensure_dir(out_dir)
    predictions = annotate_errors(predictions)

    for error_type, filename in [("fp", "false_positives.png"), ("fn", "false_negatives.png")]:
        subset = top_failures(predictions, error_type, top_k=top_k)
        images = [prepare_display_image(path) for path in subset["image_path"].tolist()]
        titles = _titles_from_rows(subset)
        save_montage(images, titles, out_dir / filename, cols=5)
        subset.to_csv(out_dir / filename.replace(".png", ".csv"), index=False)


def group_error_buckets(predictions, fields=None):
    fields = fields or ["source", "acquisition_batch", "quality_flags", "organelle_label"]
    predictions = annotate_errors(predictions)
    predictions["is_error"] = (predictions["label"] != predictions["pred_label"]).astype(int)
    predictions["is_fp"] = (predictions["error_type"] == "fp").astype(int)
    predictions["is_fn"] = (predictions["error_type"] == "fn").astype(int)

    frames = []
    for field in fields:
        if field not in predictions.columns:
            continue

        subset = predictions.copy()
        if field == "quality_flags":
            subset[field] = subset[field].fillna("").replace("", "none")
            subset[field] = subset[field].astype(str).str.split(r"[;,|]")
            subset = subset.explode(field)
            subset[field] = subset[field].astype(str).str.strip().replace("", "none")
        else:
            subset[field] = subset[field].fillna("").replace("", "unknown")

        summary = (
            subset.groupby(field, dropna=False)
            .agg(
                total=("label", "size"),
                errors=("is_error", "sum"),
                false_positives=("is_fp", "sum"),
                false_negatives=("is_fn", "sum"),
                mean_prob=("prob", "mean"),
            )
            .reset_index()
        )
        summary["field"] = field
        summary["error_rate"] = summary["errors"] / summary["total"].clip(lower=1)
        frames.append(summary.rename(columns={field: "bucket"}))

    if not frames:
        return pd.DataFrame(
            columns=["bucket", "total", "errors", "false_positives", "false_negatives", "mean_prob", "field", "error_rate"]
        )

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["error_rate", "errors"], ascending=[False, False]).reset_index(drop=True)
    return out


def save_failure_report(predictions, out_dir, top_k=25, fields=None):
    out_dir = ensure_dir(out_dir)
    predictions = annotate_errors(predictions)
    predictions.to_csv(out_dir / "predictions_with_errors.csv", index=False)
    save_failure_montages(predictions, out_dir, top_k=top_k)
    summary = group_error_buckets(predictions, fields=fields)
    summary.to_csv(out_dir / "error_buckets.csv", index=False)
    return summary
