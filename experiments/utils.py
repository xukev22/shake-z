import csv
import time


def save_results(csv_path, params, metrics, samples, extras=None):
    """
    Append experiment results to a CSV, logging multiple metrics.

    Args:
        csv_path (str): Path to the CSV file.
        params (dict): Hyperparameters & metadata for this run.
                       Must include a 'model' key.
        metrics (dict): Mapping metric_name -> float (e.g. {'bleu':0.12, 'chrf':0.45}).
        samples (List[Tuple[str, str, str]]): (input, reference, prediction) triplets.
    """
    extras = extras or {}
    # Build header: timestamp, model, <other params...>, <metric names...>, sample_input/ref/out
    fieldnames = (
        ["timestamp", "model"]
        + [k for k in params.keys() if k != "model"]
        + list(metrics.keys())
        + list(extras.keys())
        + ["sample_input", "sample_ref", "sample_out"]
    )

    # Check if file exists (to write header once)
    file_exists = False
    try:
        with open(csv_path, "r", encoding="utf8"):
            file_exists = True
    except FileNotFoundError:
        pass

    with open(csv_path, "a", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(fieldnames)

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        model_name = params.get("model", "")

        # For each sample, write a row with all metrics
        for inp, ref, out in samples:
            row = [ts, model_name]
            # other params in insertion order (minus 'model')
            row += [v for k, v in params.items() if k != "model"]
            # metrics in the order of metrics.keys()
            row += [f"{metrics[m]:.4f}" for m in metrics]
            # extras in the order of extras.keys()
            row += [extras[k] for k in extras]
            row += [inp, ref, out]
            writer.writerow(row)
