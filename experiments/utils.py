import torch
import os
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


def save_model(
    model,
    save_dir,
    model_name,
    optimizer: torch.optim.Optimizer = None,
    epoch: int = None,
    extra: dict = None,
):
    """
    Save a PyTorch model's state_dict (and optionally optimizer) to disk.

    Args:
        model:        nn.Module to save.
        save_dir:     directory where the .pt file will be written.
        model_name:   base name for the file (e.g. "lstm", "transformer").
        optimizer:    (optional) optimizer whose state_dict to save.
        epoch:        (optional) epoch number, appended to filename.
        extra:        (optional) dict of any extra scalars to include.
    Returns:
        filepath (str) of the saved checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)

    # build filename
    name = model_name
    if epoch is not None:
        name += f"_epoch{epoch}"
    filename = name + ".pt"
    path = os.path.join(save_dir, filename)

    # gather state
    state = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        state["epoch"] = epoch
    if extra:
        state.update(extra)

    # save
    torch.save(state, path)
    return path
