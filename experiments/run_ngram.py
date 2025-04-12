import os
import torch
from src.data.data_utils import load_data
from src.models.ngram import NgramModel
from src.evaluation.evaluation_metrics import evaluate
from src.utils.config import CONFIG
from utils import save_results


def main():
    # 1) Load data
    train_data, val_data, test_data = load_data(CONFIG)

    # 2) Train n‑gram model
    n = CONFIG["ngram_n"]
    ngram_model = NgramModel(n)
    ngram_model.train(train_data)

    # 3) Evaluate on validation set
    val_bleu, val_chrf = evaluate(ngram_model, val_data)

    print(f"n‑gram (n={n})  BLEU={val_bleu:.2f}  chrF={val_chrf:.2f}")

    # 4) Prepare results directory & CSV path
    results_dir = CONFIG["results_path"]
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "ngram.csv")

    # 5) Log validation results
    samples = [(src, ref, ngram_model.translate(src)) for src, ref in val_data[:5]]
    params = {
        "model": "ngram",
        "n": n,
        "train_size": len(train_data),
    }
    metrics = {"bleu": val_bleu, "chrf": val_chrf}
    extras = {}

    save_results(
        csv_path=csv_path,
        params=params,
        metrics=metrics,
        samples=samples,
        extras=extras,
    )

    # 6) Sample translations on test set
    print("\nSample Translations (Test):")
    for src, ref in test_data[:5]:
        print("Input:      ", src)
        print("Reference:  ", ref)
        print("Translation:", ngram_model.translate(src))
        print("-" * 50)

    # 7) Save final model to disk
    models_dir = os.path.join(results_dir, "models/")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"ngram_{n}.pt")
    torch.save(ngram_model, model_path)
    print(f"Saved {n}‑gram model to {model_path}")


if __name__ == "__main__":
    main()
