import os
import pickle
from utils.data import load_data
from src.models.ngram import NgramModel
from src.evaluation.evaluation_metrics import bleu, chrf
from src.utils.config import CONFIG
from .utils import save_results


def main():
    # load training test data from preprocessed files
    train_data, test_data = load_data(CONFIG)

    # train model
    n = CONFIG["ngram_n"]
    ngram_model = NgramModel(n)
    ngram_model.train(train_data)

    # metrics
    bleu_score = bleu(ngram_model, test_data)
    chrf_score = chrf(ngram_model, test_data)

    print("Test set scores:")
    print(f"BLEU: {bleu_score:.2f}")
    print(f"chrF: {chrf_score:.2f}")

    params = {
        "model": "ngram",
        "n": n,
        "train_size": len(train_data),
    }

    samples = [(src, ref, ngram_model.translate(src)) for src, ref in test_data[:5]]

    save_results(
        "results/ngram.csv",
        params,
        metrics={"bleu": bleu_score, "chrf": chrf_score},
        samples=samples,
    )

    # generate sample outputs on test data
    print("\nSample Translations:")
    for source, reference in test_data[:5]:
        translation = ngram_model.translate(source)
        print("Input:      ", source)
        print("Reference:  ", reference)
        print("Translation:", translation)
        print("-" * 50)

    # 7) Save final model to disk
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"ngram_{n}.pt")
    with open(model_path, "wb") as f:
        pickle.dump(ngram_model, f)
    print(f"Saved {n}‑gram model to {model_path}")


if __name__ == "__main__":
    main()
