import re
from src.data.data_utils import load_data
from src.models.ngram import NgramModel
from src.evaluation.evaluation_metrics import bleu, chrf
from src.utils.config import CONFIG
from nltk.translate.bleu_score import SmoothingFunction
from utils import save_results


def main():
    # Load training, validation, and test data from preprocessed files
    train_data, val_data, test_data = load_data(CONFIG)

    # Training model
    n = CONFIG["ngram_n"]
    ngram_model = NgramModel(n)
    ngram_model.train(train_data)

    smoothing_fn = SmoothingFunction().method1
    samples = [(src, ref, ngram_model.translate(src)) for src, ref in val_data[:5]]

    # --- Validation metrics ---
    bleu_score = bleu(ngram_model, val_data, smoothing_function=smoothing_fn)
    chrf_score = chrf(ngram_model, val_data)

    print("Validation set scores:")
    print(f"  BLEU:      {bleu_score:.2f}")
    print(f"  chrF:      {chrf_score:.2f}")

    params = {
        "model": "ngram",
        "n": n,
        "train_size": len(train_data),
    }

    save_results(
        "results/ngram.csv",
        params,
        metrics={"bleu": bleu_score, "chrf": chrf_score},
        samples=samples,
    )

    # Generate sample outputs on test data
    print("\nSample Translations:")
    for source, reference in test_data[:5]:
        a = ngram_model.translate(source)
        translation = re.sub(r"[']", "", a)
        print("Input:      ", source)
        print("Reference:  ", reference)
        print("Translation:", translation)
        print("-" * 50)


if __name__ == "__main__":
    main()
