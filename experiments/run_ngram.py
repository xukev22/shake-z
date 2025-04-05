from src.data.data_utils import load_data
from src.models.ngram import NgramModel
from src.evaluation.evaluation_metrics import compute_bleu
from src.utils.config import CONFIG


def main():
    # Load training, validation, and test data from preprocessed files
    train_data, val_data, test_data = load_data(CONFIG["data_path"])

    # Initialize the N-gram model (e.g., with n=3 for trigram)
    ngram_model = NgramModel(n=CONFIG["ngram_n"])
    ngram_model.train(train_data)

    # Evaluate using BLEU score on the validation set
    bleu_score = compute_bleu(ngram_model, val_data)
    print(f"N-gram Model BLEU Score (Validation): {bleu_score:.2f}")

    # Generate sample outputs on test data
    print("\nSample Translations:")
    for sentence in test_data[:5]:
        translation = ngram_model.translate(sentence)
        print("Input:      ", sentence)
        print("Translation:", translation)
        print("-" * 50)


if __name__ == "__main__":
    main()
