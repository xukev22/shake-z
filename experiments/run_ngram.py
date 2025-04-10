import re
import nltk
from src.models.ngram import NgramModel
from src.evaluation.evaluation_metrics import compute_bleu
from src.utils.config import CONFIG
from src.utils.utils import (
    train_test_split_data,
    genz_sonnet_to_arr,
    shake_sonnet_to_arr,
)
from nltk.translate.bleu_score import SmoothingFunction
import string

# Uncomment the following line if you haven't already downloaded the required tokenizer data
# nltk.download('punkt')


def post_cleanup(text: str) -> str:
    """
    Collapse any repeated punctuation sequences (with or without spaces)
    down to a single character, and remove spaces before punctuation.

    Examples:
        "Hello !!!??  world .. ."  -> "Hello !? world ."
        "Wait , , what ? ?"        -> "Wait, what?"
    """
    # 1. Collapse runs of the *same* punctuation (e.g. "!!!" -> "!")
    text = re.sub(
        r"([{}])(?:\s*\1)+".format(re.escape(string.punctuation)), r"\1", text
    )

    # 2. Collapse runs of *mixed* punctuation (e.g. "!?!?!" -> "!?")
    #    by finding any sequence of 2+ punctuation (possibly with spaces),
    #    stripping internal spaces, then keeping unique in order.
    def _mixed_collapse(match):
        seq = re.sub(r"\s+", "", match.group(0))  # remove spaces
        seen = set()
        out = []
        for ch in seq:
            if ch not in seen:
                out.append(ch)
                seen.add(ch)
        return "".join(out)

    text = re.sub(
        r"(?:[{}]|\s){{2,}}".format(re.escape(string.punctuation)),
        _mixed_collapse,
        text,
    )

    # 3. Remove spaces before punctuation
    text = re.sub(r"\s+([{}])".format(re.escape(string.punctuation)), r"\1", text)

    # 4. Collapse any leftover multi-spaces
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def tokenize_and_clean(text: str) -> str:
    """
    Tokenizes the input text using NLTK and then cleans it.
    This ensures consistent formatting for both inputs and outputs.
    """
    tokens = nltk.word_tokenize(text)
    # Rejoin tokens into a single string with a standard space separator
    tokenized_text = " ".join(tokens)
    return post_cleanup(tokenized_text)


def main():
    # Get raw data from the sonnet functions
    shake = shake_sonnet_to_arr(CONFIG["raw_data_path"])
    genz = genz_sonnet_to_arr(CONFIG["data_path"])
    train_data, test_data = train_test_split_data(shake, genz, test_ratio=0.3, seed=42)

    # Preprocess the training and testing data by tokenizing and cleaning both the source and reference.
    processed_train_data = [
        (tokenize_and_clean(source), tokenize_and_clean(reference))
        for source, reference in train_data
    ]
    processed_test_data = [
        (tokenize_and_clean(source), tokenize_and_clean(reference))
        for source, reference in test_data
    ]

    # Initialize the N-gram model (e.g., n=3 for a trigram model)
    ngram_model = NgramModel(n=CONFIG["ngram_n"])
    # Train the model on the preprocessed training data
    ngram_model.train(processed_train_data)

    # Use smoothing to help with n-gram overlap issues (if desired)
    smoothing_fn = SmoothingFunction().method1
    bleu_score = compute_bleu(
        ngram_model, processed_test_data, smoothing_function=smoothing_fn
    )
    print(f"N-gram Model BLEU Score (TEST): {bleu_score:.2f}")

    # Generate sample translations on a subset of the test data.
    print("\nSample Translations:")
    for source, reference in processed_test_data[:5]:
        # Generate model output using the preprocessed (tokenized) source sentence
        translation = ngram_model.translate(source)
        # Tokenize and clean the model's output for consistency
        translation_clean = tokenize_and_clean(translation)
        print("Input:      ", source)
        print("Reference:  ", reference)
        print("Translation:", translation_clean)
        print("-" * 50)


if __name__ == "__main__":
    main()
