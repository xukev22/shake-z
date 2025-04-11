import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from sacrebleu.metrics import CHRF

# Uncomment the following line if you haven't already downloaded the required tokenizer data
# nltk.download('punkt')


def bleu(model, dataset, smoothing_function=None):
    """
    Compute the corpus BLEU score for a given model on a dataset.

    Args:
        model: A model with a translate() method that takes a source sentence as input
               and returns a generated translation.
        dataset: A list of tuples in the form (source_sentence, reference_translation).
        smoothing_function: (Optional) A smoothing function from nltk.translate.bleu_score.SmoothingFunction.

    Returns:
        bleu_score: A float representing the corpus BLEU score.
    """
    references = []
    hypotheses = []

    for source, reference in dataset:
        # Generate model output
        hypothesis = model.translate(source)

        # Tokenize both the model's output and the reference translation
        hyp_tokens = nltk.word_tokenize(hypothesis.lower())
        ref_tokens = nltk.word_tokenize(reference.lower())

        hypotheses.append(hyp_tokens)
        references.append([ref_tokens])

    bleu_score = corpus_bleu(
        references, hypotheses, smoothing_function=smoothing_function
    )
    return bleu_score


def chrf(model, dataset):
    """
    Returns the corpus-level chrF score.
    """
    chrf = CHRF()
    hyps = []
    refs = []
    for src, ref in dataset:
        hyps.append(model.translate(src))
        refs.append([ref])
    return chrf.corpus_score(hyps, refs).score


if __name__ == "__main__":
    # Sample usage with a dummy model and dataset for testing purposes
    class DummyModel:
        def translate(self, text):
            # Dummy translation: return the input text unchanged
            return text

    # Create a dummy model instance
    dummy_model = DummyModel()

    # Dummy dataset: list of (source, reference) sentence pairs
    dummy_dataset = [
        ("This is a test.", "This is a test."),
        ("Another example sentence.", "Another example sentence."),
    ]

    # Using smoothing function from NLTK's SmoothingFunction (e.g., method1)
    smoothing_fn = SmoothingFunction().method1

    # Compute and print BLEU score with the smoothing function
    score = compute_bleu(dummy_model, dummy_dataset, smoothing_function=smoothing_fn)
    print("BLEU Score with smoothing:", score)
