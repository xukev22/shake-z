import nltk
from nltk.translate.bleu_score import corpus_bleu
from sacrebleu.metrics import CHRF

# Uncomment the following line if you haven't already downloaded the required tokenizer data
# nltk.download('punkt')


def bleu(model, dataset):
    """
    Compute the corpus BLEU score for a given model on a dataset.

    Args:
        model: A model with a translate() method that takes a source sentence as input
               and returns a generated translation.
        dataset: A list of tuples in the form (source_sentence, reference_translation).

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

    bleu_score = corpus_bleu(references, hypotheses)
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


def evaluate(model, dataset, verbose=False):
    """
    Evaluate a model or translations by computing both BLEU and chrF scores.

    This function is a convenient wrapper that calls the existing `bleu` and `chrf`
    functions, passing along any additional keyword arguments.
    """

    # Call the existing bleu and chrf functions with the provided parameters
    bleu_score = bleu(model, dataset)
    chrf_score = chrf(model, dataset)
    if verbose:
        print("Model Evaluation:")
        print(f"  BLEU:      {bleu_score:.2f}")
        print(f"  chrF:      {chrf_score:.2f}")

    return bleu_score, chrf_score


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

    # Compute and print BLEU score with the smoothing function
    score = evaluate(dummy_model, dummy_dataset)
    print("BLEU/chrF scores:", score)
