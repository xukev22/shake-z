import random
import re
from collections import defaultdict, Counter


class NgramModel:
    def __init__(self, n=3):
        """
        Initialize the N-gram model.

        Args:
            n (int): The order of the n-gram model (e.g., 3 for trigram).
        """
        self.n = n
        # Dictionary mapping (n-1)-word contexts to counts of next words.
        self.ngram_counts = defaultdict(Counter)
        # Count of each (n-1)-word context overall.
        self.context_counts = defaultdict(int)
        # Vocabulary for the target sentences.
        self.vocab = set()
        # A simple mapping from source words to target words for direct substitution.
        self.mapping = {}

    def tokenize(self, text):
        """
        Tokenize input text into a list of words and punctuation.

        Args:
            text (str): The text to tokenize.

        Returns:
            List[str]: A list of tokens.
        """
        # A simple regex-based tokenizer (can be replaced with a more robust one if needed).
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    def train(self, data):
        """
        Train the N-gram model and build a word mapping from source to target.

        Args:
            data (List[Tuple[str, str]]): A list of (source_sentence, target_sentence) pairs.
        """
        mapping_counts = defaultdict(Counter)
        for source, target in data:
            source_tokens = self.tokenize(source.lower())
            target_tokens = self.tokenize(target.lower())

            # Build a simple word mapping: count co-occurrences between source and target tokens.
            for s_word, t_word in zip(source_tokens, target_tokens):
                mapping_counts[s_word][t_word] += 1

            # Train the N-gram model on the target sentence.
            # Add start and end tokens to help with context.
            tokens = ["<s>"] * (self.n - 1) + target_tokens + ["</s>"]
            self.vocab.update(tokens)
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i : i + self.n - 1])
                word = tokens[i + self.n - 1]
                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1

        # For each source word, choose the target word with the highest count.
        self.mapping = {
            s: max(t_counts, key=t_counts.get) for s, t_counts in mapping_counts.items()
        }

    def generate_word(self, context):
        """
        Given a context, sample the next word from the trained N-gram distribution.

        Args:
            context (Tuple[str]): The (n-1)-word context.

        Returns:
            str: The next word sampled from the model.
        """
        if context not in self.ngram_counts:
            return "</s>"
        choices, counts = zip(*self.ngram_counts[context].items())
        total = sum(counts)
        probabilities = [count / total for count in counts]
        return random.choices(choices, probabilities)[0]

    def translate(self, source_sentence):
        """
        Translate a source sentence into Gen Z style.

        This method first attempts to substitute each source token with a target token
        using the learned mapping. If a word is missing, it falls back to generating a word
        using the N-gram model based on the current context.

        Args:
            source_sentence (str): The source sentence in standard English.

        Returns:
            str: The translated Gen Z style sentence.
        """
        source_tokens = self.tokenize(source_sentence.lower())
        translated_tokens = []
        # Initialize context with start tokens.
        context = ["<s>"] * (self.n - 1)
        for word in source_tokens:
            # Direct substitution if available.
            if word in self.mapping:
                target_word = self.mapping[word]
            else:
                # Otherwise, generate the next word based on the current context.
                target_word = self.generate_word(tuple(context))
            translated_tokens.append(target_word)
            # Update the context by sliding one word forward.
            context = context[1:] + [target_word]
        # Remove any end-of-sentence tokens before returning.
        final_tokens = [
            token for token in translated_tokens if token not in ["</s>", "<s>"]
        ]
        return " ".join(final_tokens)


# Quick test for the N-gram model (for development purposes)
if __name__ == "__main__":
    # Dummy dataset: list of (source, target) pairs.
    dummy_data = [
        ("hello world", "yo world"),
        ("how are you", "how u doin"),
        ("good morning", "mornin fam"),
    ]

    model = NgramModel(n=3)
    model.train(dummy_data)

    test_sentence = "hello how are you"
    print("Input:", test_sentence)
    print("Translation:", model.translate(test_sentence))
