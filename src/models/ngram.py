import random
import re
from collections import defaultdict, Counter
import string


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
        # Unigram counts for back-off and smoothing.
        self.unigram_counts = Counter()
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
            for w in tokens:
                if w not in ("<s>", "</s>"):
                    self.unigram_counts[w] += 1
                    self.vocab.add(w)

            # Train N-gram counts
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i : i + self.n - 1])
                word = tokens[i + self.n - 1]
                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1

        # For each source word, choose the target word with the highest count.
        self.mapping = {}
        for source, t_counts in mapping_counts.items():
            # remove punctuation candidates
            for p in list(t_counts):
                if all(ch in string.punctuation for ch in p):
                    del t_counts[p]
            if t_counts:
                self.mapping[source] = max(t_counts, key=t_counts.get)

    def _smoothed_prob(self, word, context):
        """
        Compute Laplace-smoothed probability P(word | context).

        (count(context→word) + 1) / (count(context) + |V|)
        """
        V = len(self.vocab)
        count_w = self.ngram_counts[context][word]
        total = self.context_counts[context]
        return (count_w + 1) / (total + V)

    def generate_word(self, context):
        """
        Given a context, sample the next word from the trained N-gram distribution.

        Args:
            context (Tuple[str]): The (n-1)-word context.

        Returns:
            str: The next word sampled from the model.
        """
        # 1) Full n‑gram
        if context in self.context_counts and self.context_counts[context] > 0:
            choices, probs = zip(
                *[(w, self._smoothed_prob(w, context)) for w in self.vocab]
            )
            return random.choices(choices, probs)[0]

        # 2) Back-off to (n−1)-gram
        if len(context) > 1:
            back_ctx = context[1:]
            if back_ctx in self.context_counts and self.context_counts[back_ctx] > 0:
                choices, probs = zip(
                    *[(w, self._smoothed_prob(w, back_ctx)) for w in self.vocab]
                )
                return random.choices(choices, probs)[0]

        # 3) Unigram fallback (add‑one smoothing)
        total_uni = sum(self.unigram_counts.values())
        V = len(self.vocab)
        words, counts = zip(*self.unigram_counts.items())
        probs = [(c + 1) / (total_uni + V) for c in counts]
        return random.choices(words, probs)[0]

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
        context = ["<s>"] * (self.n - 1)

        for word in source_tokens:
            if word in self.mapping:
                target_word = self.mapping[word]
            else:
                target_word = self.generate_word(tuple(context))

            # avoid repeating punctuation
            if target_word.strip() in string.punctuation:
                if translated_tokens and translated_tokens[-1] == target_word:
                    continue

            translated_tokens.append(target_word)
            context = context[1:] + [target_word]

        final_tokens = [t for t in translated_tokens if t not in ("</s>", "<s>")]
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
