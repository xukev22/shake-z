import torch
import torch.nn as nn
from gensim.models import Word2Vec
import nltk


# A simple tokenizer function (you already have one in your provided code)
def tokenize_line(line: str, bos="<bos>", eos="<eos>"):

    inner_pieces = nltk.word_tokenize(line.lower())
    tokens = [bos] + inner_pieces + [eos]
    return tokens


def train_word2vec(
    data: list[list[str]],
    embeddings_size: int,
    window: int = 5,
    min_count: int = 1,
    sg: int = 1,
) -> Word2Vec:
    """
    Create new word embeddings based on our data.

    Params:
        data: The corpus
        embeddings_size: The dimensions in each embedding

    Returns:
        A gensim Word2Vec model
    """

    return Word2Vec(
        sentences=data,
        vector_size=embeddings_size,
        window=window,
        min_count=min_count,
        sg=sg,
    )


def create_embedder(word2vec_model: Word2Vec) -> torch.nn.Embedding:
    """
    Create a PyTorch embedding layer based on our data.

    We will *first* train a Word2Vec model on our data.
    Then, we'll use these weights to create a PyTorch embedding layer.
        `nn.Embedding.from_pretrained(weights)`


    PyTorch docs: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding.from_pretrained
    Gensim Word2Vec docs: https://radimrehurek.com/gensim/models/word2vec.html

    Pay particular attention to the *types* of the weights and the types required by PyTorch.

    Params:
        data: The corpus
        embeddings_size: The dimensions in each embedding

    Returns:
        A PyTorch embedding layer
    """
    # Get vocabulary for the pretrained embeddings from Word2Vec.
    # We use the Word2Vec model's vocabulary for initializing weights.
    wv = word2vec_model.wv
    vocab = wv.index_to_key

    # Build token-to-index and index-to-token mappings.
    token_to_index = {token: idx for idx, token in enumerate(vocab)}
    index_to_token = {idx: token for token, idx in token_to_index.items()}

    # Create a weight tensor from the pretrained embeddings.
    weight_tensor = torch.FloatTensor(wv[vocab])

    # Add special token "<pad>" if missing.
    special_tokens = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
    for token, desired_id in special_tokens.items():
        if token not in token_to_index:
            # Append token to vocabulary.
            new_idx = len(token_to_index)
            token_to_index[token] = new_idx
            index_to_token[new_idx] = token
            # Append a new row (e.g. zeros) for the token.
            pad_vector = torch.zeros(word2vec_model.vector_size)
            weight_tensor = torch.cat([weight_tensor, pad_vector.unsqueeze(0)], dim=0)

    embedder = nn.Embedding.from_pretrained(weight_tensor)

    # Attach our custom vocabulary mappings to the embedder.
    embedder.token_to_index = token_to_index
    embedder.index_to_token = index_to_token
    embedder.vocab_size = len(token_to_index)
    embedder.pad_token_id = token_to_index.get("<pad>", 0)
    embedder.bos_token_id = token_to_index.get("<bos>", 1)
    embedder.eos_token_id = token_to_index.get("<eos>", 2)

    return embedder


def save_word2vec(embeddings: Word2Vec, filename: str) -> None:
    """
    Saves weights of trained gensim Word2Vec model to a file.

    Params:
        obj: The object.
        filename: The destination file.
    """
    embeddings.save(filename)


def load_word2vec(filename: str) -> Word2Vec:
    """
    Loads weights of trained gensim Word2Vec model from a file.

    Params:
        filename: The saved model file.
    """
    return Word2Vec.load(filename)
