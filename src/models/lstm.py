import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Encoder(nn.Module):
    def __init__(self, embedder: nn.Embedding, hid_dim, num_layers, dropout):
        super().__init__()
        self.embedding = embedder  # Use pretrained or randomly initialized embedder
        self.lstm = nn.LSTM(
            embedder.embedding_dim,
            hid_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, src):
        embedded = self.embedding(src)  # [B, src_len, embed_dim]
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(
        self, embedder: nn.Embedding, hid_dim, num_layers, dropout, vocab_size
    ):
        super().__init__()
        self.embedding = embedder
        self.lstm = nn.LSTM(
            embedder.embedding_dim,
            hid_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hid_dim, vocab_size)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)  # [B,1]
        embedded = self.embedding(input)  # [B, 1, embed_dim]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))  # [B, vocab_size]
        return prediction, hidden, cell


class LSTMModel(nn.Module):
    def __init__(self, config, src_embedder: nn.Embedding, tgt_embedder: nn.Embedding):
        super().__init__()
        self.max_length = config["max_length"]
        self.src_token_to_index = src_embedder.token_to_index
        self.src_index_to_token = src_embedder.index_to_token
        self.tgt_token_to_index = tgt_embedder.token_to_index
        self.tgt_index_to_token = tgt_embedder.index_to_token

        self.src_vocab_size = len(self.src_token_to_index)
        self.tgt_vocab_size = len(self.tgt_token_to_index)
        # Set special token ids from the vocab dictionary.
        self.pad_idx = self.tgt_index_to_token.get("<pad>", 0)
        self.bos_idx = self.tgt_index_to_token.get("<bos>", 1)
        self.eos_idx = self.tgt_index_to_token.get("<eos>", 2)

        hid_dim = config["hidden_size"]
        num_layers = config["num_layers"]
        dropout = config["dropout"]

        self.encoder = Encoder(src_embedder, hid_dim, num_layers, dropout)
        self.decoder = Decoder(
            tgt_embedder, hid_dim, num_layers, dropout, self.tgt_vocab_size
        )

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        B, trg_len = trg.size()
        outputs = torch.zeros(B, trg_len, self.tgt_vocab_size)
        hidden, cell = self.encoder(src)
        input = trg[:, 0]  # Start with BOS token

        for t in range(1, trg_len):
            pred, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = pred
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = pred.argmax(1)
            input = trg[:, t] if teacher_force else top1
        return outputs

    def translate_batch(self, src_batch, max_len=None):
        max_len = max_len or self.max_length
        self.eval()
        with torch.no_grad():
            hidden, cell = self.encoder(src_batch)
        B = src_batch.size(0)
        outputs = torch.full((B, max_len), self.pad_idx, dtype=torch.long)
        inputs = torch.full((B,), self.bos_idx, dtype=torch.long)
        for t in range(max_len):
            logits, hidden, cell = self.decoder(inputs, hidden, cell)
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            outputs[:, t] = next_tokens
            inputs = next_tokens
            if (next_tokens == self.eos_idx).all():
                break
        results = []
        for seq in outputs.tolist():
            toks = []
            for tok in seq:
                if tok == self.eos_idx:
                    break
                toks.append(self.tgt_index_to_token.get(tok, ""))
            results.append(" ".join(toks))
        return results

    def translate(self, src_text, max_len=None):
        # A simple space-split tokenization for demonstration (modify as needed)
        tokens = src_text.lower().split()
        src_ids = [self.src_token_to_index.get(tok, self.pad_idx) for tok in tokens]
        max_len = max_len or self.max_length
        if len(src_ids) < max_len:
            src_ids += [self.src_token_to_index.get("<pad>", 0)] * (
                max_len - len(src_ids)
            )
        else:
            src_ids = src_ids[:max_len]
        src_tensor = torch.tensor([src_ids], dtype=torch.long)
        return self.translate_batch(src_tensor, max_len)[0]


# test block
if __name__ == "__main__":
    # Dummy vocabularies: these would normally be created from your dataset.
    # For demonstration, we create a simple vocabulary for source and target.
    src_vocab = ["<pad>", "<bos>", "<eos>", "the", "man", "is", "sufficient"]
    tgt_vocab = ["<pad>", "<bos>", "<eos>", "the", "guy", "is", "for", "real", "enough"]

    # Create mappings.
    src_token_to_index = {token: idx for idx, token in enumerate(src_vocab)}
    src_index_to_token = {idx: token for token, idx in src_token_to_index.items()}
    tgt_token_to_index = {token: idx for idx, token in enumerate(tgt_vocab)}
    tgt_index_to_token = {idx: token for token, idx in tgt_token_to_index.items()}

    # Create dummy embedding layers and attach the vocab info.
    embed_size = 16  # small dimension for demonstration
    src_embedder = torch.nn.Embedding(len(src_vocab), embed_size)
    tgt_embedder = torch.nn.Embedding(len(tgt_vocab), embed_size)

    # Attach vocabulary mappings to the embedders.
    src_embedder.token_to_index = src_token_to_index
    src_embedder.index_to_token = src_index_to_token
    tgt_embedder.token_to_index = tgt_token_to_index
    tgt_embedder.index_to_token = tgt_index_to_token

    # Define a simple configuration.
    config = {
        "max_length": 10,
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.0,
    }

    # Initialize the model.
    model = LSTMModel(config, src_embedder, tgt_embedder)

    # Prepare dummy input and target tensors.
    # For the source, we'll assume tokens: "<bos> the man is sufficient <eos>".
    # For the target, "<bos> the guy is for real enough <eos>".
    # We'll pad sequences to 'max_length' (10 tokens).
    def prepare_tensor(text, token_to_index, max_length):
        tokens = text.lower().split()
        # For target sequences we assume tokens already include <bos> and <eos>.
        ids = [token_to_index.get(tok, token_to_index["<pad>"]) for tok in tokens]
        if len(ids) < max_length:
            ids = ids + [token_to_index["<pad>"]] * (max_length - len(ids))
        else:
            ids = ids[:max_length]
        return torch.tensor(ids, dtype=torch.long)

    src_text = "<bos> the man is sufficient <eos>"
    tgt_text = "<bos> the guy is for real enough <eos>"
    src_tensor = prepare_tensor(
        src_text, src_token_to_index, config["max_length"]
    ).unsqueeze(0)
    tgt_tensor = prepare_tensor(
        tgt_text, tgt_token_to_index, config["max_length"]
    ).unsqueeze(0)

    # Run a forward pass with teacher forcing.
    output = model(src_tensor, tgt_tensor, teacher_forcing_ratio=1.0)
    print("Forward output shape:", output.shape)
    # Should print: [1, max_length, len(tgt_vocab)]

    # Test the translation method using the source text.
    translation = model.translate("the man is sufficient", max_len=config["max_length"])
    print("Translation:", translation)
