import torch
import torch.nn as nn
import random
from transformers import AutoTokenizer


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, num_layers, dropout):
        """
        Encoder using LSTM.

        Args:
            input_dim (int): Size of the source vocabulary.
            emb_dim (int): Embedding dimension.
            hid_dim (int): Hidden state dimension.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim, hid_dim, num_layers=num_layers, dropout=dropout, batch_first=True
        )

    def forward(self, src):
        # src shape: [batch_size, src_len]
        embedded = self.embedding(src)  # [batch_size, src_len, emb_dim]
        outputs, (hidden, cell) = self.lstm(embedded)
        # We return the final hidden and cell states to be used by the decoder.
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, num_layers, dropout):
        """
        Decoder using LSTM.

        Args:
            output_dim (int): Size of the target vocabulary.
            emb_dim (int): Embedding dimension.
            hid_dim (int): Hidden state dimension.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim, hid_dim, num_layers=num_layers, dropout=dropout, batch_first=True
        )
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden, cell):
        # input shape: [batch_size] -> we add a time dimension
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.embedding(input)  # [batch_size, 1, emb_dim]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # output shape: [batch_size, 1, hid_dim] -> squeeze to [batch_size, hid_dim]
        prediction = self.fc_out(output.squeeze(1))  # [batch_size, output_dim]
        return prediction, hidden, cell


class LSTMModel(nn.Module):
    def __init__(self, config, tokenizer):
        """
        Args:
          config: dict with hyperparams (emb_dim, hid_dim, num_layers, dropout, device, max_length)
          tokenizer: a HuggingFace tokenizer, from which we derive vocab_size and special token IDs
        """
        super().__init__()
        self.max_length = config["max_length"]
        self.tokenizer = tokenizer

        # derive vocab & special IDs
        vocab_size = tokenizer.vocab_size
        self.pad_idx = tokenizer.pad_token_id
        # some tokenizers don't define bos/eos, so fall back to cls/sep
        self.bos_idx = tokenizer.bos_token_id or tokenizer.cls_token_id
        self.eos_idx = tokenizer.eos_token_id or tokenizer.sep_token_id

        # build encoder & decoder with the derived dims
        self.encoder = Encoder(
            input_dim=vocab_size,
            emb_dim=config["embed_size"],
            hid_dim=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        )
        self.decoder = Decoder(
            output_dim=vocab_size,
            emb_dim=config["embed_size"],
            hid_dim=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        )

    def forward(self, src, trg):
        """
        Forward pass without teacher forcing.
        The decoder always uses its own predictions as inputs.

        Args:
            src: Tensor of shape [B, src_len]
            trg: Tensor of shape [B, trg_len] (only used to determine sequence length)

        Returns:
            outputs: Tensor of shape [B, trg_len, vocab_size]
        """
        B, trg_len = trg.size()
        vocab_size = self.decoder.embedding.num_embeddings
        outputs = torch.zeros(B, trg_len, vocab_size, device=src.device)

        hidden, cell = self.encoder(src)

        # Start the decoding using the BOS token from the target sequence.
        input = trg[:, 0]
        for t in range(1, trg_len):
            pred, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = pred
            # Always use model's own prediction.
            input = pred.argmax(1)
        return outputs

    def translate_batch(self, src_batch, max_len=50):
        """
        Greedy decode a batch of tokenized inputs.

        Args:
            src_batch: Tensor of shape [B, src_len]
            max_len (int): Maximum length of the generated sequence.

        Returns:
            A list of decoded strings.
        """
        self.eval()
        with torch.no_grad():
            hidden, cell = self.encoder(src_batch)

        B = src_batch.size(0)
        outputs = torch.full(
            (B, max_len), self.pad_idx, dtype=torch.long, device=src_batch.device
        )
        outputs[:, 0] = self.bos_idx  # Initialize with the BOS token.

        for t in range(1, max_len):
            input = outputs[:, t - 1]
            with torch.no_grad():
                pred, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = pred.argmax(1)

        results = []
        for seq in outputs.tolist():
            tokens = []
            for token in seq:
                if token == self.eos_idx:
                    break
                tokens.append(token)
            decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
            results.append(decoded)
        return results

    def translate(self, src_text, max_len=50):
        """
        Translate a single text sequence.

        Args:
            src_text (str): Input text.
            max_len (int): Maximum length of the generated sequence.

        Returns:
            A decoded string translation.
        """
        enc = self.tokenizer(
            src_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        src_ids = enc.input_ids
        translations = self.translate_batch(src_ids, max_len=max_len)
        return translations[0]


# Test block to verify that the model works as expected.
if __name__ == "__main__":
    # Dummy configuration.
    config = {
        "max_length": 10,
        "embed_size": 32,
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.1,
        "tokenizer": "bert-base-uncased",
    }

    # Load the BERT tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])

    # Initialize the model.
    model = LSTMModel(config, tokenizer)

    # Sample source and target texts.
    src_text = "Hello world!"
    tgt_text = "Bonjour le monde!"

    # Tokenize the texts.
    enc = tokenizer(
        src_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config["max_length"],
    )
    dec = tokenizer(
        tgt_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config["max_length"],
    )

    src_ids = enc.input_ids  # [1, max_length]
    tgt_ids = dec.input_ids  # [1, max_length]

    # Test the forward pass without teacher forcing.
    outputs = model(src_ids, tgt_ids)
    print("Forward output shape:", outputs.shape)
    # Expected shape: [batch_size, trg_len, vocab_size]

    # Test the translation method (greedy decoding).
    translation = model.translate(src_text, max_len=config["max_length"])
    print("Translation:", translation)
