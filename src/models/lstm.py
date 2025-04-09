import torch
import torch.nn as nn
import random


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
    def __init__(self, encoder, decoder):
        """
        Wrapper model that ties the encoder and decoder together.

        Args:
            encoder (nn.Module): The encoder model.
            decoder (nn.Module): The decoder model.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Forward pass through the sequence-to-sequence model using teacher forcing.

        Args:
            src (Tensor): Source tensor of shape [batch_size, src_len].
            trg (Tensor): Target tensor of shape [batch_size, trg_len].
            teacher_forcing_ratio (float): Probability of using teacher forcing.

        Returns:
            outputs (Tensor): Predicted outputs of shape [batch_size, trg_len, output_dim].
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.embedding.num_embeddings

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size)

        # Encode the source sequence
        hidden, cell = self.encoder(src)

        # First input to the decoder is the <sos> token (assumed to be at index 0)
        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output

            # Decide whether to use teacher forcing: feed the actual target as the next input, or use model's prediction.
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)  # Get the highest scoring token from predictions

            input = trg[:, t] if teacher_force else top1

        return outputs

    def translate(self, src_sentence, src_field, trg_field, max_len=50):
        """
        Translate a single sentence from source to target using greedy decoding.

        Args:
            src_sentence (str): Input sentence (standard text).
            src_field: Object with attributes: init_token, eos_token, and a vocab mapping (stoi/itos) for the source.
            trg_field: Object with attributes: init_token, eos_token, and a vocab mapping (stoi/itos) for the target.
            max_len (int): Maximum length for the generated sentence.

        Returns:
            str: The translated sentence in target language style.
        """
        self.eval()
        tokens = src_sentence.split()
        tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        src_indices = [src_field.vocab.stoi[token] for token in tokens]
        src_tensor = torch.LongTensor(src_indices).unsqueeze(0)

        with torch.no_grad():
            hidden, cell = self.encoder(src_tensor)

        trg_indices = [trg_field.vocab.stoi[trg_field.init_token]]
        for _ in range(max_len):
            trg_tensor = torch.LongTensor([trg_indices[-1]])
            with torch.no_grad():
                output, hidden, cell = self.decoder(trg_tensor, hidden, cell)
            pred_token = output.argmax(1).item()
            trg_indices.append(pred_token)
            if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
                break

        # Convert indices back to words and remove special tokens
        trg_tokens = [trg_field.vocab.itos[i] for i in trg_indices]
        return " ".join(trg_tokens[1:-1])


# test block
if __name__ == "__main__":
    # Define dummy vocabulary and field objects for testing.
    class DummyVocab:
        def __init__(self):
            # Simple mapping for demonstration
            self.stoi = {
                "<s>": 0,
                "</s>": 1,
                "hello": 2,
                "world": 3,
                "how": 4,
                "are": 5,
                "you": 6,
            }
            self.itos = {i: s for s, i in self.stoi.items()}

    class DummyField:
        def __init__(self):
            self.init_token = "<s>"
            self.eos_token = "</s>"
            self.vocab = DummyVocab()

    src_field = DummyField()
    trg_field = DummyField()

    # Hyperparameters for dummy model
    INPUT_DIM = len(src_field.vocab.stoi)
    OUTPUT_DIM = len(trg_field.vocab.stoi)
    EMB_DIM = 16
    HID_DIM = 32
    NUM_LAYERS = 1
    DROPOUT = 0.1

    # Instantiate encoder, decoder, and the seq2seq model
    encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT)
    decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT)
    model = LSTMModel(encoder, decoder)

    # Test the translate() method with a dummy input
    sample_input = "hello world"
    print("Input:", sample_input)
    translation = model.translate(sample_input, src_field, trg_field)
    print("Translation:", translation)
