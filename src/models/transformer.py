import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer


class TransformerModel(nn.Module):
    def __init__(self, config):
        """
        Initializes the Transformer model by loading a pre-trained T5 model and tokenizer.

        Args:
            config (dict): Configuration parameters including device, model name, and max length.
        """
        super(TransformerModel, self).__init__()
        self.config = config
        self.model_name = config["pretrained_model_name"]

        # Load pre-trained T5 model and its tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

    def forward(self, batch):
        """
        Forward pass for fine-tuning the transformer model.

        Expects a dictionary 'batch' with:
          - "input_ids": Tensor [batch_size, seq_len]
          - "attention_mask": Tensor [batch_size, seq_len]
          - "labels": Tensor [batch_size, target_seq_len]

        Returns:
            loss: Computed loss for the batch.
        """
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return outputs.loss

    def translate(self, input_text, max_length=50):
        """
        Translate a single input sentence using the fine-tuned transformer model.

        Args:
            input_text (str): The input standard English sentence.
            max_length (int): Maximum length of the generated output.

        Returns:
            str: The generated Gen Z style sentence.
        """
        # Tokenize the input text. Adjust truncation/max_length as needed.
        input_ids = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.get("transformer_max_length", 50),
        )

        # Generate translation using beam search for better quality
        generated_ids = self.model.generate(
            input_ids, max_length=max_length, num_beams=5, early_stopping=True
        )
        # Decode the generated tokens into a string, skipping special tokens
        output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return output


# test block
if __name__ == "__main__":
    # Define a basic config for testing purposes
    config = {
        "pretrained_model_name": "t5-small",
        "transformer_max_length": 50,
    }
    model = TransformerModel(config)
    sample_input = "Translate: How are you doing today?"
    print("Input:", sample_input)
    print("Translation:", model.translate(sample_input))
