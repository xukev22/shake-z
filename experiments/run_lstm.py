import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from src.data.data_utils import load_data, create_dataloaders
from src.models.lstm import LSTMModel
from src.evaluation.evaluation_metrics import evaluate
from utils import save_results
from src.utils.config import CONFIG


def main():
    # 1) Load train/val/test splits
    train_pairs, val_pairs, test_pairs = load_data(CONFIG)

    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["tokenizer"])

    # 3) Collate function to batch & tokenize
    def collate_fn(batch):
        src_texts = [ex["source"] for ex in batch]
        tgt_texts = [ex["target"] for ex in batch]
        enc = tokenizer(
            src_texts,
            padding="max_length",
            truncation=True,
            max_length=CONFIG["max_length"],
            return_tensors="pt",
        )
        dec = tokenizer(
            tgt_texts,
            padding="max_length",
            truncation=True,
            max_length=CONFIG["max_length"],
            return_tensors="pt",
        )
        # Use the original token ids for the decoder input
        decoder_input_ids = dec.input_ids.clone()

        # Create labels for the loss, replacing pad tokens with -100
        labels = dec.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            "input_ids": enc.input_ids,
            "attention_mask": enc.attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }

    # 4) DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_pairs,
        val_pairs,
        test_pairs,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    # 5) Model, optimizer, loss
    model = LSTMModel(CONFIG, tokenizer)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 6) Prepare results paths
    results_dir = CONFIG["results_path"]
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "lstm.csv")

    # 7) Training loop with progress bar
    for epoch in range(1, CONFIG["num_epochs"] + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{CONFIG['num_epochs']}",
            unit="batch",
        ):
            optimizer.zero_grad()
            output = model(batch["input_ids"], batch["decoder_input_ids"][:, :-1])
            loss = criterion(
                output.reshape(-1, output.size(-1)),
                batch["labels"][:, 1:].reshape(-1),
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            tqdm.write(f"  batch loss: {loss.item():.4f}", end="\r")

        avg_loss = total_loss / len(train_loader)

        # 8) Evaluate on validation set
        val_bleu, val_chrf = evaluate(model, val_pairs)

        print(
            f"\nEpoch {epoch}/{CONFIG['num_epochs']}  "
            f"Train Loss={avg_loss:.4f}  "
            f"BLEU={val_bleu:.2f}  chrF={val_chrf:.2f}"
        )

        # 9) Log to CSV
        samples = [(src, ref, model.translate(src)) for src, ref in val_pairs[:5]]
        params = {
            "model": "lstm",
            "epoch": epoch,
            "lr": CONFIG["learning_rate"],
            "batch_size": CONFIG["batch_size"],
            "max_length": CONFIG["max_length"],
        }
        metrics = {"bleu": val_bleu, "chrf": val_chrf}
        extras = {"train_loss": f"{avg_loss:.4f}"}

        save_results(
            csv_path=csv_path,
            params=params,
            metrics=metrics,
            samples=samples,
            extras=extras,
        )

    # 10) Sample translations on test set
    print("\nSample Translations (Test):")
    for src, ref in test_pairs[:5]:
        print("Input:      ", src)
        print("Reference:  ", ref)
        print("Translation:", model.translate(src))
        print("-" * 50)

    # 11) Save final trained model
    final_dir = os.path.join(results_dir, "final_models")
    os.makedirs(final_dir, exist_ok=True)
    model_path = os.path.join(final_dir, "lstm.pt")
    torch.save(model, model_path)
    print(f"Saved final LSTM model to {model_path}")


if __name__ == "__main__":
    main()
