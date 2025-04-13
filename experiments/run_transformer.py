import os
import torch
from src.utils.data import load_data, create_dataloaders
from src.models.transformer import TransformerModel
from src.utils.evaluation import evaluate
from src.utils.config import CONFIG
from transformers import get_linear_schedule_with_warmup
from utils import save_results
from tqdm.auto import tqdm


def main():

    # Load data
    train_pairs, test_pairs = load_data(CONFIG)

    # Initialize Model + Tokenizer
    model = TransformerModel(CONFIG)
    tokenizer = model.tokenizer

    def collate_fn(batch):
        sources = [item["source"] for item in batch]
        targets = [item["target"] for item in batch]

        # Encode the inputs
        enc = tokenizer(
            sources,
            padding="max_length",
            truncation=True,
            max_length=CONFIG["max_length"],
            return_tensors="pt",
        )
        # Encode the targets
        dec = tokenizer(
            targets,
            padding="max_length",
            truncation=True,
            max_length=CONFIG["max_length"],
            return_tensors="pt",
        )
        # Prepare labels, masking out pad tokens
        labels = dec.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            "input_ids": enc.input_ids,
            "attention_mask": enc.attention_mask,
            "labels": labels,
        }

    # Build DataLoaders
    train_loader, test_loader = create_dataloaders(
        train_pairs,
        test_pairs,
        batch_size=CONFIG["batch_size"],
        collate_fn=collate_fn,
    )

    # Optimizer + Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG["warmup_steps"],
        num_training_steps=len(train_loader) * CONFIG["num_epochs"],
    )

    # Prepare results directory & CSV
    results_dir = CONFIG["out_path"]
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "transformer.csv")

    # Training loop
    for epoch in range(1, CONFIG["num_epochs"] + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch}/{CONFIG['num_epochs']}", unit="batch"
        ):
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            tqdm.write(f"  batch loss: {loss.item():.4f}", end="\r")
        avg_loss = total_loss / len(train_loader)

        # Evaluate on validation set
        val_bleu, val_chrf = evaluate(model, test_pairs)

        print(
            f"\tTrain Loss={avg_loss:.4f}  " f"BLEU={val_bleu:.2f}  chrF={val_chrf:.2f}"
        )

        # Log to CSV
        samples = [(src, ref, model.translate(src)) for src, ref in test_pairs[:5]]
        params = {
            "model": "transformer",
            "pretrained": CONFIG["pretrained_model_name"],
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

    # Generate sample outputs on test data
    model.eval()
    print("\nSample Translations:")
    for source, reference in test_pairs[:5]:
        translation = model.translate(source)
        print("Input:      ", source)
        print("Reference:  ", reference)
        print("Translation:", translation)
        print("-" * 50)

    # Save trained model
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(
        models_dir,
        f"{CONFIG['pretrained_model_name']}_{CONFIG['num_epochs']}e_{CONFIG['learning_rate']}lr.pt",
    )
    torch.save(model.state_dict(), model_path)
    print(f"Saved final {CONFIG['pretrained_model_name']} model to {model_path}")


if __name__ == "__main__":
    main()
