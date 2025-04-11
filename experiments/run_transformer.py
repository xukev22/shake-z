import torch
from src.data.data_utils import load_data, create_dataloaders
from src.models.transformer import TransformerModel
from src.evaluation.evaluation_metrics import bleu, chrf
from src.utils.config import CONFIG
from transformers import get_linear_schedule_with_warmup
from utils import save_results


def main():

    # Initialize the Transformer model (fine-tuning a pre-trained model)
    model = TransformerModel(CONFIG)
    tokenizer = model.tokenizer

    # Load dataset and create DataLoader objects
    train_pairs, val_pairs, test_pairs = load_data(CONFIG)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_pairs,
        val_pairs,
        test_pairs,
        tokenizer=tokenizer,
        max_length=CONFIG["transformer_max_length"],
        batch_size=CONFIG["batch_size"],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG["warmup_steps"],
        num_training_steps=len(train_loader) * CONFIG["num_epochs"],
    )

    # Training loop
    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model(batch)  # Assume the model returns a loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} - Training Loss: {avg_loss:.4f}")

        samples = [(src, ref, model.translate(src)) for src, ref in val_pairs[:5]]
        # Evaluate on validation set
        model.eval()
        bleu_score = bleu(model, val_pairs)
        chrf_score = chrf(model, val_pairs)
        print(f"Epoch {epoch+1} - Validation BLEU Score: {bleu_score:.2f}")
        print(f"Epoch {epoch+1} - Validation chrF Score: {chrf_score:.2f}")

        params = {
            "model": "transformer",
            "pretrained": CONFIG["pretrained_model_name"],
            "epoch": epoch + 1,
            "lr": CONFIG["learning_rate"],
            "batch_size": CONFIG["batch_size"],
            "max_length": CONFIG["transformer_max_length"],
        }

        save_results(
            "results/transformer.csv",
            params=params,
            metrics={"bleu": bleu_score, "chrf": chrf_score},
            samples=samples,
            extras={
                "train_loss": avg_loss,
            },
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


if __name__ == "__main__":
    main()
