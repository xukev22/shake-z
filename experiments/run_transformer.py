import torch
from src.data.data_utils import load_data, create_dataloaders
from src.models.transformer import TransformerModel
from src.evaluation.evaluation_metrics import compute_bleu
from src.utils.config import CONFIG
from transformers import get_linear_schedule_with_warmup


def main():
    # Load dataset and create DataLoader objects
    train_data, val_data, test_data = load_data(CONFIG["data_path"])
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, batch_size=CONFIG["batch_size"]
    )

    # Initialize the Transformer model (fine-tuning a pre-trained model)
    model = TransformerModel(CONFIG)
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

        # Evaluate on validation set
        model.eval()
        bleu_score = compute_bleu(model, val_loader)
        print(f"Epoch {epoch+1} - Validation BLEU Score: {bleu_score:.2f}")

    # Generate sample outputs on test data
    model.eval()
    print("\nSample Translations:")
    for sentence in test_data[:5]:
        translation = model.translate(sentence)
        print("Input:      ", sentence)
        print("Translation:", translation)
        print("-" * 50)


if __name__ == "__main__":
    main()
