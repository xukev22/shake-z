import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from utils.data import load_data, create_dataloaders
from src.models.lstm import LSTMModel
from src.evaluation.evaluation_metrics import evaluate
from utils import save_results
from src.utils.config import CONFIG
from src.utils.utils import (
    tokenize_line,
    train_word2vec,
    create_embedder,
    save_word2vec,
)


def main():
    # 1) Load train/val/test splits
    train_pairs, val_pairs, test_pairs = load_data(CONFIG)

    # 2) Vocabularies
    src_sentences = [tokenize_line(src) for src, _ in train_pairs]
    tgt_sentences = [tokenize_line(tgt) for _, tgt in train_pairs]

    EMBED_SIZE = CONFIG["embed_size"]
    src_w2v = train_word2vec(src_sentences, EMBED_SIZE)
    tgt_w2v = train_word2vec(tgt_sentences, EMBED_SIZE)

    out_dir = CONFIG["out_path"]
    os.makedirs(out_dir, exist_ok=True)
    src_path = os.path.join(out_dir, f"src_embedding_{EMBED_SIZE}.model")
    save_word2vec(src_w2v, src_path)
    tgt_path = os.path.join(out_dir, f"tgt_embedding_{EMBED_SIZE}.model")
    save_word2vec(tgt_w2v, tgt_path)

    # 3) Create embedding layers and attach vocabulary info.
    src_embedder = create_embedder(src_w2v)
    tgt_embedder = create_embedder(tgt_w2v)

    # 4) Initialize model, optimizer, loss.
    model = LSTMModel(CONFIG, src_embedder, tgt_embedder)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_embedder.token_to_index["<pad>"])

    # 3) Collate function to batch & tokenize
    def collate_fn(batch):
        src_texts = [item["source"] for item in batch]
        tgt_texts = [item["target"] for item in batch]

        # Convert texts to tensors using our vocab mappings.
        def to_tensor(text, token_to_index):
            tokens = tokenize_line(text)
            ids = [token_to_index.get(tok, token_to_index["<pad>"]) for tok in tokens]
            if len(ids) < CONFIG["max_length"]:
                ids += [token_to_index["<pad>"]] * (CONFIG["max_length"] - len(ids))
            else:
                ids = ids[: CONFIG["max_length"]]
            return torch.tensor(ids, dtype=torch.long)

        src_tensors = torch.stack(
            [to_tensor(s, src_embedder.token_to_index) for s in src_texts]
        )
        tgt_tensors = torch.stack(
            [to_tensor(t, tgt_embedder.token_to_index) for t in tgt_texts]
        )
        return {"src": src_tensors, "trg": tgt_tensors}

    # 4) DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_pairs,
        val_pairs,
        test_pairs,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    # 6) Prepare results paths
    results_dir = CONFIG["out_path"]
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
            src_tensor = batch["src"]
            tgt_tensor = batch["trg"]
            optimizer.zero_grad()
            output = model(src_tensor, tgt_tensor)
            output = output[:, 1:].reshape(-1, len(tgt_embedder.token_to_index))
            target = tgt_tensor[:, 1:].reshape(-1)
            loss = criterion(output, target)
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
            "embed_size": CONFIG["embed_size"],
            "hidden_size": CONFIG["hidden_size"],
            "num_layers": CONFIG["num_layers"],
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
    final_dir = os.path.join(results_dir, "models")
    os.makedirs(final_dir, exist_ok=True)
    model_path = os.path.join(
        final_dir, f"lstm_{CONFIG['num_epochs']}e_{CONFIG['learning_rate']}lr.pt"
    )
    torch.save(model.state_dict(), model_path)
    print(f"Saved final LSTM model to {model_path}")


if __name__ == "__main__":
    main()
